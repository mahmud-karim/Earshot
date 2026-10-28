// Adapted from speech-swift's F5-TTS port (Sources/F5TTS/F5TTSFlow.swift and F5TTSPrimitives.swift,
// Apache License 2.0). Changes: Raon's sizes (24 heads of 64 on a 1408 wide model, so attention is wider
// than the model), separate head size, fused Q/K/V, MLXFast layer norm, RoPE and attention, half-precision
// compute, the text side kept in float32, and the conditioning half of the input projection computed once
// per sentence instead of once per step.

import Foundation
import MLX
import MLXNN

/// The numbers in config.yaml's `model.arch` that change the network.
struct RaonDiTConfig {
    var dim = 1408
    var depth = 28
    var heads = 24
    var headDim = 64
    var ffMult = 4
    var textDim = 512
    var convLayers = 4
    var melDim = 80

    /// Reads the flat `key: value` lines of config.yaml and refuses options this port does not implement.
    init(yaml: String) throws {
        var values: [String: String] = [:]
        for line in yaml.components(separatedBy: .newlines) {
            let parts = line.split(separator: ":", maxSplits: 1).map { $0.trimmingCharacters(in: .whitespaces) }
            if parts.count == 2, !parts[1].isEmpty { values[parts[0]] = parts[1] }
        }
        func int(_ key: String, _ fallback: Int) -> Int { values[key].flatMap { Int($0) } ?? fallback }
        dim = int("dim", dim)
        depth = int("depth", depth)
        heads = int("heads", heads)
        ffMult = int("ff_mult", ffMult)
        textDim = int("text_dim", textDim)
        convLayers = int("conv_layers", convLayers)
        melDim = int("n_mel_channels", melDim)
        let off: Set<String> = ["null", "None", "~", "False", "false"]
        for key in ["qk_norm", "pe_attn_head", "logit_softcapping", "post_norm"] {
            if let value = values[key], !off.contains(value) {
                throw LocalVoiceError("Raon option \(key): \(value) is not supported.")
            }
        }
        if let type = values["mel_spec_type"], type != "sbhifigan16k" {
            throw LocalVoiceError("Raon mel type \(type) is not supported.")
        }
    }
}

/// A linear layer stored as PyTorch stores it, `[out, in]`.
private struct Projection {
    let weight: MLXArray
    let bias: MLXArray?

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if let bias { return addMM(bias, x, weight.T) }
        return matmul(x, weight.T)
    }
}

/// F5-TTS's DiT backbone with Raon's settings: ConvNeXt text encoder, convolutional position embedding,
/// adaLN-zero blocks with rotary self-attention, and a final adaptive layer norm.
final class RaonDiT {
    let config: RaonDiTConfig
    /// Half precision for the transformer. The official CUDA inference runs it in float16 too.
    let dtype: DType

    private struct Block {
        let modulation: Projection
        let qkv: Projection
        let out: Projection
        let ff1: Projection
        let ff2: Projection
    }

    private struct TextBlock {
        let dwconvWeight: MLXArray // [dim, 7, 1]
        let dwconvBias: MLXArray
        let normWeight: MLXArray
        let normBias: MLXArray
        let pw1: Projection
        let grnGamma: MLXArray
        let grnBeta: MLXArray
        let pw2: Projection
    }

    private let textTable: MLXArray
    private let textBlocks: [TextBlock]
    private let textPositions: MLXArray // [4096, textDim]
    private let timeMLP1: Projection
    private let timeMLP2: Projection
    private let inputNoise: Projection // the noisy-mel columns of input_embed.proj
    private let inputContext: Projection // the reference-mel and text columns, with the bias
    private let posConv1: (MLXArray, MLXArray)
    private let posConv2: (MLXArray, MLXArray)
    private let blocks: [Block]
    private let normOut: Projection
    private let projOut: Projection

    /// Rows in the text embedding, the filler row included.
    var textRows: Int { textTable.dim(0) }

    init(weights raw: [String: MLXArray], config: RaonDiTConfig, dtype: DType) throws {
        self.config = config
        self.dtype = dtype
        func tensor(_ key: String, _ type: DType? = nil) throws -> MLXArray {
            guard let value = raw["transformer.\(key)"] else { throw LocalVoiceError("Raon weights have no \(key).") }
            return value.asType(type ?? dtype)
        }
        func linear(_ key: String, _ type: DType? = nil) throws -> Projection {
            Projection(weight: try tensor("\(key).weight", type), bias: try tensor("\(key).bias", type))
        }

        let projWeight = try tensor("input_embed.proj.weight")
        let inner = config.heads * config.headDim
        guard projWeight.shape == [config.dim, 2 * config.melDim + config.textDim],
              try tensor("transformer_blocks.0.attn.to_q.weight").shape == [inner, config.dim] else {
            throw LocalVoiceError("Raon weights do not match config.yaml.")
        }

        textTable = try tensor("text_embed.text_embed.weight", .float32)
        textBlocks = try (0 ..< config.convLayers).map { i in
            let p = "text_embed.text_blocks.\(i)"
            return TextBlock(
                dwconvWeight: try tensor("\(p).dwconv.weight", .float32).transposed(0, 2, 1),
                dwconvBias: try tensor("\(p).dwconv.bias", .float32),
                normWeight: try tensor("\(p).norm.weight", .float32),
                normBias: try tensor("\(p).norm.bias", .float32),
                pw1: try linear("\(p).pwconv1", .float32),
                grnGamma: try tensor("\(p).grn.gamma", .float32),
                grnBeta: try tensor("\(p).grn.beta", .float32),
                pw2: try linear("\(p).pwconv2", .float32))
        }
        // precompute_freqs_cis(text_dim, 4096): cos then sin of position x 10000^(-2i/d).
        let half = config.textDim / 2
        let frequencies = 1 / pow(MLXArray(Float(10000)), MLXArray(0 ..< Int32(half)).asType(.float32) * 2 / Float(config.textDim))
        let angles = MLXArray(0 ..< Int32(4096)).asType(.float32).expandedDimensions(axis: 1) * frequencies.expandedDimensions(axis: 0)
        textPositions = concatenated([cos(angles), sin(angles)], axis: -1)

        timeMLP1 = try linear("time_embed.time_mlp.0")
        timeMLP2 = try linear("time_embed.time_mlp.2")
        inputNoise = Projection(weight: projWeight[0..., 0 ..< config.melDim], bias: nil)
        inputContext = Projection(weight: projWeight[0..., config.melDim...], bias: try tensor("input_embed.proj.bias"))
        posConv1 = (try tensor("input_embed.conv_pos_embed.conv1d.0.weight").transposed(0, 2, 1),
                    try tensor("input_embed.conv_pos_embed.conv1d.0.bias"))
        posConv2 = (try tensor("input_embed.conv_pos_embed.conv1d.2.weight").transposed(0, 2, 1),
                    try tensor("input_embed.conv_pos_embed.conv1d.2.bias"))
        blocks = try (0 ..< config.depth).map { i in
            let p = "transformer_blocks.\(i)"
            let qkv = Projection(
                weight: concatenated([try tensor("\(p).attn.to_q.weight"), try tensor("\(p).attn.to_k.weight"), try tensor("\(p).attn.to_v.weight")], axis: 0),
                bias: concatenated([try tensor("\(p).attn.to_q.bias"), try tensor("\(p).attn.to_k.bias"), try tensor("\(p).attn.to_v.bias")], axis: 0))
            return Block(modulation: try linear("\(p).attn_norm.linear"), qkv: qkv, out: try linear("\(p).attn.to_out.0"),
                         ff1: try linear("\(p).ff.ff.0.0"), ff2: try linear("\(p).ff.ff.2"))
        }
        normOut = try linear("norm_out.linear")
        projOut = try linear("proj_out")
    }

    /// Every weight, to evaluate at load time so the file is read once and the bf16 originals freed.
    var arrays: [MLXArray] {
        var all = [textTable, textPositions, posConv1.0, posConv1.1, posConv2.0, posConv2.1]
        for l in [timeMLP1, timeMLP2, inputNoise, inputContext, normOut, projOut] { all += [l.weight] + (l.bias.map { [$0] } ?? []) }
        for b in textBlocks {
            all += [b.dwconvWeight, b.dwconvBias, b.normWeight, b.normBias, b.pw1.weight, b.pw1.bias!, b.grnGamma, b.grnBeta, b.pw2.weight, b.pw2.bias!]
        }
        for b in blocks {
            for l in [b.modulation, b.qkv, b.out, b.ff1, b.ff2] { all += [l.weight, l.bias!] }
        }
        return all
    }

    /// `TextEmbedding.forward`: `[1, frames, textDim]` float32 for token ids already cut or padded by the caller.
    /// With `drop`, every token reads as filler (the unconditional pass) but the padding mask stays.
    func embedText(_ tokens: [Int32], frames: Int, drop: Bool) -> MLXArray {
        let ids = (Array(tokens.prefix(frames).map { $0 + 1 }) + [Int32](repeating: 0, count: max(0, frames - tokens.count)))
        let mask = MLXArray(ids.map { $0 == 0 ? Float(0) : Float(1) }, [1, frames, 1])
        let lookup = drop ? [Int32](repeating: 0, count: frames) : ids
        var h = take(textTable, MLXArray(lookup, [1, frames]), axis: 0)
        h = (h + textPositions[0 ..< min(frames, 4096)].expandedDimensions(axis: 0)) * mask
        for block in textBlocks {
            var y = conv1d(h, block.dwconvWeight, padding: 3, groups: config.textDim) + block.dwconvBias
            y = MLXFast.layerNorm(y, weight: block.normWeight, bias: block.normBias, eps: 1e-6)
            y = gelu(block.pw1(y))
            // Global response normalisation: L2 over time, divided by its mean over channels.
            let gx = sqrt((y * y).sum(axis: 1, keepDims: true))
            let nx = gx / (gx.mean(axis: -1, keepDims: true) + 1e-6)
            y = block.grnGamma * (y * nx) + block.grnBeta + y
            h = (h + block.pw2(y)) * mask
        }
        return h
    }

    /// The reference-mel and text part of the input projection for the conditional and unconditional
    /// rows, `[2, frames, dim]`: fixed for a whole sentence, so it is computed once.
    func context(referenceMel: MLXArray, text: MLXArray, nullText: MLXArray) -> MLXArray {
        let conditional = concatenated([referenceMel, text], axis: -1)
        let unconditional = concatenated([zeros(like: referenceMel), nullText], axis: -1)
        let both = concatenated([conditional, unconditional], axis: 0).asType(dtype)
        return inputContext(both)
    }

    /// The flow for noisy mel `x` (`[1, frames, mel]`) at time `t`, conditional and unconditional rows.
    func velocity(_ x: MLXArray, time t: Float, context: MLXArray) -> MLXArray {
        let batch = context.dim(0), frames = context.dim(1)
        // Every adaptive norm reads SiLU of the time embedding, so it is taken once.
        let time = silu(timeMLP2(silu(timeMLP1(Self.timeFeatures(t).asType(dtype)))))

        var h = inputNoise(x.asType(dtype)) + context
        var p = mish(conv1d(h, posConv1.0, padding: 15, groups: 16) + posConv1.1)
        p = mish(conv1d(p, posConv2.0, padding: 15, groups: 16) + posConv2.1)
        h = h + p

        let scale = 1 / Float(config.headDim).squareRoot()
        for block in blocks {
            let m = block.modulation(time).split(parts: 6, axis: -1).map { $0.expandedDimensions(axis: 1) }
            let (shiftAttn, scaleAttn, gateAttn, shiftFF, scaleFF, gateFF) = (m[0], m[1], m[2], m[3], m[4], m[5])

            let normed = MLXFast.layerNorm(h, eps: 1e-6) * (1 + scaleAttn) + shiftAttn
            let qkv = block.qkv(normed).reshaped(batch, frames, 3, config.heads, config.headDim).transposed(2, 0, 3, 1, 4)
            // x-transformers rotary embedding rotates interleaved pairs, which is MLX's "traditional" RoPE.
            let q = MLXFast.RoPE(qkv[0], dimensions: config.headDim, traditional: true, base: 10000, scale: 1, offset: 0)
            let k = MLXFast.RoPE(qkv[1], dimensions: config.headDim, traditional: true, base: 10000, scale: 1, offset: 0)
            let attended = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: qkv[2], scale: scale,
                mask: MLXFast.ScaledDotProductAttentionMaskMode.none)
            h = h + gateAttn * block.out(attended.transposed(0, 2, 1, 3).reshaped(batch, frames, -1))

            let normedFF = MLXFast.layerNorm(h, eps: 1e-6) * (1 + scaleFF) + shiftFF
            h = h + gateFF * block.ff2(geluApproximate(block.ff1(normedFF)))
        }

        let final = normOut(time).split(parts: 2, axis: -1).map { $0.expandedDimensions(axis: 1) }
        h = MLXFast.layerNorm(h, eps: 1e-6) * (1 + final[0]) + final[1]
        return projOut(h)
    }

    /// `SinusPositionEmbedding(256)` of `t` scaled by 1000: `[1, 256]`.
    private static func timeFeatures(_ t: Float) -> MLXArray {
        let half = 128
        let step = log(Float(10000)) / Float(half - 1)
        let frequencies = exp(MLXArray(0 ..< Int32(half)).asType(.float32) * -step)
        let args = (1000 * t) * frequencies
        return concatenated([sin(args), cos(args)], axis: -1).expandedDimensions(axis: 0)
    }
}
