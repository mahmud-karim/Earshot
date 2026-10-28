// Adapted from mlx-audio-swift (MIT, Prince Canuma and contributors), Sources/MLXAudioTTS/Models/Audio8/
// Audio8Codec.swift at fe2c182. The 0.1B ships the same 44.1 kHz codec as the 0.6B, so this is that
// code under Audio8Mini names (the upstream types are internal to MLXAudioTTS).

import Foundation
import MLX
import MLXNN

@inline(__always) private func audio8MiniUnary(_ m: Module, _ x: MLXArray) -> MLXArray {
    (m as! UnaryLayer)(x)
}

final class Audio8MiniCodecTransformer: Module, UnaryLayer {
    let layers: [Audio8MiniBlock]
    let norm: Audio8MiniNorm
    let window: Int, headDim: Int
    init(layers: Int, dim: Int, heads: Int, kvHeads: Int? = nil, intermediate: Int, window: Int) {
        self.window = window; headDim = 64
        self.layers = (0..<layers).map { _ in Audio8MiniBlock(dim: dim, intermediate: intermediate,
            heads: heads, kvHeads: kvHeads ?? heads, headDim: 64, layerScale: true) }
        norm = Audio8MiniNorm(dim, eps: 1e-5)
    }
    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let t = input.shape[1]
        let q = MLX.arange(t).expandedDimensions(axis: 1)
        let k = MLX.arange(t).expandedDimensions(axis: 0)
        let mask = (k .<= q) .&& (k .>= maximum(q - Int32(window - 1), 0))
        let rope = Audio8MiniRoPE(length: t, dim: headDim, base: 10000)
        var x = input
        for layer in layers { x = layer(x, rope: rope, mask: .array(mask.reshaped([1, 1, t, t]))) }
        return norm(x)
    }
}

final class Audio8MiniCausalConv: Module, UnaryLayer {
    let conv: Conv1d
    let stride: Int, kernel: Int, left: Int
    init(_ input: Int, _ output: Int, kernel: Int, stride: Int = 1, dilation: Int = 1, groups: Int = 1) {
        conv = Conv1d(inputChannels: input, outputChannels: output, kernelSize: kernel,
                      stride: stride, dilation: dilation, groups: groups)
        self.stride = stride; self.kernel = (kernel - 1) * dilation + 1
        left = self.kernel - stride
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let t = x.shape[1]
        let frames = Int(ceil(Double(t - kernel + left) / Double(stride) + 1))
        let right = (frames - 1) * stride + kernel - left - t
        return conv(padded(x, widths: [0, .init((left, right)), 0]))
    }
}

final class Audio8MiniCausalTranspose: Module, UnaryLayer {
    let conv: ConvTransposed1d
    let crop: Int
    init(_ input: Int, _ output: Int, kernel: Int, stride: Int) {
        conv = ConvTransposed1d(inputChannels: input, outputChannels: output, kernelSize: kernel, stride: stride)
        crop = kernel - stride
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = conv(x)
        return crop == 0 ? y : y[0..., 0..<(y.shape[1] - crop), 0...]
    }
}

final class Audio8MiniSnake: Module, UnaryLayer {
    var alpha: MLXArray
    init(_ channels: Int) { alpha = .ones([1, 1, channels]) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { x + square(sin(alpha * x)) / (alpha + 1e-9) }
}

final class Audio8MiniResidual: Module, UnaryLayer {
    let block: [Module]
    init(_ dim: Int, dilation: Int) {
        block = [Audio8MiniSnake(dim), Audio8MiniCausalConv(dim, dim, kernel: 7, dilation: dilation),
                 Audio8MiniSnake(dim), Audio8MiniCausalConv(dim, dim, kernel: 1)]
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { x + block.reduce(x) { audio8MiniUnary($1, $0) } }
}

final class Audio8MiniCodecEncoderBlock: Module, UnaryLayer {
    let block: [Module]
    init(_ dim: Int, stride: Int, transformer: Int) {
        var b: [Module] = [Audio8MiniResidual(dim / 2, dilation: 1), Audio8MiniResidual(dim / 2, dilation: 3),
            Audio8MiniResidual(dim / 2, dilation: 9), Audio8MiniSnake(dim / 2),
            Audio8MiniCausalConv(dim / 2, dim, kernel: 2 * stride, stride: stride)]
        if transformer > 0 { b.append(Audio8MiniCodecTransformer(layers: transformer, dim: dim,
                    heads: dim / 64, intermediate: dim * 3, window: 512)) }
        block = b
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { block.reduce(x) { audio8MiniUnary($1, $0) } }

    /// The block up to its closing transformer, if it has one: causal convolutions only.
    func convolutions(_ x: MLXArray) -> MLXArray {
        block.filter { !($0 is Audio8MiniCodecTransformer) }.reduce(x) { audio8MiniUnary($1, $0) }
    }
    func transformer(_ x: MLXArray) -> MLXArray {
        block.compactMap { $0 as? Audio8MiniCodecTransformer }.reduce(x) { $1($0) }
    }
}

final class Audio8MiniCodecEncoder: Module, UnaryLayer {
    let block: [Module]
    override init() {
        var b: [Module] = [Audio8MiniCausalConv(1, 64, kernel: 7)]
        var dim = 64
        for (stride, count) in zip([2, 4, 8, 8], [0, 0, 0, 4]) {
            dim *= 2; b.append(Audio8MiniCodecEncoderBlock(dim, stride: stride, transformer: count))
        }
        b += [Audio8MiniSnake(dim), Audio8MiniCausalConv(dim, 1024, kernel: 3)]
        block = b
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { block.reduce(x) { audio8MiniUnary($1, $0) } }

    /// Input samples per output step: the product of the block strides.
    static let hop = 2 * 4 * 8 * 8

    /// Encodes `x` (1, samples, 1), a whole number of steps long, running the full-rate convolutions a
    /// chunk of output steps at a time. They are causal and reach about 6,400 samples (13 steps) back,
    /// so `context` earlier steps per chunk reproduce the whole-sequence result. Only the last block's
    /// transformer looks further, and it runs over the whole, low-rate sequence afterwards.
    func chunked(_ x: MLXArray, chunk: Int = 128, context: Int = 16) -> MLXArray {
        let steps = x.dim(1) / Self.hop
        guard steps > chunk else { return self(x) }
        let blocks = block.compactMap { $0 as? Audio8MiniCodecEncoderBlock }
        func convolutions(_ x: MLXArray) -> MLXArray {
            blocks.reduce(audio8MiniUnary(block[0], x)) { $1.convolutions($0) }
        }
        var pieces: [MLXArray] = []
        for start in stride(from: 0, to: steps, by: chunk) {
            let from = max(0, start - context), end = min(start + chunk, steps)
            let piece = convolutions(x[0..., (from * Self.hop)..<(end * Self.hop), 0...])[0..., (start - from)..., 0...]
            // Evaluate now so this chunk's full-rate intermediates are freed before the next one.
            eval(piece)
            pieces.append(piece)
        }
        let encoded = blocks.reduce(concatenated(pieces, axis: 1)) { $1.transformer($0) }
        return block.dropFirst(1 + blocks.count).reduce(encoded) { audio8MiniUnary($1, $0) }
    }
}

final class Audio8MiniCodecDecoderBlock: Module, UnaryLayer {
    let block: [Module]
    init(_ input: Int, _ output: Int, stride: Int) {
        block = [Audio8MiniSnake(input), Audio8MiniCausalTranspose(input, output, kernel: 2 * stride, stride: stride),
            Audio8MiniResidual(output, dilation: 1), Audio8MiniResidual(output, dilation: 3), Audio8MiniResidual(output, dilation: 9)]
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { block.reduce(x) { audio8MiniUnary($1, $0) } }
}

final class Audio8MiniCodecDecoder: Module, UnaryLayer {
    let model: [Module]
    override init() {
        var b: [Module] = [Audio8MiniCausalConv(1024, 1536, kernel: 7)]
        for (i, stride) in [8, 8, 4, 2].enumerated() {
            b.append(Audio8MiniCodecDecoderBlock(1536 / (1 << i), 1536 / (1 << (i + 1)), stride: stride))
        }
        b += [Audio8MiniSnake(96), Audio8MiniCausalConv(96, 1, kernel: 7)]
        model = b
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { tanh(model.reduce(x) { audio8MiniUnary($1, $0) }) }

    /// Output samples per latent step: the product of the block strides.
    static let hop = 8 * 8 * 4 * 2

    /// Decodes `x` (1, steps, 1024) a chunk of latent steps at a time. Every layer here is causal and
    /// together they see about 19 steps back, so giving each chunk `context` earlier steps and dropping
    /// their samples reproduces the whole-sequence decode, while the large high-rate activations stay
    /// the size of one chunk however long the sentence is.
    func chunked(_ x: MLXArray, chunk: Int = 128, context: Int = 32) -> MLXArray {
        let steps = x.dim(1)
        guard steps > chunk else { return self(x) }
        var pieces: [MLXArray] = []
        for start in stride(from: 0, to: steps, by: chunk) {
            let from = max(0, start - context), end = min(start + chunk, steps)
            let piece = self(x[0..., from..<end, 0...])[0..., ((start - from) * Self.hop)..., 0...]
            // Evaluate now so this chunk's intermediates are freed before the next one is built.
            eval(piece)
            pieces.append(piece)
        }
        return concatenated(pieces, axis: 1)
    }
}

final class Audio8MiniVectorQuantizer: Module {
    @ModuleInfo(key: "in_proj") var inProj: Conv1d
    @ModuleInfo(key: "out_proj") var outProj: Conv1d
    let codebook: Embedding
    let size: Int
    init(size: Int) {
        self.size = size
        _inProj.wrappedValue = Conv1d(inputChannels: 1024, outputChannels: 8, kernelSize: 1)
        _outProj.wrappedValue = Conv1d(inputChannels: 8, outputChannels: 1024, kernelSize: 1)
        codebook = Embedding(embeddingCount: size, dimensions: 8)
    }
    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let p = inProj(x)
        let flat = p.reshaped([-1, 8])
        let f = flat / maximum(sqrt(sum(square(flat), axis: -1, keepDims: true)), 1e-12)
        let w = codebook.weight
        let c = w / maximum(sqrt(sum(square(w), axis: -1, keepDims: true)), 1e-12)
        let distance = sum(square(f), axis: -1, keepDims: true) - 2 * matmul(f, c.T)
            + sum(square(c), axis: -1, keepDims: true).T
        let ids = argMax(-distance, axis: -1).reshaped([x.shape[0], x.shape[1]])
        return (outProj(codebook(ids)), ids)
    }
    func decode(_ ids: MLXArray) -> MLXArray { outProj(codebook(clip(ids, min: 0, max: size - 1))) }
}

final class Audio8MiniResidualQuantizer: Module {
    let quantizers: [Audio8MiniVectorQuantizer]
    init(count: Int, size: Int) { quantizers = (0..<count).map { _ in Audio8MiniVectorQuantizer(size: size) } }
    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        var residual = x, total = MLXArray.zeros(x.shape, dtype: x.dtype)
        var codes = [MLXArray]()
        for q in quantizers { let (y, c) = q(residual); total = total + y; residual = residual - y; codes.append(c) }
        return (total, stacked(codes, axis: 1))
    }
    func decode(_ ids: MLXArray) -> MLXArray {
        let values = quantizers.enumerated().map { $0.element.decode(ids[0..., $0.offset, 0...]) }
        return values.dropFirst().reduce(values[0], +)
    }
}

final class Audio8MiniConvNeXt: Module, UnaryLayer {
    let dwconv: Audio8MiniCausalConv
    let norm: LayerNorm
    @ModuleInfo var pwconv1: Linear
    @ModuleInfo var pwconv2: Linear
    var gamma: MLXArray
    init(_ dim: Int) {
        dwconv = Audio8MiniCausalConv(dim, dim, kernel: 7, groups: dim)
        norm = LayerNorm(dimensions: dim, eps: 1e-6)
        _pwconv1.wrappedValue = Linear(dim, dim * 4)
        _pwconv2.wrappedValue = Linear(dim * 4, dim)
        gamma = MLXArray.ones([dim]) * 1e-6
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x + gamma * pwconv2(gelu(pwconv1(norm(dwconv(x)))))
    }
}

final class Audio8MiniCodecResample: Module, UnaryLayer {
    let layers: [Module]
    init(up: Bool) {
        let conv: Module = up ? Audio8MiniCausalTranspose(1024, 1024, kernel: 2, stride: 2)
            : Audio8MiniCausalConv(1024, 1024, kernel: 2, stride: 2)
        layers = [conv, Audio8MiniConvNeXt(1024)]
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { layers.reduce(x) { audio8MiniUnary($1, $0) } }
}

final class Audio8MiniDownsampleQuantizer: Module {
    @ModuleInfo(key: "semantic_quantizer") var semantic: Audio8MiniResidualQuantizer
    let quantizer: Audio8MiniResidualQuantizer
    let downsample: [Audio8MiniCodecResample], upsample: [Audio8MiniCodecResample]
    @ModuleInfo(key: "pre_module") var pre: Audio8MiniCodecTransformer
    @ModuleInfo(key: "post_module") var post: Audio8MiniCodecTransformer
    init(_ c: Audio8MiniConfig) {
        _semantic.wrappedValue = Audio8MiniResidualQuantizer(count: 1, size: 4096)
        quantizer = Audio8MiniResidualQuantizer(count: 9, size: 1024)
        downsample = (0..<2).map { _ in Audio8MiniCodecResample(up: false) }
        upsample = (0..<2).map { _ in Audio8MiniCodecResample(up: true) }
        _pre.wrappedValue = Audio8MiniCodecTransformer(layers: 8, dim: 1024, heads: 16, intermediate: 3072, window: 128)
        _post.wrappedValue = Audio8MiniCodecTransformer(layers: c.codecPostNLayer, dim: 1024,
            heads: c.codecPostNHead, kvHeads: c.codecPostNLocalHeads, intermediate: c.codecPostIntermediateSize, window: 128)
    }
    func encode(_ x: MLXArray) -> MLXArray {
        let z = pre(downsample.reduce(x) { $1($0) })
        let (s, sc) = semantic(z)
        let (_, rc) = quantizer(z - s)
        return concatenated([sc, rc], axis: 1)
    }
    func decode(_ ids: MLXArray) -> MLXArray {
        let s = semantic.decode(ids[0..., 0..<1, 0...])
        let r = quantizer.decode(ids[0..., 1..., 0...])
        return upsample.reduce(post(s + r)) { $1($0) }
    }
}

/// The bundled Audio8Mini codec operates at 44.1 kHz with 2,048 samples per frame.
final class Audio8MiniCodec: Module {
    let encoder: Audio8MiniCodecEncoder
    let quantizer: Audio8MiniDownsampleQuantizer
    let decoder: Audio8MiniCodecDecoder
    init(_ config: Audio8MiniConfig) {
        encoder = Audio8MiniCodecEncoder(); quantizer = Audio8MiniDownsampleQuantizer(config); decoder = Audio8MiniCodecDecoder()
    }
    /// The type the codec computes in, matching its loaded weights. Audio comes back as Float32.
    var computeType: DType = .float32
    func encode(_ audio: MLXArray) -> MLXArray {
        let waveform = audio.asType(computeType).reshaped([1, audio.size, 1])
        let right = (2048 - audio.size % 2048) % 2048
        return quantizer.encode(encoder.chunked(padded(waveform, widths: [0, .init((0, right)), 0]))).asType(.int32)
    }
    func decode(_ codes: MLXArray) -> MLXArray {
        decoder.chunked(quantizer.decode(codes))[0, 0..., 0].asType(.float32)
    }
}
