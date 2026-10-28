// Adapted from mlx-swift-lm (MIT, John Mai and contributors), Libraries/MLXLLM/Models/FalconH1.swift, and
// the FalconSlowStack of mlx-audio's arktts.py (MIT). Only what Audio8 0.1B uses is kept: no gated Mamba
// norm, no muP folding (the checkpoint's multipliers are all 1, which the config checks), batch of one.

import Foundation
import MLX
import MLXFast
import MLXNN

/// Per-layer generation state: the Mamba conv tail and SSM state, and the attention keys and values.
final class Audio8MiniFalconCache {
    var convState: MLXArray?
    var ssmState: MLXArray?
    let attention = Audio8MiniKVCache()
}

final class Audio8MiniFalconAttention: Module {
    let heads: Int, kvHeads: Int, headDim: Int
    let scale: Float
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    let rope: RoPE

    init(_ c: Audio8MiniConfig) {
        heads = c.nHead; kvHeads = c.nLocalHeads; headDim = c.headDim
        scale = 1 / sqrt(Float(headDim))
        _qProj.wrappedValue = Linear(c.dim, heads * headDim, bias: c.attentionBias)
        _kProj.wrappedValue = Linear(c.dim, kvHeads * headDim, bias: c.attentionBias)
        _vProj.wrappedValue = Linear(c.dim, kvHeads * headDim, bias: c.attentionBias)
        _oProj.wrappedValue = Linear(heads * headDim, c.dim, bias: c.attentionBias)
        // Falcon-H1 uses the usual half-split rotary embedding (not Audio8's interleaved table).
        rope = RoPE(dimensions: headDim, traditional: false, base: c.ropeBase)
    }

    func callAsFunction(_ x: MLXArray, cache: Audio8MiniKVCache) -> MLXArray {
        let b = x.dim(0), t = x.dim(1)
        var q = qProj(x).reshaped(b, t, heads, headDim).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(b, t, kvHeads, headDim).transposed(0, 2, 1, 3)
        var v = vProj(x).reshaped(b, t, kvHeads, headDim).transposed(0, 2, 1, 3)
        q = rope(q, offset: cache.offset)
        k = rope(k, offset: cache.offset)
        (k, v) = cache.update(keys: k, values: v)
        // The cache only ever holds earlier positions, so a single new step needs no mask.
        let mask: MLXFast.ScaledDotProductAttentionMaskMode = t > 1 ? .causal : .none
        let y = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: mask)
        return oProj(y.transposed(0, 2, 1, 3).reshaped(b, t, heads * headDim))
    }
}

/// The Mamba-2 half of a Falcon-H1 layer.
final class Audio8MiniFalconMixer: Module {
    let heads: Int, headDim: Int, stateSize: Int, groups: Int
    let innerSize: Int, convDim: Int, convKernel: Int, chunkSize: Int
    let timeStepLimit: (Float, Float) = (0, .infinity)

    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ParameterInfo(key: "D") var d: MLXArray

    init(_ c: Audio8MiniConfig) {
        heads = c.mambaNHeads; headDim = c.mambaDHead; stateSize = c.mambaDState; groups = c.mambaNGroups
        innerSize = c.mambaDSSM; convKernel = c.mambaDConv; chunkSize = c.mambaChunkSize
        convDim = innerSize + 2 * groups * stateSize
        _inProj.wrappedValue = Linear(c.dim, innerSize + convDim + heads, bias: c.mambaProjBias)
        _conv1d.wrappedValue = Conv1d(inputChannels: convDim, outputChannels: convDim, kernelSize: convKernel,
                                      groups: convDim, bias: c.mambaConvBias)
        _outProj.wrappedValue = Linear(innerSize, c.dim, bias: c.projectorsBias)
        _dtBias.wrappedValue = MLXArray.ones([heads])
        _aLog.wrappedValue = log(MLXArray(Array(1...heads).map(Float.init)))
        _d.wrappedValue = MLXArray.ones([heads])
    }

    func callAsFunction(_ x: MLXArray, cache: Audio8MiniFalconCache) -> MLXArray {
        let b = x.dim(0), t = x.dim(1)
        let parts = MLX.split(inProj(x), indices: [innerSize, innerSize + convDim], axis: -1)
        let gate = parts[0], convInput = parts[1], dt = parts[2]

        // Causal depthwise conv: prepend the last kernel-1 inputs of the previous call (zeros at first).
        let tail = cache.convState ?? MLXArray.zeros([b, convKernel - 1, convDim], dtype: convInput.dtype)
        let padded = concatenated([tail, convInput], axis: 1)
        cache.convState = padded[0..., (padded.dim(1) - (convKernel - 1))..., 0...]
        let convOutput = silu(conv1d(padded))

        let streams = MLX.split(convOutput, indices: [innerSize, innerSize + groups * stateSize], axis: -1)
        let (y, state) = Audio8MiniSSM.update(
            x: streams[0].reshaped(b, t, heads, headDim), aLog: aLog,
            B: streams[1].reshaped(b, t, groups, stateSize), C: streams[2].reshaped(b, t, groups, stateSize),
            D: d, dt: dt, dtBias: dtBias, state: cache.ssmState, timeStepLimit: timeStepLimit, step: chunkSize)
        cache.ssmState = state
        return outProj(y.reshaped(b, t, innerSize) * silu(gate))
    }
}

final class Audio8MiniFalconMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ c: Audio8MiniConfig) {
        _gateProj.wrappedValue = Linear(c.dim, c.intermediateSize, bias: c.mlpBias)
        _upProj.wrappedValue = Linear(c.dim, c.intermediateSize, bias: c.mlpBias)
        _downProj.wrappedValue = Linear(c.intermediateSize, c.dim, bias: c.mlpBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { downProj(silu(gateProj(x)) * upProj(x)) }
}

/// Mamba-2 and attention read the same normed input in parallel; their outputs add to the residual.
final class Audio8MiniFalconLayer: Module {
    @ModuleInfo(key: "feed_forward") var feedForward: Audio8MiniFalconMLP
    @ModuleInfo(key: "mamba") var mamba: Audio8MiniFalconMixer
    @ModuleInfo(key: "self_attn") var attention: Audio8MiniFalconAttention
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "pre_ff_layernorm") var preFeedForwardNorm: RMSNorm

    init(_ c: Audio8MiniConfig) {
        _feedForward.wrappedValue = Audio8MiniFalconMLP(c)
        _mamba.wrappedValue = Audio8MiniFalconMixer(c)
        _attention.wrappedValue = Audio8MiniFalconAttention(c)
        _inputNorm.wrappedValue = RMSNorm(dimensions: c.dim, eps: c.normEps)
        _preFeedForwardNorm.wrappedValue = RMSNorm(dimensions: c.dim, eps: c.normEps)
    }

    func callAsFunction(_ x: MLXArray, cache: Audio8MiniFalconCache) -> MLXArray {
        let normed = inputNorm(x)
        let h = x + mamba(normed, cache: cache) + attention(normed, cache: cache.attention)
        return h + feedForward(preFeedForwardNorm(h))
    }
}

/// The 0.1B slow backbone, entered at the embedding rather than at token ids: Audio8's slow input is
/// the text embedding plus ten codebook embeddings, which a plain Falcon-H1 model cannot take.
final class Audio8MiniFalconStack: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [Audio8MiniFalconLayer]
    @ModuleInfo(key: "final_layernorm") var finalNorm: RMSNorm
    let embeddingMultiplier: Float

    init(_ c: Audio8MiniConfig) {
        _embedTokens.wrappedValue = Embedding(embeddingCount: c.vocabSize, dimensions: c.dim)
        layers = (0..<c.nLayer).map { _ in Audio8MiniFalconLayer(c) }
        _finalNorm.wrappedValue = RMSNorm(dimensions: c.dim, eps: c.normEps)
        embeddingMultiplier = c.embeddingMultiplier
    }

    func makeCache() -> [Audio8MiniFalconCache] { layers.map { _ in Audio8MiniFalconCache() } }

    /// `embedding` is the unscaled composite. The multiplier must scale all of it, which is why it is
    /// applied here and not folded into `embed_tokens` the way Falcon-H1's sanitize does.
    func callAsFunction(_ embedding: MLXArray, cache: [Audio8MiniFalconCache]) -> MLXArray {
        var h = embedding * embeddingMultiplier
        for (layer, layerCache) in zip(layers, cache) { h = layer(h, cache: layerCache) }
        return finalNorm(h)
    }
}
