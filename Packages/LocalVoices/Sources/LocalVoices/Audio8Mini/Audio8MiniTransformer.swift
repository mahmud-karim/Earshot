// Adapted from mlx-audio-swift (MIT, Prince Canuma and contributors), Sources/MLXAudioTTS/Models/Audio8/
// Audio8Transformer.swift at fe2c182. Those types are internal to MLXAudioTTS, so they live here
// under Audio8Mini names. The fast stack and the codec use these blocks unchanged from the 0.6B.

import Foundation
import MLX
import MLXFast
import MLXNN

/// RMS norm computed in Float32, as the reference does.
final class Audio8MiniNorm: Module, UnaryLayer {
    var weight: MLXArray
    let eps: Float
    init(_ dim: Int, eps: Float) { self.weight = .ones([dim]); self.eps = eps }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let f = x.asType(.float32)
        return (f * rsqrt(mean(f * f, axis: -1, keepDims: true) + eps)).asType(x.dtype) * weight
    }
}

/// Interleaved-pair rotary embedding. Audio8 rounds the table to BF16 before applying it in Float32.
final class Audio8MiniRoPE {
    let cosine: MLXArray
    let sine: MLXArray
    init(length: Int, dim: Int, base: Float) {
        let frequency = 1 / pow(MLXArray(base), MLX.arange(0, dim, step: 2, dtype: .float32) / Float(dim))
        let phase = MLX.outer(MLX.arange(length, dtype: .float32), frequency)
        cosine = cos(phase).asType(.bfloat16)
        sine = sin(phase).asType(.bfloat16)
        eval(cosine, sine)
    }
    func apply(_ x: MLXArray, offset: Int) -> MLXArray {
        let length = x.shape[1], half = x.shape[3] / 2
        let c = cosine[offset..<(offset + length)].asType(.float32).reshaped([1, length, 1, half])
        let s = sine[offset..<(offset + length)].asType(.float32).reshaped([1, length, 1, half])
        let y = x.asType(.float32).reshaped(x.shape.dropLast() + [half, 2])
        let a = y[0..., 0..., 0..., 0..., 0], b = y[0..., 0..., 0..., 0..., 1]
        return stacked([a * c - b * s, b * c + a * s], axis: -1).reshaped(x.shape).asType(x.dtype)
    }
}

final class Audio8MiniAttention: Module {
    @ModuleInfo var wqkv: Linear
    @ModuleInfo var wo: Linear
    let heads: Int, kvHeads: Int, headDim: Int
    init(dim: Int, heads: Int, kvHeads: Int, headDim: Int, qkvBias: Bool = false, outputBias: Bool = false) {
        self.heads = heads; self.kvHeads = kvHeads; self.headDim = headDim
        _wqkv.wrappedValue = Linear(dim, (heads + 2 * kvHeads) * headDim, bias: qkvBias)
        _wo.wrappedValue = Linear(heads * headDim, dim, bias: outputBias)
    }
    func callAsFunction(_ x: MLXArray, rope: Audio8MiniRoPE, offset: Int,
                        mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: Audio8MiniKVCache? = nil) -> MLXArray {
        let b = x.shape[0], t = x.shape[1], qSize = heads * headDim, kvSize = kvHeads * headDim
        let qkv = wqkv(x)
        var q = qkv[0..., 0..., 0..<qSize].reshaped([b, t, heads, headDim])
        var k = qkv[0..., 0..., qSize..<(qSize + kvSize)].reshaped([b, t, kvHeads, headDim])
        var v = qkv[0..., 0..., (qSize + kvSize)...].reshaped([b, t, kvHeads, headDim]).transposed(0, 2, 1, 3)
        q = rope.apply(q, offset: offset).transposed(0, 2, 1, 3)
        k = rope.apply(k, offset: offset).transposed(0, 2, 1, 3)
        if let cache { (k, v) = cache.update(keys: k, values: v) }
        let y = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v,
                    scale: 1 / sqrt(Float(headDim)), mask: mask)
        return wo(y.transposed(0, 2, 1, 3).reshaped([b, t, qSize]))
    }
}

final class Audio8MiniFeedForward: Module, UnaryLayer {
    @ModuleInfo var w1: Linear
    @ModuleInfo var w2: Linear
    @ModuleInfo var w3: Linear
    init(dim: Int, intermediate: Int) {
        _w1.wrappedValue = Linear(dim, intermediate, bias: false)
        _w2.wrappedValue = Linear(intermediate, dim, bias: false)
        _w3.wrappedValue = Linear(dim, intermediate, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { w2(silu(w1(x)) * w3(x)) }
}

final class Audio8MiniLayerScale: Module, UnaryLayer {
    var gamma: MLXArray
    init(_ dim: Int) { gamma = MLXArray.ones([dim]) * 0.01 }
    func callAsFunction(_ x: MLXArray) -> MLXArray { x * gamma }
}

final class Audio8MiniBlock: Module {
    let attention: Audio8MiniAttention
    @ModuleInfo(key: "feed_forward") var feedForward: Audio8MiniFeedForward
    @ModuleInfo(key: "attention_norm") var attentionNorm: Audio8MiniNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: Audio8MiniNorm
    @ModuleInfo(key: "attention_layer_scale") var attentionScale: Audio8MiniLayerScale?
    @ModuleInfo(key: "ffn_layer_scale") var ffnScale: Audio8MiniLayerScale?
    init(dim: Int, intermediate: Int, heads: Int, kvHeads: Int, headDim: Int,
         qkvBias: Bool = false, outputBias: Bool = false, eps: Float = 1e-5, layerScale: Bool = false) {
        attention = Audio8MiniAttention(dim: dim, heads: heads, kvHeads: kvHeads, headDim: headDim,
                                        qkvBias: qkvBias, outputBias: outputBias)
        _feedForward.wrappedValue = Audio8MiniFeedForward(dim: dim, intermediate: intermediate)
        _attentionNorm.wrappedValue = Audio8MiniNorm(dim, eps: eps)
        _ffnNorm.wrappedValue = Audio8MiniNorm(dim, eps: eps)
        _attentionScale.wrappedValue = layerScale ? Audio8MiniLayerScale(dim) : nil
        _ffnScale.wrappedValue = layerScale ? Audio8MiniLayerScale(dim) : nil
    }
    func callAsFunction(_ x: MLXArray, rope: Audio8MiniRoPE, offset: Int = 0,
                        mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: Audio8MiniKVCache? = nil) -> MLXArray {
        var a = attention(attentionNorm(x), rope: rope, offset: offset, mask: mask, cache: cache)
        if let attentionScale { a = attentionScale(a) }
        let h = x + a
        var f = feedForward(ffnNorm(h))
        if let ffnScale { f = ffnScale(f) }
        return h + f
    }
}

/// A growing key/value cache for one attention layer. The slow stack's Falcon attention and the fast
/// stack both append to it; `offset` is the number of positions already stored.
final class Audio8MiniKVCache {
    private var keys: MLXArray?
    private var values: MLXArray?
    private(set) var offset = 0
    private let step = 256

    func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let previous = offset
        let needed = previous + newKeys.dim(2)
        if keys == nil || needed > keys!.dim(2) {
            let grow = ((newKeys.dim(2) + step - 1) / step) * step
            var kShape = newKeys.shape, vShape = newValues.shape
            kShape[2] = grow; vShape[2] = grow
            let k = MLXArray.zeros(kShape, dtype: newKeys.dtype), v = MLXArray.zeros(vShape, dtype: newValues.dtype)
            if let oldKeys = keys, let oldValues = values {
                keys = concatenated([oldKeys[.ellipsis, ..<previous, 0...], k], axis: 2)
                values = concatenated([oldValues[.ellipsis, ..<previous, 0...], v], axis: 2)
            } else {
                keys = k; values = v
            }
        }
        offset = needed
        keys![.ellipsis, previous..<needed, 0...] = newKeys
        values![.ellipsis, previous..<needed, 0...] = newValues
        return (keys![.ellipsis, ..<needed, 0...], values![.ellipsis, ..<needed, 0...])
    }
}
