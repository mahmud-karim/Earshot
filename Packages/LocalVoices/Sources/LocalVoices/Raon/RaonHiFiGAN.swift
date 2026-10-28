import Foundation
import MLX
import MLXNN

/// speechbrain's HiFi-GAN generator (V1: upsample 8, 8, 2, 2; residual kernels 3, 7, 11 with dilations
/// 1, 3, 5), as in Raon's `f5_tts/model/vocoder.py`. Turns 80-bin log mel frames into 16 kHz audio.
/// Weight norm is folded into plain weights at load time; everything runs channels-last.
struct RaonHiFiGAN {
    private struct Conv {
        let weight: MLXArray // [out, kernel, in]
        let bias: MLXArray?
        let padding: Int
        let dilation: Int

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var y = conv1d(x, weight, padding: padding, dilation: dilation)
            if let bias { y = y + bias }
            return y
        }
    }

    private struct Upsample {
        let weight: MLXArray // [out, kernel, in]
        let bias: MLXArray
        let stride: Int
        let padding: Int

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            convTransposed1d(x, weight, stride: stride, padding: padding) + bias
        }
    }

    private let convPre: Conv
    private let upsamples: [Upsample]
    /// Per upsample stage, three residual blocks; each block is a list of (dilated conv, plain conv) pairs.
    private let resblocks: [[[(Conv, Conv)]]]
    private let convPost: Conv

    private static let upsampleFactors = [8, 8, 2, 2]
    private static let upsampleKernels = [16, 16, 4, 4]
    private static let resblockKernels = [3, 7, 11]
    private static let resblockDilations = [1, 3, 5]

    init(weights: [String: MLXArray], dtype: DType) throws {
        func folded(_ prefix: String, transposed: Bool = false) throws -> MLXArray {
            guard let g = weights["\(prefix).weight_g"], let v = weights["\(prefix).weight_v"] else {
                throw LocalVoiceError("The vocoder has no \(prefix).")
            }
            // torch weight_norm(dim=0): w = g * v / |v|, the norm taken over all but the first axis.
            let norm = sqrt((v * v).sum(axes: [1, 2], keepDims: true))
            let w = g * v / norm
            // Conv1d is [out, in, k] and ConvTranspose1d is [in, out, k]; MLX wants [out, k, in] for both.
            return (transposed ? w.transposed(1, 2, 0) : w.transposed(0, 2, 1)).asType(dtype)
        }
        func bias(_ prefix: String) throws -> MLXArray {
            guard let b = weights["\(prefix).bias"] else { throw LocalVoiceError("The vocoder has no \(prefix).bias.") }
            return b.asType(dtype)
        }
        func conv(_ prefix: String, kernel: Int, dilation: Int = 1) throws -> Conv {
            Conv(weight: try folded("\(prefix).conv"), bias: try bias("\(prefix).conv"),
                 padding: (kernel * dilation - dilation) / 2, dilation: dilation)
        }

        convPre = try conv("conv_pre", kernel: 7)
        var upsamples: [Upsample] = []
        var resblocks: [[[(Conv, Conv)]]] = []
        for stage in 0 ..< Self.upsampleFactors.count {
            let factor = Self.upsampleFactors[stage], kernel = Self.upsampleKernels[stage]
            upsamples.append(Upsample(weight: try folded("ups.\(stage).conv", transposed: true),
                                      bias: try bias("ups.\(stage).conv"),
                                      stride: factor, padding: (kernel - factor) / 2))
            var blocks: [[(Conv, Conv)]] = []
            for (j, k) in Self.resblockKernels.enumerated() {
                let index = stage * Self.resblockKernels.count + j
                blocks.append(try Self.resblockDilations.enumerated().map { i, d in
                    (try conv("resblocks.\(index).convs1.\(i)", kernel: k, dilation: d),
                     try conv("resblocks.\(index).convs2.\(i)", kernel: k))
                })
            }
            resblocks.append(blocks)
        }
        self.upsamples = upsamples
        self.resblocks = resblocks
        convPost = try conv("conv_post", kernel: 7)
    }

    /// Every weight, to fold the weight norm once at load time.
    var arrays: [MLXArray] {
        var convs = [convPre, convPost]
        for stage in resblocks { for block in stage { for (a, b) in block { convs += [a, b] } } }
        return convs.flatMap { [$0.weight] + ($0.bias.map { [$0] } ?? []) } + upsamples.flatMap { [$0.weight, $0.bias] }
    }

    /// `mel`: `[1, frames, 80]`. Returns `[frames * 256]` samples in -1...1.
    func callAsFunction(_ mel: MLXArray) -> MLXArray {
        var x = convPre(mel.asType(convPre.weight.dtype))
        for (upsample, blocks) in zip(upsamples, resblocks) {
            x = upsample(leakyRelu(x, negativeSlope: 0.1))
            var sum: MLXArray?
            for block in blocks {
                var y = x
                for (dilated, plain) in block {
                    y = y + plain(leakyRelu(dilated(leakyRelu(y, negativeSlope: 0.1)), negativeSlope: 0.1))
                }
                sum = sum.map { $0 + y } ?? y
            }
            x = sum! / Float(blocks.count)
        }
        x = tanh(convPost(leakyRelu(x, negativeSlope: 0.01)))
        return x.reshaped(-1).asType(.float32)
    }
}
