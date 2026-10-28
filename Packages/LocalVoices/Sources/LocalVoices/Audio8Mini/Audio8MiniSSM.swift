// Adapted from mlx-swift-lm (MIT, John Mai and contributors), Libraries/MLXLLM/Models/SSM.swift, itself a
// port of mlx-lm's ssm.py. mlx-swift-lm is not a dependency here, so the Mamba-2 scan is copied under
// an Audio8Mini namespace. Batch masking and per-row lengths are dropped: Audio8 runs one row at a time.

import Foundation
import MLX
import MLXFast
import MLXNN

enum Audio8MiniSSM {
    static func computeDt(_ dt: MLXArray, _ dtBias: MLXArray, _ timeStepLimit: (Float, Float)) -> MLXArray {
        let dt = softplus(dt.asType(.float32) + dtBias)
        return MLX.clip(dt, min: timeStepLimit.0, max: timeStepLimit.1)
    }

    /// One recurrent step on the GPU: state' = exp(A dt) state + dt B x, y = C state' + D x.
    private static let stepKernel = MLXFast.metalKernel(
        name: "audio8mini_ssm_step",
        inputNames: ["X", "A_log", "B", "C", "D", "dt", "state_in"],
        outputNames: ["out", "state_out"],
        source: """
            auto n = thread_position_in_grid.z;
            auto h_idx = n % H;
            auto g_idx = n / G;
            constexpr int n_per_t = Ds / 32;

            auto x = X + n * Dh;
            out += n * Dh;
            auto i_state = state_in + n * Dh * Ds;
            auto o_state = state_out + n * Dh * Ds;

            auto C_ = C + g_idx * Ds;
            auto B_ = B + g_idx * Ds;

            auto ds_idx = thread_position_in_threadgroup.x;
            auto d_idx = thread_position_in_grid.y;

            auto dt_ = static_cast<float>(dt[n]);
            auto A = -fast::exp(static_cast<float>(A_log[h_idx]));
            auto dA = fast::exp(A * dt_);

            float acc = 0.0;
            auto x_ = static_cast<float>(x[d_idx]);

            for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * ds_idx + i;
                auto idx = d_idx * Ds + s_idx;
                auto dB_by_x = x_ * dt_ * static_cast<float>(B_[s_idx]);
                auto state = dA * static_cast<float>(i_state[idx]) + dB_by_x;
                o_state[idx] = static_cast<U>(state);
                acc += state * static_cast<float>(C_[s_idx]);
            }
            acc = simd_sum(acc);
            if (thread_index_in_simdgroup == 0) {
                out[d_idx] = static_cast<T>(acc + x_ * static_cast<float>(D[h_idx]));
            }
        """)

    private static func stepOnGPU(
        x: MLXArray, aLog: MLXArray, B: MLXArray, C: MLXArray, D: MLXArray,
        dt: MLXArray, dtBias: MLXArray, state: MLXArray, timeStepLimit: (Float, Float)
    ) -> (MLXArray, MLXArray) {
        let n = x.dim(0), h = x.dim(2), d = x.dim(3)
        let groups = B.dim(-2), stateSize = B.dim(-1)
        let dt = computeDt(dt, dtBias, timeStepLimit)
        let outputs = stepKernel(
            [x, aLog, B, C, D, dt, state],
            template: [("T", x.dtype), ("U", state.dtype), ("Dh", d), ("Ds", stateSize), ("H", h), ("G", h / groups)],
            grid: (32, d, h * n),
            threadGroup: (32, 8, 1),
            outputShapes: [[n, 1, h, d], state.shape],
            outputDTypes: [x.dtype, state.dtype])
        return (outputs[0], outputs[1])
    }

    private static func segsum(_ x: MLXArray) -> MLXArray {
        let l = x.dim(-1)
        let repeated = MLX.tril(MLX.repeated(x[.ellipsis, .newAxis], count: l, axis: -1), k: -1)
        return MLX.cumsum(repeated, axis: -2)
    }

    /// The chunked SSD form of the scan, for prompts (and for every step on the CPU).
    private static func scan(
        x: MLXArray, aLog: MLXArray, B: MLXArray, C: MLXArray, D: MLXArray, dt: MLXArray, dtBias: MLXArray,
        state: MLXArray?, timeStepLimit: (Float, Float), step: Int
    ) -> (MLXArray, MLXArray) {
        let b = x.dim(0), l = x.dim(1), h = x.dim(2), dh = x.dim(3)
        let g = B.dim(2), d = B.dim(3)
        let dt = computeDt(dt, dtBias, timeStepLimit)
        let repeats = h / g
        let A = -MLX.exp(aLog).asType(dt.dtype)
        let dtA = dt * A.reshaped(1, 1, -1)
        let dtx = dt.reshaped(b, l, h, 1) * x

        func chunk(dtx: MLXArray, dtA: MLXArray, B: MLXArray, C: MLXArray, state: MLXArray?) -> (MLXArray, MLXArray) {
            let s = dtx.dim(1)
            var B = MLX.transposed(B, axes: [0, 2, 3, 1])
            let CB = MLX.repeated(MLX.swappedAxes(C, 1, 2).matmul(B), count: repeats, axis: 1)
            var decay = MLX.exp(segsum(dtA.swappedAxes(1, 2)))
            var y = MLX.tril(CB * decay, k: 0).matmul(dtx.swappedAxes(1, 2))
            y = MLX.swappedAxes(y, 1, 2)

            decay = decay[0..., 0..., (-1)..., 0...].transposed(0, 3, 1, 2)
            B = MLX.repeated(B, count: repeats, axis: 1).swappedAxes(2, 3)
            let dtxDecay = (dtx * decay).swappedAxes(1, 2).swappedAxes(2, 3)
            var next = dtxDecay.matmul(B)

            if let state {
                let decayed = MLX.exp(MLX.cumsum(dtA, axis: -2))
                next = next + decayed[0..., -1, 0..., .newAxis, .newAxis] * state
                let previous = state.reshaped(b, 1, g, repeats, dh, d)
                    .matmul(C.reshaped(b, s, g, 1, d, 1)).reshaped(b, s, h, dh)
                y = y + decayed[.ellipsis, .newAxis] * previous
            }
            // The recurrent state stays Float32 across the whole utterance, as mlx-lm's GPU path keeps it.
            return (y.asType(x.dtype), next.asType(.float32))
        }

        var ys = [MLXArray]()
        var state = state
        for start in stride(from: 0, to: l, by: step) {
            let end = min(start + step, l)
            let (y, next) = chunk(dtx: dtx[0..., start..<end], dtA: dtA[0..., start..<end],
                                  B: B[0..., start..<end], C: C[0..., start..<end], state: state)
            ys.append(y)
            state = next
        }
        return (MLX.concatenated(ys, axis: 1) + x * D.reshaped(1, 1, h, 1), state!)
    }

    /// Runs the selective scan over `x` (batch, time, heads, headDim), continuing from `state`
    /// (batch, heads, headDim, stateSize) when given. Returns the output and the new Float32 state.
    static func update(
        x: MLXArray, aLog: MLXArray, B: MLXArray, C: MLXArray, D: MLXArray, dt: MLXArray, dtBias: MLXArray,
        state: MLXArray?, timeStepLimit: (Float, Float), step: Int
    ) -> (MLXArray, MLXArray) {
        if x.dim(1) == 1, let state, B.dim(-1) % 32 == 0, Device.defaultDevice() == .gpu {
            return stepOnGPU(x: x, aLog: aLog, B: B, C: C, D: D, dt: dt, dtBias: dtBias,
                             state: state, timeStepLimit: timeStepLimit)
        }
        return scan(x: x, aLog: aLog, B: B, C: C, D: D, dt: dt, dtBias: dtBias,
                    state: state, timeStepLimit: timeStepLimit, step: step)
    }
}
