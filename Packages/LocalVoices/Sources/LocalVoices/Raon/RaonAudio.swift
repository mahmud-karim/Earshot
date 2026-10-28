import Foundation
import MLX

/// Audio helpers that follow Raon's reference handling in `f5_tts/infer/utils_infer.py` and
/// `get_sb_hifigan_mel_spectrogram` in `f5_tts/model/modules.py`.
enum RaonAudio {
    static let sampleRate = 16_000
    static let hopLength = 256
    static let fftSize = 1024
    static let melBins = 80

    /// Log mel frames `[frames, 80]` as speechbrain's 16 kHz HiFi-GAN was trained on: the tail is cropped
    /// so frames fit exactly (no centre padding), a periodic Hann window, magnitude (not power), a Slaney
    /// filterbank from 0 to 8 kHz and the natural log clamped at 1e-5.
    static func melSpectrogram(_ samples: [Float]) -> MLXArray {
        let frames = (samples.count - fftSize) / hopLength + 1
        precondition(frames > 0, "the reference is shorter than one window")
        let window = (0 ..< fftSize).map { Float(0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(fftSize))) }
        var framed = [Float](repeating: 0, count: frames * fftSize)
        for frame in 0 ..< frames {
            let start = frame * hopLength
            for i in 0 ..< fftSize {
                framed[frame * fftSize + i] = samples[start + i] * window[i]
            }
        }
        let magnitude = abs(MLXFFT.rfft(MLXArray(framed, [frames, fftSize]), axis: -1))
        let mel = matmul(magnitude, filterbank)
        return log(maximum(mel, MLXArray(Float(1e-5))))
    }

    /// torchaudio `melscale_fbanks(513, 0, 8000, 80, 16000, norm="slaney", mel_scale="slaney")`, `[513, 80]`.
    private static let filterbank: MLXArray = {
        let bins = fftSize / 2 + 1
        let logStep = log(6.4) / 27.0
        func toMel(_ hz: Double) -> Double { hz < 1000 ? hz / (200.0 / 3) : 15 + log(hz / 1000) / logStep }
        func toHz(_ mel: Double) -> Double { mel < 15 ? mel * (200.0 / 3) : 1000 * exp((mel - 15) * logStep) }
        let low = toMel(0), high = toMel(8000)
        let points = (0 ..< melBins + 2).map { toHz(low + (high - low) * Double($0) / Double(melBins + 1)) }
        var weights = [Float](repeating: 0, count: bins * melBins)
        for bin in 0 ..< bins {
            let hz = 8000.0 * Double(bin) / Double(bins - 1)
            for band in 0 ..< melBins {
                let rising = (hz - points[band]) / (points[band + 1] - points[band])
                let falling = (points[band + 2] - hz) / (points[band + 2] - points[band + 1])
                let area = 2.0 / (points[band + 2] - points[band])
                weights[bin * melBins + band] = Float(max(0, min(rising, falling)) * area)
            }
        }
        return MLXArray(weights, [bins, melBins])
    }()

    /// `normalize_peak`: scale so the loudest sample is at full scale.
    static func normalizedPeak(_ samples: [Float]) -> [Float] {
        let peak = max(samples.reduce(Float(0)) { max($0, abs($1)) }, 1e-3)
        return samples.map { min(max($0 / peak, -1), 1) }
    }

    static func rms(_ samples: [Float]) -> Float {
        guard !samples.isEmpty else { return 0 }
        return sqrt(samples.reduce(Float(0)) { $0 + $1 * $1 } / Float(samples.count))
    }

    /// `estimate_ref_seconds_trimmed`: how long the speech in the reference lasts once silence at either
    /// end is dropped, which sets the speaking rate. It mirrors the pydub steps: level the clip to -20
    /// dBFS, call anything 22 dB quieter silence (or quieter than -42 dBFS for a clip that could not be
    /// levelled fully), trim leading silence in 10 ms steps and trailing silence in 1 ms steps, add 50 ms.
    static func speechSeconds(_ samples: [Float]) -> Double {
        let perMs = sampleRate / 1000
        let total = Double(samples.count) / Double(sampleRate)
        func dBFS(_ slice: ArraySlice<Float>) -> Double {
            guard !slice.isEmpty else { return -.infinity }
            let power = slice.reduce(0.0) { $0 + Double($1) * Double($1) } / Double(slice.count)
            return power > 0 ? 10 * log10(power) : -.infinity
        }
        let level = dBFS(samples[...])
        guard level.isFinite else { return total }
        let gain = Float(pow(10, min(max(-20 - level, -30), 30) / 20))
        let leveled = samples.map { min(max($0 * gain, -1), 1) }
        let leveledDB = dBFS(leveled[...])
        let threshold = min(-42.0, Double(Int(leveledDB - 18)))

        let lengthMs = Int((1000 * Double(leveled.count) / Double(sampleRate)).rounded())
        var startMs = 0
        while startMs < lengthMs {
            let from = min(startMs * perMs, leveled.count), to = min((startMs + 10) * perMs, leveled.count)
            if dBFS(leveled[from ..< to]) >= threshold { break }
            startMs += 10
        }
        startMs = min(startMs, lengthMs)
        let speech = Array(leveled[min(startMs * perMs, leveled.count)...])

        let speechMs = Int((1000 * Double(speech.count) / Double(sampleRate)).rounded())
        var seconds = Double(speech.count) / Double(sampleRate)
        for ms in stride(from: speechMs - 1, through: 0, by: -1) {
            let from = min(ms * perMs, speech.count), to = min((ms + 1) * perMs, speech.count)
            if dBFS(speech[from ..< to]) > threshold { break }
            seconds -= 0.001
        }
        let keptMs = Int(seconds * 1000) + 50
        return keptMs < 80 ? total : Double(keptMs) / 1000
    }
}
