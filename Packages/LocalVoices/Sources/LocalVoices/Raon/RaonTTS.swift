import Foundation
@preconcurrency import MLX
import MLXAudioCore
import MLXAudioTTS

/// Settings for Raon's flow-matching sampler. The defaults are the official inference settings except
/// for `steps`, which trades speed for detail.
public struct RaonOptions: Sendable {
    /// ODE steps, one network pass each (NFE). 16 uses the official pruned schedule (EPSS).
    public var steps: Int
    public var cfgStrength: Float = 2
    public var swaySamplingCoefficient: Float = -1
    /// Speaking rate; above 1 is faster.
    public var speed: Float = 1

    public init(steps: Int = RaonTTS.defaultSteps) {
        self.steps = steps
    }
}

/// Raon-OpenTTS (KRAFTON): zero-shot speech in the voice of a reference recording. Follows
/// `infer_process` in `f5_tts/infer/utils_infer.py` with the settings of `infer_cli.py`.
public final class RaonTTS: SpeechGenerationModel, @unchecked Sendable {
    public static let defaultSteps = 16

    public let sampleRate = RaonAudio.sampleRate
    public var defaultGenerationParameters: GenerateParameters { GenerateParameters(temperature: 0) }
    public var options = RaonOptions()

    private let dit: RaonDiT
    private let vocoder: RaonHiFiGAN
    private let tokenizer: RaonTokenizer

    /// Loudness the reference is raised to when it is quieter; the output is lowered to match.
    private static let targetRMS: Float = 0.1
    private static let crossFadeSeconds = 0.15
    private static let maxFrames = 4096

    init(dit: RaonDiT, vocoder: RaonHiFiGAN, tokenizer: RaonTokenizer) {
        self.dit = dit
        self.vocoder = vocoder
        self.tokenizer = tokenizer
    }

    /// One network pass and a short vocoder run on silence, so the GPU kernels are compiled while
    /// loading rather than during the first sentence (Raon has no built-in voice for the engine's warm-up).
    func warmUp() {
        let frames = 256
        let silence = MLXArray.zeros([1, frames, RaonAudio.melBins])
        let tokens = tokenizer.encode("hi.")
        let context = dit.context(referenceMel: silence, text: dit.embedText(tokens, frames: frames, drop: false),
                                  nullText: dit.embedText(tokens, frames: frames, drop: true))
        eval(dit.velocity(silence, time: 0.5, context: context), vocoder(silence[0..., 0 ..< 32, 0...]))
    }

    public func generate(text: String, voice: String?, refAudio: sending MLXArray?, refText: String?, language: String?,
                         generationParameters: GenerateParameters) async throws -> sending MLXArray {
        let audio = try synthesize(text: text, reference: refAudio, transcript: refText, seed: generationParameters.seed)
        nonisolated(unsafe) let result = audio
        return result
    }

    /// Raon renders whole sentences, so the stream carries one waveform.
    public func generateStream(text: String, voice: String?, refAudio: sending MLXArray?, refText: String?, language: String?,
                               generationParameters: GenerateParameters) -> sending AsyncThrowingStream<AudioGeneration, Error> {
        nonisolated(unsafe) let reference = refAudio
        return AsyncThrowingStream { continuation in
            let task = Task { @Sendable [self] in
                do {
                    let audio = try synthesize(text: text, reference: reference, transcript: refText,
                                               seed: generationParameters.seed)
                    nonisolated(unsafe) let output = audio
                    continuation.yield(.audio(output))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    private func synthesize(text: String, reference: MLXArray?, transcript: String?, seed: UInt64?) throws -> MLXArray {
        guard let reference, reference.size >= RaonAudio.fftSize else {
            throw AudioGenerationError.invalidInput("Raon needs a reference recording of at least a tenth of a second")
        }
        let referenceText = RaonText.referenceEnding((transcript ?? "").lowercased())
        guard !referenceText.isEmpty else {
            throw AudioGenerationError.invalidInput("Raon needs the words spoken in the reference recording")
        }
        let words = text.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !words.isEmpty else { return MLXArray([Float]()) }

        let raw = reference.asType(.float32).reshaped(-1).asArray(Float.self)
        let speechSeconds = RaonAudio.speechSeconds(raw)
        var samples = RaonAudio.normalizedPeak(raw)
        let rms = RaonAudio.rms(samples)
        if rms > 0, rms < Self.targetRMS {
            samples = samples.map { $0 * Self.targetRMS / rms }
        }
        let referenceMel = RaonAudio.melSpectrogram(samples).expandedDimensions(axis: 0)
        // Generated audio starts at the reference's length in hops, a few frames past its last mel frame.
        let referenceFrames = samples.count / RaonAudio.hopLength

        let maxBytes = Int(Double(referenceText.utf8.count) / max(speechSeconds, 1e-6) * (22 - speechSeconds))
        let prompt = referenceText.unicodeScalars.last?.isASCII == true ? referenceText + " " : referenceText

        if let seed { MLXRandom.seed(seed) }
        var waves: [[Float]] = []
        for chunk in RaonText.chunks(words, maxBytes: maxBytes) {
            try Task.checkCancellation()
            let mel = try sampleMel(prompt: prompt, text: chunk, referenceMel: referenceMel,
                                    referenceFrames: referenceFrames, speechSeconds: speechSeconds)
            var wave = vocoder(mel)
            if rms > 0, rms < Self.targetRMS { wave = wave * (rms / Self.targetRMS) }
            waves.append(wave.asArray(Float.self))
            Memory.clearCache()
        }
        return MLXArray(Self.crossFaded(waves))
    }

    /// One chunk: Euler steps of the classifier-free-guided flow from noise to mel, as `CFM.sample`.
    /// Returns the generated frames only, `[1, frames, 80]`.
    private func sampleMel(prompt: String, text: String, referenceMel: MLXArray, referenceFrames: Int,
                           speechSeconds: Double) throws -> MLXArray {
        let tokens = tokenizer.encode(prompt + text)

        // Duration from the reference's speaking rate per UTF-8 byte, at least 12 characters a second.
        let hopsPerSecond = Double(RaonAudio.sampleRate) / Double(RaonAudio.hopLength)
        let speechHops = Int(speechSeconds * hopsPerSecond)
        let secondsPerByte = min(Double(speechHops) / hopsPerSecond / Double(max(prompt.utf8.count, 1)), 1.0 / 12)
        let speed = text.utf8.count < 10 ? 0.3 : Double(options.speed)
        let newFrames = max(Int(secondsPerByte * Double(text.utf8.count) / max(speed, 1e-6) * hopsPerSecond), 1)
        let conditionFrames = referenceMel.dim(1)
        var frames = max(referenceFrames + newFrames, max(tokens.count, conditionFrames) + 1)
        frames = min(frames, Self.maxFrames)
        guard frames > referenceFrames else {
            throw AudioGenerationError.invalidInput("The reference recording is too long for Raon")
        }

        let cond = padded(referenceMel, widths: [IntOrPair((0, 0)), IntOrPair((0, frames - conditionFrames)), IntOrPair((0, 0))])
        let context = dit.context(referenceMel: cond,
                                  text: dit.embedText(tokens, frames: frames, drop: false),
                                  nullText: dit.embedText(tokens, frames: frames, drop: true))
        eval(context)

        var x = MLXRandom.normal([1, frames, RaonAudio.melBins])
        let times = Self.timesteps(options.steps, sway: options.swaySamplingCoefficient)
        let guidance = options.cfgStrength
        for (t, next) in zip(times, times.dropFirst()) {
            try Task.checkCancellation()
            let both = dit.velocity(x, time: t, context: context).asType(.float32)
            let conditional = both[0 ..< 1], unconditional = both[1 ..< 2]
            x = x + (next - t) * (conditional + (conditional - unconditional) * guidance)
            eval(x)
        }
        return x[0..., referenceFrames ..< frames, 0...]
    }

    /// `get_epss_timesteps` (the pruned schedule for the step counts it lists, otherwise uniform), then
    /// sway sampling, which spends more of the steps near the noise end.
    static func timesteps(_ steps: Int, sway: Float) -> [Float] {
        let pruned: [Int: [Float]] = [
            5: [0, 2, 4, 8, 16, 32], 6: [0, 2, 4, 6, 8, 16, 32], 7: [0, 2, 4, 6, 8, 16, 24, 32],
            10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32], 12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
            16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32]
        ]
        let base = pruned[steps].map { $0.map { $0 / 32 } } ?? (0 ... steps).map { Float($0) / Float(steps) }
        return base.map { $0 + sway * (cos(Float.pi / 2 * $0) - 1 + $0) }
    }

    /// Joins chunks with a 0.15 s linear cross-fade, as `infer_batch_process`.
    private static func crossFaded(_ waves: [[Float]]) -> [Float] {
        guard var result = waves.first else { return [] }
        for next in waves.dropFirst() {
            let overlap = min(Int(crossFadeSeconds * Double(RaonAudio.sampleRate)), result.count, next.count)
            guard overlap > 1 else { result += next; continue }
            let start = result.count - overlap
            for i in 0 ..< overlap {
                let fade = Float(i) / Float(overlap - 1)
                result[start + i] = result[start + i] * (1 - fade) + next[i] * fade
            }
            result += next[overlap...]
        }
        return result
    }
}
