import Foundation
@preconcurrency import MLX
import MLXAudioCore
import MLXAudioTTS

/// Speaks with one local model at a time. Models are big, so switching unloads the previous one,
/// and every request waits for the one before it: a model must never run two generations at once.
public final class LocalVoiceEngine: @unchecked Sendable {
    public static let shared = LocalVoiceEngine()

    private let store: LocalModelStore
    private let lock = NSLock()
    private var tail: Task<Void, Never>?
    private var loaded: (model: LocalVoiceModel, speech: SpeechGenerationModel)?

    public init(store: LocalModelStore = .shared) {
        self.store = store
        // Keep freed GPU memory small: the phone has other apps to run.
        Memory.cacheLimit = 256 * 1024 * 1024
    }

    public var loadedModel: LocalVoiceModel? {
        lock.withLock { loaded?.model }
    }

    /// Runs `work` after every earlier request has finished.
    private func serial<T: Sendable>(_ work: @escaping @Sendable () async throws -> T) async throws -> T {
        let current: Task<T, Error> = lock.withLock {
            let previous = tail
            let current = Task {
                await previous?.value
                return try await work()
            }
            tail = Task { _ = try? await current.value }
            return current
        }
        return try await withTaskCancellationHandler {
            try await current.value
        } onCancel: {
            current.cancel()
        }
    }

    /// Loads the model so the first sentence does not wait for it.
    public func prepare(_ model: LocalVoiceModel) async throws {
        _ = try await serial { try await self.speechModel(model) }
    }

    public func unload() async {
        _ = try? await serial {
            self.lock.withLock { self.loaded = nil }
            Memory.clearCache()
        }
    }

    /// One sentence of speech: mono samples at `sampleRate`.
    /// `pace` applies to Ming with a reference only; other voices keep their own pace.
    public func speak(_ text: String, model: LocalVoiceModel, voice: String?, reference: VoiceReference?,
                      language: String? = nil, pace: LocalVoicePace = .natural) async throws -> (samples: [Float], sampleRate: Int) {
        try await serial {
            let speech = try await self.speechModel(model)
            try Task.checkCancellation()
            var refAudio: MLXArray?
            if let reference {
                refAudio = try loadAudioArray(from: reference.audio, sampleRate: speech.sampleRate).1
            }
            var refText = reference?.transcript
            if let ming = speech as? MingDenseModel {
                ming.options.instruction = nil
                if reference != nil, let speed = pace.mingSpeed {
                    // Ming takes a pace only when it copies the voice from the sound alone: the
                    // recording's words would carry its own pace with them.
                    ming.options.instruction = MingText.instruction(speed: speed, pitch: "中", volume: "中")
                    refText = nil
                }
            }
            let audio = try await speech.generate(
                text: text,
                voice: reference == nil ? voice : nil,
                refAudio: refAudio,
                refText: refText,
                language: language ?? Self.language(model),
                generationParameters: speech.defaultGenerationParameters
            )
            let samples = audio.asArray(Float.self)
            Memory.clearCache()
            return (samples, speech.sampleRate)
        }
    }

    private static func language(_ model: LocalVoiceModel) -> String? {
        switch model {
        case .qwen3: return "English"
        case .ming: return "en"
        default: return nil
        }
    }

    private func speechModel(_ model: LocalVoiceModel) async throws -> SpeechGenerationModel {
        if let current = lock.withLock({ loaded }), current.model == model { return current.speech }
        guard store.isDownloaded(model) else {
            throw LocalVoiceError("\(model.info.name) is not downloaded yet.")
        }
        lock.withLock { loaded = nil }
        Memory.clearCache()
        let folder = store.folder(model)
        let speech: SpeechGenerationModel
        switch model {
        case .pocket:
            speech = try await PocketTTSModel.fromModelDirectory(folder)
        case .ming:
            speech = try await MingDenseModel.fromModelDirectory(folder)
        case .qwen3:
            speech = try await Qwen3TTSModel.fromModelDirectory(folder)
        case .audio8:
            speech = try await Audio8MiniLoader.load(folder)
        case .raon:
            speech = try await RaonLoader.load(folder)
        }
        // The first generation compiles the GPU kernels and takes several times longer than the
        // rest. A throwaway word here moves that cost into loading, which happens while the user
        // is still talking.
        if let voice = model.info.voices.first?.id {
            _ = try? await speech.generate(text: "Hi.", voice: voice, refAudio: nil, refText: nil,
                                           language: Self.language(model),
                                           generationParameters: speech.defaultGenerationParameters)
        }
        lock.withLock { loaded = (model, speech) }
        return speech
    }
}
