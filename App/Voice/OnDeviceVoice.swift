import AVFoundation
import Foundation
import LocalVoices
import UIKit

// Voices that run on the iPhone. The models themselves are in the LocalVoices package; this file ties
// them to the app: which voice a preset means, the recording a voice is copied from, downloads, and
// the Synthesizer the speech queue uses.

/// A preset's on-device voice, stored in VoiceChoice.voice as "model/voice", e.g. "pocket/alba".
struct OnDeviceVoice: Equatable {
    var model: LocalVoiceModel
    /// A built-in voice id, or one of the recordings: "ref:alba" and "ref:kyon" ship with the app,
    /// "ref:mine" is the user's own.
    var voice: String

    static let builtInReference = "ref:alba"
    static let kyonReference = "ref:kyon"
    static let myReference = "ref:mine"

    init(model: LocalVoiceModel, voice: String) {
        self.model = model
        self.voice = voice
    }

    init?(_ stored: String) {
        let parts = stored.split(separator: "/", maxSplits: 1).map(String.init)
        guard let first = parts.first, let model = LocalVoiceModel(rawValue: first) else { return nil }
        self.model = model
        voice = parts.count > 1 ? parts[1] : OnDeviceVoice.defaultVoice(model)
    }

    var stored: String { model.rawValue + "/" + voice }

    static func defaultVoice(_ model: LocalVoiceModel) -> String {
        model.info.needsReference ? builtInReference : LocalVoiceCatalog.defaultVoice(model)
    }

    var voiceName: String {
        switch voice {
        case OnDeviceVoice.builtInReference: return "Alba recording"
        case OnDeviceVoice.kyonReference: return "Kyon"
        case OnDeviceVoice.myReference: return "My voice"
        default: return model.info.voices.first(where: { $0.id == voice })?.name ?? voice
        }
    }

    var label: String { model.info.name + " · " + voiceName }

    /// The recording to copy, for the voices that are recordings.
    var reference: VoiceReference? {
        switch voice {
        case OnDeviceVoice.builtInReference: return VoiceSamples.builtIn
        case OnDeviceVoice.kyonReference: return VoiceSamples.kyon
        case OnDeviceVoice.myReference: return VoiceSamples.mine
        default: return nil
        }
    }
}

/// Recordings that cloning models copy a voice from.
enum VoiceSamples {
    /// Read aloud when the user records their own voice.
    static let readAloud = "Hi, I'm recording a sample of my voice so Earshot can speak like me. I'm reading this at my normal pace, clearly and calmly."

    /// A clip Pocket TTS made in its Alba voice, shipped with the app.
    static var builtIn: VoiceReference? { bundled("alba") }

    /// A clean clip in Kyon's voice, made with Breeze TTS 2, shipped with the app.
    static var kyon: VoiceReference? { bundled("kyon") }

    private static func bundled(_ name: String) -> VoiceReference? {
        guard let audio = Bundle.main.url(forResource: name, withExtension: "wav"),
              let textURL = Bundle.main.url(forResource: name, withExtension: "txt"),
              let text = try? String(contentsOf: textURL, encoding: .utf8) else { return nil }
        return VoiceReference(audio: audio, transcript: text.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    static var myFile: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("MyVoice.wav")
    }

    static var mine: VoiceReference? {
        guard FileManager.default.fileExists(atPath: myFile.path) else { return nil }
        return VoiceReference(audio: myFile, transcript: readAloud)
    }
}

/// Speaks with a model on the phone. iOS does not let an app use the GPU in the background, so a
/// reply that arrives while Earshot is not in front is spoken by Apple's voice instead.
struct OnDeviceSynthesizer: Synthesizer {
    let choice: OnDeviceVoice
    let fallback: AppleSynthesizer

    func audio(for text: String) async throws -> [AVAudioPCMBuffer] {
        let active = await MainActor.run { UIApplication.shared.applicationState == .active }
        guard active else { return try await fallback.audio(for: text) }
        var reference = choice.reference
        if choice.model.info.needsReference && reference == nil {
            reference = VoiceSamples.builtIn
        }
        let (samples, rate) = try await LocalVoiceEngine.shared.speak(
            text, model: choice.model, voice: reference == nil ? choice.voice : nil, reference: reference)
        guard !samples.isEmpty,
              let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(rate), channels: 1, interleaved: false),
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)),
              let channel = buffer.floatChannelData?[0] else { return [] }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { channel.update(from: $0.baseAddress!, count: samples.count) }
        return [buffer]
    }
}

/// Download state for each model, shown on the voices page.
@MainActor
final class LocalVoiceDownloads: ObservableObject {
    static let shared = LocalVoiceDownloads()

    struct Progress: Equatable {
        var done: Int64
        var total: Int64
        var fraction: Double { total > 0 ? Double(done) / Double(total) : 0 }
    }

    @Published private(set) var progress: [LocalVoiceModel: Progress] = [:]
    @Published private(set) var failures: [LocalVoiceModel: String] = [:]
    /// Bumped when a model is added or removed so views re-read the disk.
    @Published private(set) var version = 0
    private var tasks: [LocalVoiceModel: Task<Void, Never>] = [:]

    func isDownloaded(_ model: LocalVoiceModel) -> Bool {
        _ = version
        return LocalModelStore.shared.isDownloaded(model)
    }

    func isDownloading(_ model: LocalVoiceModel) -> Bool { tasks[model] != nil }

    func start(_ model: LocalVoiceModel) {
        guard tasks[model] == nil else { return }
        failures[model] = nil
        progress[model] = Progress(done: 0, total: model.info.approximateBytes)
        // The screen may lock during a long download; it carries on while iOS allows.
        tasks[model] = Task {
            do {
                try await LocalModelStore.shared.download(model) { done, total in
                    Task { @MainActor in self.progress[model] = Progress(done: done, total: total) }
                }
            } catch is CancellationError {
            } catch {
                failures[model] = error.localizedDescription
            }
            tasks[model] = nil
            progress[model] = nil
            version += 1
        }
    }

    func cancel(_ model: LocalVoiceModel) {
        tasks[model]?.cancel()
    }

    /// The "My voice" recording was saved or deleted; the voice lists read it from disk.
    func recordingChanged() {
        version += 1
    }

    func delete(_ model: LocalVoiceModel) {
        Task {
            if LocalVoiceEngine.shared.loadedModel == model { await LocalVoiceEngine.shared.unload() }
            try? LocalModelStore.shared.delete(model)
            version += 1
        }
    }
}

/// Collects the microphone while the user reads the sample sentence. It plugs into the audio router
/// the way a transcriber does.
final class SampleRecorder: Transcriber {
    var onText: ((String) -> Void)?
    var onStatus: ((String?) -> Void)?
    let givesLiveText = false
    private let lock = NSLock()
    private var samples: [Float] = []

    func start() async throws {}

    func feed(_ samples: [Float]) {
        lock.withLock { self.samples.append(contentsOf: samples) }
    }

    var seconds: Double { lock.withLock { Double(samples.count) / 16_000 } }

    /// Writes the recording, trimmed of leading and trailing quiet.
    func finish() async throws -> String {
        let all = lock.withLock { samples }
        let loud = all.indices.filter { abs(all[$0]) > 0.02 }
        guard let first = loud.first, let last = loud.last, last - first > 16_000 * 2 else {
            throw BrainError(message: "That was too quiet or too short. Try again a little closer to the phone.")
        }
        let start = max(0, first - 4_000)
        let end = min(all.count, last + 4_000)
        try WAV.encode(Array(all[start..<end]), sampleRate: 16_000).write(to: VoiceSamples.myFile)
        return VoiceSamples.readAloud
    }

    func cancel() {}
}
