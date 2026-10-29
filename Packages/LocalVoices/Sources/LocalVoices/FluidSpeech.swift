@preconcurrency import CoreML
import FluidAudio
import Foundation

/// Fluid Speech Mini (FluidVoice Labs, CC BY 4.0): English speech to text on the Neural Engine, in the
/// format of Parakeet TDT 0.6B v3. About 253 MB, pinned to one revision of the Core ML repository.
public struct FluidSpeechMini: DownloadableModel {
    public static let model = FluidSpeechMini()
    public static let approximateBytes: Int64 = 252_700_000

    public let folderName = "fluid-speech-mini"
    public let displayName = "Fluid Speech Mini"
    public let sources = [LocalModelSource(
        repo: "FluidVoice-Labs/fluid-speech-mini-coreml",
        revision: "d81c3160dfe5a5e1f8f7267164a07d30617a3a1a",
        files: ["Preprocessor", "Encoder", "Decoder", "JointDecision"].flatMap { name in
            ["analytics/coremldata.bin", "coremldata.bin", "metadata.json", "model.mil", "weights/weight.bin"]
                .map { "\(name).mlmodelc/\($0)" }
        } + ["parakeet_vocab.json"]
    )]
}

/// Loads Fluid Speech Mini once and transcribes 16 kHz mono samples with it. The pinned FluidAudio runs
/// it as its Parakeet v3; its own v3 loader expects other file names, so the four models are loaded here
/// and handed over ready-made, as LiveCue Remake does. No language hint is passed: with one, the package
/// reads outputs this model's joint network does not have.
public actor FluidSpeech {
    public static let shared = FluidSpeech()

    private var manager: AsrManager?

    public var isLoaded: Bool { manager != nil }

    /// Loads the models. The first load after a download is slow: iOS prepares them for the Neural Engine.
    public func load() async throws {
        _ = try await loaded()
    }

    /// The words in `samples`. Shorter than the model's minimum gives an empty string; longer than 15
    /// seconds is cut into windows by the package.
    public func transcribe(_ samples: [Float]) async throws -> String {
        let manager = try await loaded()
        guard samples.count >= ASRConstants.minimumRequiredSamples(forSampleRate: 16_000) else { return "" }
        var state = TdtDecoderState.make(decoderLayers: await manager.decoderLayerCount)
        let result = try await manager.transcribe(samples, decoderState: &state)
        return result.text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    public func unload() async {
        await manager?.cleanup()
        manager = nil
    }

    private func loaded() async throws -> AsrManager {
        if let manager { return manager }
        let store = LocalModelStore.shared
        guard store.isDownloaded(FluidSpeechMini.model) else {
            throw LocalVoiceError("Fluid Speech Mini isn't downloaded yet.")
        }
        let folder = store.folder(FluidSpeechMini.model)
        let models = AsrModels(
            encoder: try Self.model("Encoder", in: folder, units: .cpuAndNeuralEngine),
            preprocessor: try Self.model("Preprocessor", in: folder, units: .cpuOnly),
            decoder: try Self.model("Decoder", in: folder, units: .cpuAndNeuralEngine),
            joint: try Self.model("JointDecision", in: folder, units: .cpuAndNeuralEngine),
            configuration: MLModelConfigurationUtils.defaultConfiguration(computeUnits: .cpuAndNeuralEngine),
            vocabulary: try Self.vocabulary(folder.appendingPathComponent("parakeet_vocab.json")),
            version: .v3
        )
        let manager = AsrManager()
        try await manager.loadModels(models)
        self.manager = manager
        return manager
    }

    private static func model(_ name: String, in folder: URL, units: MLComputeUnits) throws -> MLModel {
        try MLModel(contentsOf: folder.appendingPathComponent(name + ".mlmodelc", isDirectory: true),
                    configuration: MLModelConfigurationUtils.defaultConfiguration(computeUnits: units))
    }

    /// The vocabulary file maps each token's number, written as text, to its piece.
    private static func vocabulary(_ url: URL) throws -> [Int: String] {
        guard let pieces = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: String] else {
            throw LocalVoiceError("The vocabulary of Fluid Speech Mini could not be read.")
        }
        let vocabulary = Dictionary(uniqueKeysWithValues: pieces.compactMap { key, piece in Int(key).map { ($0, piece) } })
        guard !vocabulary.isEmpty else { throw LocalVoiceError("The vocabulary of Fluid Speech Mini is empty.") }
        return vocabulary
    }
}
