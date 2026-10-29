import FluidAudio
import Foundation

/// CAM++ (FunASR, via FluidInference's Core ML conversion): turns speech into a 192-number voiceprint.
/// Two prints of the same person score high on cosine similarity, different people low. About 16 MB,
/// pinned to one revision.
public struct CamPlusModel: DownloadableModel {
    public static let model = CamPlusModel()
    public static let approximateBytes: Int64 = 16_640_000

    public let folderName = "campplus"
    public let displayName = "Voiceprint (CAM++)"
    public let sources = [LocalModelSource(
        repo: "FluidInference/campplus-coreml",
        revision: "321b18e270a4e19e69ef4a13d179b4eac07cec6a",
        files: ["CamPlusPreprocessor", "CamPlusPlus"].flatMap { name in
            ["analytics/coremldata.bin", "coremldata.bin", "model.mil", "weights/weight.bin"]
                .map { "\(name).mlmodelc/\($0)" }
        }
    )]
}

/// Voiceprints of 16 kHz mono speech, and how alike two of them are.
public actor SpeakerPrint {
    public static let shared = SpeakerPrint()

    /// Shorter speech gives a print too unreliable to judge by.
    public static let minimumSamples = 16_000

    private var embedder: CampPlusEmbedder?

    public var isLoaded: Bool { embedder != nil }

    public func load() throws {
        _ = try loaded()
    }

    /// The voiceprint of `samples`, or nil when there is too little speech.
    public func print(_ samples: [Float]) async throws -> [Float]? {
        guard samples.count >= Self.minimumSamples else { return nil }
        return try await loaded().embed(audio: samples)
    }

    /// From -1 to 1, higher for the same speaker. On English phone audio different speakers still reach
    /// about 0.5, so a useful bar is well above that.
    public nonisolated static func similarity(_ a: [Float], _ b: [Float]) -> Float {
        CampPlusEmbedder.cosine(a, b)
    }

    public func unload() {
        embedder = nil
    }

    private func loaded() throws -> CampPlusEmbedder {
        if let embedder { return embedder }
        let store = LocalModelStore.shared
        guard store.isDownloaded(CamPlusModel.model) else {
            throw LocalVoiceError("The voiceprint model isn't downloaded yet.")
        }
        let made = CampPlusEmbedder(models: try CampPlusModels.load(from: store.folder(CamPlusModel.model)))
        embedder = made
        return made
    }
}
