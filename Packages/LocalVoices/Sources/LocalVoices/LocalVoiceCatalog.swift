import Foundation

/// How fast Ming speaks a copied voice. `natural` keeps the recording's own pace.
public enum LocalVoicePace: String, CaseIterable, Codable, Sendable {
    case natural, slow, medium, fast

    /// The publisher's speed values for Ming-3.0's basic controls.
    var mingSpeed: String? {
        switch self {
        case .natural: return nil
        case .slow: return "慢速"
        case .medium: return "中速"
        case .fast: return "快速"
        }
    }
}

/// A speech model that runs on the phone. Each is downloaded once from Hugging Face and kept in the
/// app's own storage.
public enum LocalVoiceModel: String, CaseIterable, Codable, Sendable {
    case pocket
    case audio8
    case ming
    case qwen3
    case raon

    public var info: LocalVoiceInfo { LocalVoiceCatalog.info(self) }
}

/// One voice a model can speak in.
public struct LocalVoiceOption: Identifiable, Hashable, Sendable {
    public let id: String
    public let name: String
    public let detail: String

    public init(id: String, name: String, detail: String) {
        self.id = id
        self.name = name
        self.detail = detail
    }
}

/// A recording the model copies a voice from, and the words spoken in it.
public struct VoiceReference: Hashable, Sendable {
    public let audio: URL
    public let transcript: String

    public init(audio: URL, transcript: String) {
        self.audio = audio
        self.transcript = transcript
    }
}

/// Files from one Hugging Face repository, pinned to the commit the app was tested against.
public struct LocalModelSource: Sendable {
    public let repo: String
    public let revision: String
    /// Repository paths to fetch. Empty takes every file.
    public let files: [String]
    /// Subfolder of the model folder the files land in.
    public let folder: String

    public init(repo: String, revision: String, files: [String] = [], folder: String = "") {
        self.repo = repo
        self.revision = revision
        self.files = files
        self.folder = folder
    }
}

public struct LocalVoiceInfo: Sendable {
    public let model: LocalVoiceModel
    public let name: String
    public let maker: String
    public let parameters: String
    public let license: String
    /// Where the files come from on Hugging Face.
    public let sources: [LocalModelSource]
    /// What the download weighs, for the button before the file list is fetched.
    public let approximateBytes: Int64
    /// Built-in voices. A model without any copies a voice from a recording.
    public let voices: [LocalVoiceOption]
    public let clonesVoices: Bool
    public let note: String

    public var needsReference: Bool { voices.isEmpty }
}

public enum LocalVoiceCatalog {
    /// The voice used when nothing is chosen yet.
    public static func defaultVoice(_ model: LocalVoiceModel) -> String {
        info(model).voices.first?.id ?? "reference"
    }

    public static func info(_ model: LocalVoiceModel) -> LocalVoiceInfo {
        switch model {
        case .pocket:
            return LocalVoiceInfo(
                model: model, name: "Pocket TTS", maker: "Kyutai", parameters: "100M", license: "CC BY 4.0",
                sources: [LocalModelSource(repo: "mlx-community/pocket-tts", revision: "cbf71d5f6657bbc3f4bc02f85ee408261225bec7")], approximateBytes: 240_000_000,
                voices: [
                    LocalVoiceOption(id: "alba", name: "Alba", detail: ""),
                    LocalVoiceOption(id: "marius", name: "Marius", detail: ""),
                    LocalVoiceOption(id: "jean", name: "Jean", detail: ""),
                    LocalVoiceOption(id: "javert", name: "Javert", detail: ""),
                    LocalVoiceOption(id: "fantine", name: "Fantine", detail: ""),
                    LocalVoiceOption(id: "cosette", name: "Cosette", detail: ""),
                    LocalVoiceOption(id: "eponine", name: "Eponine", detail: ""),
                    LocalVoiceOption(id: "azelma", name: "Azelma", detail: "")
                ],
                clonesVoices: true,
                note: "Smallest and quickest. English.")
        case .audio8:
            return LocalVoiceInfo(
                model: model, name: "Audio8 TTS 0.1B", maker: "Audio8", parameters: "170M", license: "Audio8 Community",
                sources: [LocalModelSource(repo: "mlx-community/Audio8-TTS-Preview-0.1b-bf16", revision: "084df8928ee718eab19afe4329f0e5a9ce180133")], approximateBytes: 1_700_000_000,
                voices: [],
                clonesVoices: true,
                note: "Copies a voice from a short recording. English and Chinese, others experimental.")
        case .ming:
            return LocalVoiceInfo(
                model: model, name: "Ming-3.0 TTS 0.5B", maker: "Ant Group", parameters: "0.5B", license: "MIT",
                sources: [LocalModelSource(repo: "mlx-community/Ming-3.0-tts-0.5B-bf16", revision: "283b3c342c6fb69a7a87832c97f9799567efe05d")], approximateBytes: 3_000_000_000,
                voices: [
                    LocalVoiceOption(id: "US_Emma", name: "Emma", detail: "Female, American"),
                    LocalVoiceOption(id: "UK_George", name: "George", detail: "Male, British"),
                    LocalVoiceOption(id: "UK_Charlotte", name: "Charlotte", detail: "Female, British"),
                    LocalVoiceOption(id: "US_James", name: "James", detail: "Male, American")
                ],
                clonesVoices: true,
                note: "Best quality for its size. Needs about 4.5 GB of memory while it speaks.")
        case .qwen3:
            return LocalVoiceInfo(
                model: model, name: "Qwen3-TTS 0.6B", maker: "Alibaba Qwen", parameters: "0.6B", license: "Apache 2.0",
                sources: [LocalModelSource(repo: "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit", revision: "049ef77fe8816b536193c0c25f9a214d17921282")], approximateBytes: 1_970_000_000,
                voices: [
                    LocalVoiceOption(id: "ryan", name: "Ryan", detail: "Male, English"),
                    LocalVoiceOption(id: "aiden", name: "Aiden", detail: "Male, English"),
                    LocalVoiceOption(id: "serena", name: "Serena", detail: "Female, Chinese"),
                    LocalVoiceOption(id: "vivian", name: "Vivian", detail: "Female, Chinese"),
                    LocalVoiceOption(id: "ono_anna", name: "Ono Anna", detail: "Female, Japanese"),
                    LocalVoiceOption(id: "sohee", name: "Sohee", detail: "Female, Korean")
                ],
                clonesVoices: false,
                note: "Named voices in 10 languages.")
        case .raon:
            return LocalVoiceInfo(
                model: model, name: "Raon-OpenTTS 1B", maker: "KRAFTON", parameters: "1B", license: "CC BY-NC 4.0",
                sources: [
                    LocalModelSource(repo: "drbaph/Raon-OpenTTS-comfyui", revision: "0ebfe0f8a18a3639f749c17b90e910d6290403bb", files: [
                        "Raon-OpenTTS-1B/Raon-OpenTTS-1B-bf16.safetensors",
                        "Raon-OpenTTS-1B/config.yaml",
                        "Raon-OpenTTS-1B/vocab.txt"
                    ]),
                    LocalModelSource(repo: "speechbrain/tts-hifigan-libritts-16kHz", revision: "4d32c6f19e7ff9316c5c56d7903d3d345ddcace4", files: [
                        "generator.ckpt", "hyperparams.yaml"
                    ], folder: "vocoder")
                ],
                approximateBytes: 2_150_000_000,
                voices: [],
                clonesVoices: true,
                note: "Largest. Copies a voice from a short recording. English only, personal use.")
        }
    }
}
