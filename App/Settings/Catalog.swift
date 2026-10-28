import Foundation

// What can answer, listen and speak. Each choice is plain data; the factories in Conversation turn a
// choice into a working provider.

/// Where a model runs and which key it uses.
enum BrainKind: String, Codable, CaseIterable {
    case anthropic
    case openRouter
    case openAI
    case pc
}

struct BrainChoice: Codable, Equatable, Hashable {
    var kind: BrainKind
    /// The provider's own model id, e.g. claude-sonnet-5-5.
    var model: String

    var label: String {
        if kind == .anthropic, let known = Catalog.anthropicModels.first(where: { $0.id == model }) {
            return known.name
        }
        return model.isEmpty ? Catalog.brainKindName(kind) : Catalog.shortModelName(model)
    }
}

/// How hard the model thinks. Sent as effort to Anthropic models.
enum Effort: String, Codable, CaseIterable {
    case low
    case medium
    case high

    var label: String {
        switch self {
        case .low: return "Low"
        case .medium: return "Medium"
        case .high: return "High"
        }
    }
}

enum EarsKind: String, Codable, CaseIterable {
    case apple
    case openAI
    case pc
}

enum VoiceKind: String, Codable, CaseIterable {
    case apple
    case openAI
    case elevenLabs
    case pc
    case onDevice
}

struct VoiceChoice: Codable, Equatable, Hashable {
    var kind: VoiceKind
    /// Apple: voice identifier (empty picks the best English voice). OpenAI and PC: voice name.
    /// ElevenLabs: voice id. On device: "model/voice", see OnDeviceVoice.
    var voice: String

    var label: String {
        switch kind {
        case .apple: return "Apple · " + (AppleVoices.name(for: voice) ?? "Default")
        case .openAI: return "OpenAI · " + voice.capitalized
        case .elevenLabs: return "ElevenLabs"
        case .pc: return "Kokoro · " + voice
        case .onDevice: return OnDeviceVoice(voice)?.label ?? "On this iPhone"
        }
    }
}

/// A named mix of brain, thinking level and voice.
struct Preset: Codable, Identifiable, Equatable, Hashable {
    var id: UUID = UUID()
    var name: String
    var brain: BrainChoice
    var effort: Effort
    var voice: VoiceChoice

    var summary: String { brain.label + " · " + voice.label }
}

struct AnthropicModel: Identifiable {
    let id: String
    let name: String
    let note: String
    /// The thinking level it starts at until the user picks one.
    let defaultEffort: Effort
    /// US dollars per million tokens.
    let inputPrice: Double
    let outputPrice: Double
}

enum Catalog {
    static let anthropicModels: [AnthropicModel] = [
        AnthropicModel(id: "claude-haiku-5-5", name: "Haiku 5.5", note: "fastest", defaultEffort: .low, inputPrice: 0.10, outputPrice: 0.50),
        AnthropicModel(id: "claude-sonnet-5-5", name: "Sonnet 5.5", note: "balanced", defaultEffort: .medium, inputPrice: 2, outputPrice: 10),
        AnthropicModel(id: "claude-opus-5-5", name: "Opus 5.5", note: "deepest thinking", defaultEffort: .high, inputPrice: 4, outputPrice: 20)
    ]

    static let openAIVoices = ["alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer", "verse"]

    static let defaultPresets: [Preset] = [
        Preset(name: "Quick question",
               brain: BrainChoice(kind: .anthropic, model: "claude-haiku-5-5"),
               effort: .low,
               voice: VoiceChoice(kind: .apple, voice: "")),
        Preset(name: "Balanced",
               brain: BrainChoice(kind: .anthropic, model: "claude-sonnet-5-5"),
               effort: .medium,
               voice: VoiceChoice(kind: .apple, voice: "")),
        Preset(name: "Think hard",
               brain: BrainChoice(kind: .anthropic, model: "claude-opus-5-5"),
               effort: .high,
               voice: VoiceChoice(kind: .apple, voice: ""))
    ]

    static func brainKindName(_ kind: BrainKind) -> String {
        switch kind {
        case .anthropic: return "Anthropic"
        case .openRouter: return "OpenRouter"
        case .openAI: return "OpenAI"
        case .pc: return "Your PC"
        }
    }

    /// "openai/gpt-5-mini" reads as "gpt-5-mini".
    static func shortModelName(_ model: String) -> String {
        model.split(separator: "/").last.map(String.init) ?? model
    }

    /// Dollars per web search, on top of tokens ($10 per 1,000).
    static let searchPrice = 0.01

    /// Dollars for one reply, when the price is known.
    static func cost(model: String, input: Int, output: Int) -> Double? {
        guard let known = anthropicModels.first(where: { $0.id == model }) else { return nil }
        return (Double(input) * known.inputPrice + Double(output) * known.outputPrice) / 1_000_000
    }
}
