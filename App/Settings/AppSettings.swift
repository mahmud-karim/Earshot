import Foundation
import Security
import SwiftUI

/// Everything the user chooses, saved to UserDefaults as one JSON value. Keys live in the Keychain.
struct SettingsData: Codable, Equatable {
    var presets: [Preset] = Catalog.defaultPresets
    var currentPresetID: UUID?
    var ears: EarsKind = .apple
    /// Model names for the non-Anthropic providers, remembered per provider.
    var openRouterModel = "openai/gpt-5-mini"
    var openAIModel = "gpt-5-mini"
    var pcModel = "llama3.2"
    /// The thinking level last picked for each Claude model, by model id.
    var effortByModel: [String: Effort] = [:]
    var openAITranscribeModel = "gpt-4o-mini-transcribe"
    var pcTranscribeModel = "Systran/faster-whisper-small"
    var openAIVoiceModel = "gpt-4o-mini-tts"
    var elevenLabsModel = "eleven_flash_v2_5"
    var pcVoiceModel = "kokoro"
    /// The last voice picked for each provider, so switching back keeps it.
    var appleVoice = ""
    var openAIVoice = "coral"
    var elevenLabsVoiceID = ""
    var pcVoice = "af_heart"
    var onDeviceVoice = "pocket/alba"
    /// Host name or address of the PC, e.g. my-pc.tail1234.ts.net. Empty when there is none.
    var pcAddress = ""
    /// OpenAI-compatible chat server on the PC: Ollama is 11434, LM Studio 1234.
    var pcChatPort = 11434
    /// OpenAI-compatible speech server on the PC (speaches: Whisper and Kokoro).
    var pcSpeechPort = 8000

    var handsFree = false
    var interruptByTalking = true
    var pauseSeconds = 1.2
    var liveCaptions = true
    var listenOnOpen = true
    var monthlyBudget = 100.0
}

enum KeyName: String, CaseIterable, Identifiable {
    case anthropic, openRouter, openAI, elevenLabs

    var id: String { rawValue }

    var label: String {
        switch self {
        case .anthropic: return "Anthropic"
        case .openRouter: return "OpenRouter"
        case .openAI: return "OpenAI"
        case .elevenLabs: return "ElevenLabs"
        }
    }

    var hint: String {
        switch self {
        case .anthropic: return "From platform.claude.com, starts with sk-ant-"
        case .openRouter: return "From openrouter.ai/keys, starts with sk-or-"
        case .openAI: return "From platform.openai.com, starts with sk-"
        case .elevenLabs: return "From elevenlabs.io, under API keys"
        }
    }
}

@MainActor
final class AppSettings: ObservableObject {
    @Published var data: SettingsData {
        didSet { save() }
    }
    /// Bumped when a key changes so views that show key state refresh.
    @Published private(set) var keysVersion = 0

    private let defaultsKey = "settings.v1"
    private let keychain = Keychain(service: "com.mahmudkarim.Earshot.keys")

    init() {
        data = AppSettings.load(UserDefaults.standard.data(forKey: defaultsKey))
        if data.presets.isEmpty {
            data.presets = Catalog.defaultPresets
        }
        if data.currentPresetID == nil || !data.presets.contains(where: { $0.id == data.currentPresetID }) {
            data.currentPresetID = data.presets[min(1, data.presets.count - 1)].id
        }
    }

    var preset: Preset {
        get { data.presets.first(where: { $0.id == data.currentPresetID }) ?? data.presets[0] }
        set {
            if let index = data.presets.firstIndex(where: { $0.id == newValue.id }) {
                data.presets[index] = newValue
            }
        }
    }

    func key(_ name: KeyName) -> String? {
        let value = keychain.get(name.rawValue)
        return (value?.isEmpty ?? true) ? nil : value
    }

    func setKey(_ name: KeyName, _ value: String) {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            keychain.delete(name.rawValue)
        } else {
            keychain.set(trimmed, for: name.rawValue)
        }
        keysVersion += 1
    }

    /// "•••• 3F2A" or nil.
    func keyHint(_ name: KeyName) -> String? {
        guard let key = key(name) else { return nil }
        return "•••• " + String(key.suffix(4))
    }

    func pcURL(port: Int) -> URL? {
        let host = data.pcAddress.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !host.isEmpty else { return nil }
        if host.hasPrefix("http://") || host.hasPrefix("https://") {
            return URL(string: host)
        }
        return URL(string: "http://\(host):\(port)/v1")
    }

    /// Saved values laid over the defaults, so a setting added in a later version never makes the
    /// saved ones unreadable.
    private static func load(_ raw: Data?) -> SettingsData {
        guard let raw,
              let saved = (try? JSONSerialization.jsonObject(with: raw)) as? [String: Any],
              let defaultsData = try? JSONEncoder().encode(SettingsData()),
              var merged = (try? JSONSerialization.jsonObject(with: defaultsData)) as? [String: Any] else {
            return SettingsData()
        }
        merged.merge(saved) { _, new in new }
        guard let mergedData = try? JSONSerialization.data(withJSONObject: merged),
              let decoded = try? JSONDecoder().decode(SettingsData.self, from: mergedData) else {
            return SettingsData()
        }
        return decoded
    }

    private func save() {
        if let raw = try? JSONEncoder().encode(data) {
            UserDefaults.standard.set(raw, forKey: defaultsKey)
        }
    }
}

/// Generic-password items, this device only.
struct Keychain {
    let service: String

    func set(_ value: String, for account: String) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account
        ]
        let attributes: [String: Any] = [
            kSecValueData as String: Data(value.utf8),
            kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
        ]
        if SecItemUpdate(query as CFDictionary, attributes as CFDictionary) == errSecItemNotFound {
            var item = query
            item.merge(attributes) { _, new in new }
            SecItemAdd(item as CFDictionary, nil)
        }
    }

    func get(_ account: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]
        var found: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &found) == errSecSuccess,
              let data = found as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    func delete(_ account: String) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account
        ]
        SecItemDelete(query as CFDictionary)
    }
}
