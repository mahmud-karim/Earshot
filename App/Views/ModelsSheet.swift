import AVFoundation
import SwiftUI

enum ModelsRoute: Hashable {
    case openRouterModel, openAIModel, pcModel
    case appleVoices, openAIVoices, elevenLabsVoice, pcVoice
}

/// Brain, ears and voice for the current preset.
struct ModelsSheet: View {
    @EnvironmentObject var settings: AppSettings
    @EnvironmentObject var conversation: Conversation
    @Environment(\.dismiss) private var dismiss
    @State private var naming = false
    @State private var newName = ""

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 22) {
                    HStack {
                        Text("Models").font(Theme.serif(28)).foregroundStyle(Theme.text)
                        Spacer()
                        Button("Done") { dismiss() }
                            .font(.system(size: 16, weight: .semibold)).foregroundStyle(Theme.clay)
                            .buttonStyle(.plain)
                    }
                    .padding(.top, 8)
                    effort
                    brain
                    ears
                    voice
                    Button {
                        newName = ""
                        naming = true
                    } label: {
                        Text("Save as preset").font(.system(size: 16, weight: .medium)).foregroundStyle(Theme.text)
                            .frame(maxWidth: .infinity).frame(height: 52)
                            .overlay(Capsule().stroke(Theme.border, lineWidth: 0.5))
                    }
                    .buttonStyle(.plain)
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 34)
            }
            .background(Theme.ground.ignoresSafeArea())
            .toolbar(.hidden, for: .navigationBar)
            .navigationDestination(for: ModelsRoute.self) { route in
                destination(route)
            }
        }
        .presentationDragIndicator(.visible)
        .presentationBackground(Theme.ground)
        .alert("Name this preset", isPresented: $naming) {
            TextField("Name", text: $newName)
            Button("Save") {
                var preset = settings.preset
                preset.id = UUID()
                preset.name = newName.isEmpty ? "My preset" : newName
                settings.data.presets.append(preset)
                settings.data.currentPresetID = preset.id
            }
            Button("Cancel", role: .cancel) {}
        }
    }

    // MARK: - Sections

    private var effort: some View {
        VStack(spacing: 10) {
            GroupLabel(text: "How hard it thinks")
            Segmented(options: Effort.allCases.map { ($0, $0.label) }, selection: $settings.preset.effort, height: 40)
            Text(effortNote).font(.system(size: 14)).foregroundStyle(Theme.muted)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var effortNote: String {
        guard settings.preset.brain.kind == .anthropic else {
            return "Only Claude models use this setting."
        }
        switch settings.preset.effort {
        case .low: return "Fast: answers right away with little thinking."
        case .medium: return "Balanced: thinks a little first. Good for most questions."
        case .high: return "Deep: thinks it through. Slower, and costs more."
        }
    }

    private var brain: some View {
        let current = settings.preset.brain
        return VStack(spacing: 10) {
            GroupLabel(text: "Brain · what answers")
            RowGroup {
                ForEach(Catalog.anthropicModels) { model in
                    row(model.name, "Anthropic · \(model.note)", selected: current.kind == .anthropic && current.model == model.id) {
                        settings.preset.brain = BrainChoice(kind: .anthropic, model: model.id)
                    }
                    Hairline()
                }
                row("OpenRouter", keyNote(.openRouter, settings.data.openRouterModel),
                    selected: current.kind == .openRouter, route: .openRouterModel) {
                    settings.preset.brain = BrainChoice(kind: .openRouter, model: settings.data.openRouterModel)
                }
                Hairline()
                row("OpenAI", keyNote(.openAI, settings.data.openAIModel),
                    selected: current.kind == .openAI, route: .openAIModel) {
                    settings.preset.brain = BrainChoice(kind: .openAI, model: settings.data.openAIModel)
                }
                Hairline()
                row("Ollama on your PC", pcNote(settings.data.pcModel),
                    selected: current.kind == .pc, route: .pcModel) {
                    settings.preset.brain = BrainChoice(kind: .pc, model: settings.data.pcModel)
                }
            }
        }
    }

    private var ears: some View {
        VStack(spacing: 10) {
            GroupLabel(text: "Ears · speech to text")
            RowGroup {
                row("Apple on-device", "Free · instant · private", selected: settings.data.ears == .apple) {
                    settings.data.ears = .apple
                }
                Hairline()
                row("OpenAI", settings.key(.openAI) == nil ? "Add an OpenAI key in Settings" : "Uses your OpenAI key · \(settings.data.openAITranscribeModel)",
                    selected: settings.data.ears == .openAI) {
                    settings.data.ears = .openAI
                }
                Hairline()
                row("Whisper on your PC", pcNote(settings.data.pcTranscribeModel), selected: settings.data.ears == .pc) {
                    settings.data.ears = .pc
                }
            }
        }
    }

    private var voice: some View {
        let current = settings.preset.voice
        return VStack(spacing: 10) {
            GroupLabel(text: "Voice · text to speech")
            RowGroup {
                voiceRow("Apple · " + (AppleVoices.name(for: settings.data.appleVoice) ?? "Default"), "Free · instant",
                         VoiceChoice(kind: .apple, voice: settings.data.appleVoice), selected: current.kind == .apple, route: .appleVoices)
                Hairline()
                voiceRow("OpenAI · " + settings.data.openAIVoice.capitalized,
                         settings.key(.openAI) == nil ? "Add an OpenAI key in Settings" : "Uses your OpenAI key",
                         VoiceChoice(kind: .openAI, voice: settings.data.openAIVoice), selected: current.kind == .openAI, route: .openAIVoices)
                Hairline()
                voiceRow("ElevenLabs", settings.key(.elevenLabs) == nil ? "Add an ElevenLabs key in Settings" : "Most natural · uses your key",
                         VoiceChoice(kind: .elevenLabs, voice: settings.data.elevenLabsVoiceID), selected: current.kind == .elevenLabs, route: .elevenLabsVoice)
                Hairline()
                voiceRow("Kokoro on your PC", pcNote(settings.data.pcVoice),
                         VoiceChoice(kind: .pc, voice: settings.data.pcVoice), selected: current.kind == .pc, route: .pcVoice)
            }
        }
    }

    // MARK: - Rows

    private func row(_ title: String, _ subtitle: String, selected: Bool, route: ModelsRoute? = nil, select: @escaping () -> Void) -> some View {
        HStack(spacing: 0) {
            Button(action: select) {
                RowLabel(title: title, subtitle: subtitle) {
                    if selected { Checkmark() }
                }
            }
            .buttonStyle(.plain)
            if let route {
                NavigationLink(value: route) {
                    Chevron().frame(width: 44, height: 56)
                }
                .accessibilityLabel("Change \(title)")
            }
        }
    }

    private func voiceRow(_ title: String, _ subtitle: String, _ choice: VoiceChoice, selected: Bool, route: ModelsRoute) -> some View {
        HStack(spacing: 0) {
            Button {
                settings.preset.voice = choice
            } label: {
                RowLabel(title: title, subtitle: subtitle) {
                    if selected { Checkmark() }
                }
            }
            .buttonStyle(.plain)
            PlayButton { conversation.preview(choice) }
            NavigationLink(value: route) {
                Chevron().frame(width: 40, height: 56)
            }
            .accessibilityLabel("Change \(title)")
        }
    }

    private func keyNote(_ key: KeyName, _ model: String) -> String {
        settings.key(key) == nil ? "Add a key in Settings · \(model)" : model
    }

    private func pcNote(_ model: String) -> String {
        settings.data.pcAddress.isEmpty ? "Set your PC's address in Settings" : model
    }

    // MARK: - Pages

    @ViewBuilder private func destination(_ route: ModelsRoute) -> some View {
        switch route {
        case .openRouterModel:
            TextPage(title: "OpenRouter model", hint: "Any model id from openrouter.ai/models, like anthropic/claude-sonnet-5-5 or google/gemini-3-flash.",
                     text: modelBinding(\.openRouterModel, kind: .openRouter))
        case .openAIModel:
            TextPage(title: "OpenAI model", hint: "A chat model id, like gpt-5-mini.", text: modelBinding(\.openAIModel, kind: .openAI))
        case .pcModel:
            TextPage(title: "Model on your PC", hint: "The name Ollama or LM Studio shows, like llama3.2 or qwen3:8b. The chat server port is set in Settings.",
                     text: modelBinding(\.pcModel, kind: .pc))
        case .appleVoices:
            AppleVoicePage()
        case .openAIVoices:
            ListPage(title: "OpenAI voice", options: Catalog.openAIVoices, selection: voiceBinding(\.openAIVoice, kind: .openAI)) { name in
                conversation.preview(VoiceChoice(kind: .openAI, voice: name))
            }
        case .elevenLabsVoice:
            TextPage(title: "ElevenLabs voice", hint: "Paste a voice id from your ElevenLabs voice library. Empty uses Rachel.",
                     text: voiceBinding(\.elevenLabsVoiceID, kind: .elevenLabs))
        case .pcVoice:
            TextPage(title: "Kokoro voice", hint: "A Kokoro voice name, like af_heart, af_bella or am_michael.",
                     text: voiceBinding(\.pcVoice, kind: .pc))
        }
    }

    /// Edits the remembered model of a provider, and the preset's when it uses that provider.
    private func modelBinding(_ path: WritableKeyPath<SettingsData, String>, kind: BrainKind) -> Binding<String> {
        Binding(get: { settings.data[keyPath: path] }, set: { value in
            settings.data[keyPath: path] = value
            if settings.preset.brain.kind == kind { settings.preset.brain.model = value }
        })
    }

    private func voiceBinding(_ path: WritableKeyPath<SettingsData, String>, kind: VoiceKind) -> Binding<String> {
        Binding(get: { settings.data[keyPath: path] }, set: { value in
            settings.data[keyPath: path] = value
            settings.preset.voice = VoiceChoice(kind: kind, voice: value)
        })
    }
}

/// A page with one text field.
struct TextPage: View {
    let title: String
    let hint: String
    @Binding var text: String
    @Environment(\.dismiss) private var dismiss
    @State private var draft = ""
    @FocusState private var focused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            PageHeader(title: title, back: { dismiss() }) { EmptyView() }
            TextField("", text: $draft)
                .focused($focused)
                .font(.system(size: 16)).foregroundStyle(Theme.text)
                .textInputAutocapitalization(.never).autocorrectionDisabled()
                .padding(.horizontal, 14).frame(height: 48)
                .background(RoundedRectangle(cornerRadius: 12).fill(Theme.surface))
                .onSubmit(save)
            Text(hint).font(.system(size: 14)).foregroundStyle(Theme.muted)
            PrimaryPill(title: "Save", systemImage: "checkmark", action: save)
            Spacer()
        }
        .padding(20)
        .background(Theme.ground.ignoresSafeArea())
        .toolbar(.hidden, for: .navigationBar)
        .onAppear {
            draft = text
            focused = true
        }
    }

    private func save() {
        text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        dismiss()
    }
}

/// A list of names with a check on the chosen one.
struct ListPage: View {
    let title: String
    let options: [String]
    @Binding var selection: String
    let play: (String) -> Void
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        ScrollView {
            VStack(spacing: 16) {
                PageHeader(title: title, back: { dismiss() }) { EmptyView() }
                RowGroup {
                    ForEach(options, id: \.self) { option in
                        HStack(spacing: 0) {
                            Button {
                                selection = option
                            } label: {
                                RowLabel(title: option.capitalized) {
                                    if option == selection { Checkmark() }
                                }
                            }
                            .buttonStyle(.plain)
                            PlayButton { play(option) }.padding(.trailing, 10)
                        }
                        if option != options.last { Hairline() }
                    }
                }
            }
            .padding(20)
        }
        .background(Theme.ground.ignoresSafeArea())
        .toolbar(.hidden, for: .navigationBar)
    }
}

struct AppleVoicePage: View {
    @EnvironmentObject var settings: AppSettings
    @EnvironmentObject var conversation: Conversation
    @Environment(\.dismiss) private var dismiss
    private let voices = AppleVoices.english()

    var body: some View {
        ScrollView {
            VStack(spacing: 16) {
                PageHeader(title: "Apple voice", back: { dismiss() }) { EmptyView() }
                Text("Better voices can be downloaded in iOS Settings › Accessibility › Read & Speak › Voices.")
                    .font(.system(size: 14)).foregroundStyle(Theme.muted).frame(maxWidth: .infinity, alignment: .leading)
                RowGroup {
                    ForEach(voices, id: \.identifier) { voice in
                        HStack(spacing: 0) {
                            Button {
                                settings.data.appleVoice = voice.identifier
                                settings.preset.voice = VoiceChoice(kind: .apple, voice: voice.identifier)
                            } label: {
                                RowLabel(title: voice.name, subtitle: "\(AppleVoices.quality(voice)) · \(voice.language)") {
                                    if AppleVoices.best(settings.data.appleVoice)?.identifier == voice.identifier { Checkmark() }
                                }
                            }
                            .buttonStyle(.plain)
                            PlayButton { conversation.preview(VoiceChoice(kind: .apple, voice: voice.identifier)) }
                                .padding(.trailing, 10)
                        }
                        Hairline()
                    }
                }
            }
            .padding(20)
        }
        .background(Theme.ground.ignoresSafeArea())
        .toolbar(.hidden, for: .navigationBar)
    }
}

struct PlayButton: View {
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: "play.fill").font(.system(size: 11)).foregroundStyle(Theme.secondary)
                .frame(width: 32, height: 32).background(Circle().fill(Theme.raised))
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Play a sample")
    }
}
