import SwiftUI
import UIKit

enum SettingsRoute: Hashable {
    case key(KeyName)
    case preset(UUID)
    case pcAddress
    case actionButton
    case safari
}

struct SettingsView: View {
    @EnvironmentObject var settings: AppSettings
    @EnvironmentObject var spend: SpendMeter
    @Environment(\.dismiss) private var dismiss
    @State private var pcStatus = "Not checked"

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 22) {
                    PageHeader(title: "Settings", back: { dismiss() }) { EmptyView() }
                    presets
                    keys
                    pc
                    conversation
                    shortcuts
                    VStack(spacing: 10) {
                        GroupLabel(text: "Usage")
                        SpendCard(title: "This month", spent: spend.thisMonth, budget: settings.data.monthlyBudget)
                        Text("Counts Claude replies on this iPhone. Your console at platform.claude.com has the exact bill.")
                            .font(.system(size: 13)).foregroundStyle(Theme.muted).frame(maxWidth: .infinity, alignment: .leading)
                    }
                    Text("Earshot · version \(Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "")")
                        .font(.system(size: 12)).foregroundStyle(Theme.muted)
                }
                .padding(.horizontal, 20)
                .padding(.top, 8)
                .padding(.bottom, 34)
            }
            .background(Theme.ground.ignoresSafeArea())
            .toolbar(.hidden, for: .navigationBar)
            .navigationDestination(for: SettingsRoute.self) { route in
                switch route {
                case .key(let name): KeyPage(name: name)
                case .preset(let id): PresetPage(id: id)
                case .pcAddress:
                    TextPage(title: "Your PC", hint: "The PC's Tailscale name or address, like my-pc.tail1234.ts.net or 100.64.0.5. A full URL also works.",
                             text: $settings.data.pcAddress)
                case .actionButton: ActionButtonPage()
                case .safari: SafariPage()
                }
            }
        }
    }

    private var presets: some View {
        VStack(spacing: 10) {
            GroupLabel(text: "Presets")
            RowGroup {
                ForEach(settings.data.presets) { preset in
                    NavigationLink(value: SettingsRoute.preset(preset.id)) {
                        RowLabel(title: preset.name, subtitle: preset.summary) {
                            if preset.id == settings.data.currentPresetID {
                                Text("Current").font(.system(size: 14)).foregroundStyle(Theme.secondary)
                            }
                            Chevron()
                        }
                    }
                    .buttonStyle(.plain)
                    Hairline()
                }
                Button {
                    var preset = settings.preset
                    preset.id = UUID()
                    preset.name = "New preset"
                    settings.data.presets.append(preset)
                } label: {
                    RowLabel(title: "New preset", titleColor: Theme.clay) { EmptyView() }
                }
                .buttonStyle(.plain)
            }
        }
    }

    private var keys: some View {
        VStack(spacing: 10) {
            GroupLabel(text: "API keys · stored in the Keychain")
            RowGroup {
                ForEach(KeyName.allCases) { name in
                    NavigationLink(value: SettingsRoute.key(name)) {
                        RowLabel(title: name.label) {
                            Text(settings.keyHint(name) ?? "Not set").font(.system(size: 14)).foregroundStyle(Theme.secondary)
                            Chevron()
                        }
                    }
                    .buttonStyle(.plain)
                    if name != KeyName.allCases.last { Hairline() }
                }
            }
            .id(settings.keysVersion)
        }
    }

    private var pc: some View {
        VStack(spacing: 10) {
            GroupLabel(text: "Your PC · optional")
            RowGroup {
                NavigationLink(value: SettingsRoute.pcAddress) {
                    RowLabel(title: "Address", subtitle: settings.data.pcAddress.isEmpty ? "Not set" : settings.data.pcAddress) { Chevron() }
                }
                .buttonStyle(.plain)
                Hairline()
                RowLabel(title: "Chat port", subtitle: "Ollama 11434 · LM Studio 1234") {
                    PortField(value: $settings.data.pcChatPort)
                }
                Hairline()
                RowLabel(title: "Speech port", subtitle: "Whisper and Kokoro, e.g. speaches on 8000") {
                    PortField(value: $settings.data.pcSpeechPort)
                }
                Hairline()
                RowLabel(title: "Status") {
                    Text(pcStatus).font(.system(size: 14)).foregroundStyle(Theme.secondary)
                }
                Hairline()
                Button {
                    Task { await testPC() }
                } label: {
                    RowLabel(title: "Test connection", titleColor: Theme.clay) { EmptyView() }
                }
                .buttonStyle(.plain)
            }
        }
    }

    private var conversation: some View {
        VStack(spacing: 10) {
            GroupLabel(text: "Conversation")
            RowGroup {
                RowLabel(title: "Listen when it opens", subtitle: "Starts listening when you open Earshot or press the Action Button") {
                    ClayToggle(isOn: $settings.data.listenOnOpen)
                }
                Hairline()
                RowLabel(title: "Hands-free", subtitle: "Keeps listening after each reply") {
                    ClayToggle(isOn: $settings.data.handsFree)
                }
                Hairline()
                RowLabel(title: "Interrupt by talking", subtitle: "Stops speaking when you start") {
                    ClayToggle(isOn: $settings.data.interruptByTalking)
                }
                Hairline()
                RowLabel(title: "Send after a pause of", subtitle: nil) {
                    Stepper(String(format: "%.1f s", settings.data.pauseSeconds), value: $settings.data.pauseSeconds, in: 0.6...3.0, step: 0.2)
                        .font(.system(size: 14)).foregroundStyle(Theme.secondary).fixedSize()
                }
                Hairline()
                RowLabel(title: "Monthly budget", subtitle: "Warns you at 80%") {
                    Stepper(Money.text(settings.data.monthlyBudget, cents: false), value: $settings.data.monthlyBudget, in: 5...1000, step: 5)
                        .font(.system(size: 14)).foregroundStyle(Theme.secondary).fixedSize()
                }
            }
        }
    }

    private var shortcuts: some View {
        VStack(spacing: 10) {
            GroupLabel(text: "Shortcuts")
            RowGroup {
                NavigationLink(value: SettingsRoute.actionButton) {
                    RowLabel(title: "Action Button", subtitle: "Screenshot, then ask about it") { Chevron() }
                }
                .buttonStyle(.plain)
                Hairline()
                NavigationLink(value: SettingsRoute.safari) {
                    RowLabel(title: "Safari share sheet", subtitle: "Ask about this page") { Chevron() }
                }
                .buttonStyle(.plain)
            }
        }
    }

    private func testPC() async {
        guard let url = settings.pcURL(port: settings.data.pcChatPort) else {
            pcStatus = "Set the address first"
            return
        }
        pcStatus = "Checking…"
        var request = URLRequest(url: url.appendingPathComponent("models"))
        request.timeoutInterval = 6
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            let status = (response as? HTTPURLResponse)?.statusCode ?? 0
            if status == 200 {
                let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
                let count = (object?["data"] as? [Any])?.count ?? 0
                pcStatus = "Connected · \(count) model" + (count == 1 ? "" : "s")
            } else {
                pcStatus = "Answered with \(status)"
            }
        } catch {
            pcStatus = "Not reachable"
        }
    }
}

struct PortField: View {
    @Binding var value: Int

    var body: some View {
        TextField("", value: $value, format: .number.grouping(.never))
            .keyboardType(.numberPad)
            .multilineTextAlignment(.trailing)
            .font(.system(size: 14)).foregroundStyle(Theme.secondary)
            .frame(width: 70)
    }
}

struct KeyPage: View {
    let name: KeyName
    @EnvironmentObject var settings: AppSettings
    @Environment(\.dismiss) private var dismiss
    @State private var draft = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            PageHeader(title: name.label, back: { dismiss() }) { EmptyView() }
            SecureField("", text: $draft, prompt: Text(settings.keyHint(name) ?? "Paste your key").foregroundStyle(Theme.muted))
                .font(.system(size: 16)).foregroundStyle(Theme.text)
                .textInputAutocapitalization(.never).autocorrectionDisabled()
                .padding(.horizontal, 14).frame(height: 48)
                .background(RoundedRectangle(cornerRadius: 12).fill(Theme.surface))
            Text(name.hint + ". It stays in this iPhone's Keychain and is only sent to that provider.")
                .font(.system(size: 14)).foregroundStyle(Theme.muted)
            HStack(spacing: 12) {
                Button {
                    if let pasted = UIPasteboard.general.string { draft = pasted }
                } label: {
                    Text("Paste").font(.system(size: 16, weight: .medium)).foregroundStyle(Theme.text)
                        .frame(maxWidth: .infinity).frame(height: 52)
                        .overlay(Capsule().stroke(Theme.border, lineWidth: 0.5))
                }
                .buttonStyle(.plain)
                PrimaryPill(title: "Save", systemImage: "checkmark") {
                    settings.setKey(name, draft)
                    dismiss()
                }
            }
            if settings.key(name) != nil {
                Button("Remove this key") {
                    settings.setKey(name, "")
                    dismiss()
                }
                .font(.system(size: 15)).foregroundStyle(Theme.danger).buttonStyle(.plain)
            }
            Spacer()
        }
        .padding(20)
        .background(Theme.ground.ignoresSafeArea())
        .toolbar(.hidden, for: .navigationBar)
    }
}

struct PresetPage: View {
    let id: UUID
    @EnvironmentObject var settings: AppSettings
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        let index = settings.data.presets.firstIndex(where: { $0.id == id })
        VStack(alignment: .leading, spacing: 16) {
            PageHeader(title: "Preset", back: { dismiss() }) { EmptyView() }
            if let index {
                TextField("Name", text: $settings.data.presets[index].name)
                    .font(.system(size: 16)).foregroundStyle(Theme.text)
                    .padding(.horizontal, 14).frame(height: 48)
                    .background(RoundedRectangle(cornerRadius: 12).fill(Theme.surface))
                Text(settings.data.presets[index].summary + " · " + settings.data.presets[index].effort.label + " effort")
                    .font(.system(size: 14)).foregroundStyle(Theme.muted)
                Text("To change its brain or voice, make it current and open Models from the top of the talk screen.")
                    .font(.system(size: 14)).foregroundStyle(Theme.muted)
                PrimaryPill(title: "Make current", systemImage: "checkmark") {
                    settings.data.currentPresetID = id
                    dismiss()
                }
                if settings.data.presets.count > 1 {
                    Button("Delete this preset") {
                        settings.data.presets.remove(at: index)
                        if settings.data.currentPresetID == id { settings.data.currentPresetID = settings.data.presets.first?.id }
                        dismiss()
                    }
                    .font(.system(size: 15)).foregroundStyle(Theme.danger).buttonStyle(.plain)
                }
            }
            Spacer()
        }
        .padding(20)
        .background(Theme.ground.ignoresSafeArea())
        .toolbar(.hidden, for: .navigationBar)
    }
}

enum LaunchLink {
    /// How LiveContainer opens this app from a Shortcut.
    static let url = "livecontainer://livecontainer-launch?bundle-name=com.mahmudkarim.Earshot.app"
}

struct ActionButtonPage: View {
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                PageHeader(title: "Action Button", back: { dismiss() }) { EmptyView() }
                Text("Make a Shortcut that takes a screenshot, copies it and opens Earshot. Earshot picks up the new picture from the clipboard and starts listening.")
                    .font(.system(size: 15)).foregroundStyle(Theme.secondary)
                Steps(lines: [
                    "Open Shortcuts and tap + for a new shortcut. Name it Ask Earshot.",
                    "Add the action Take Screenshot.",
                    "Add Copy to Clipboard.",
                    "Add Open URLs, and paste the link below.",
                    "In iOS Settings › Action Button, choose Shortcut › Ask Earshot.",
                    "In iOS Settings › Apps › LiveContainer, set Paste from Other Apps to Allow, so iOS stops asking each time."
                ])
                LinkCard(link: LaunchLink.url)
                Text("For a plain Talk button without a screenshot, make a second shortcut with only the Open URLs step.")
                    .font(.system(size: 14)).foregroundStyle(Theme.muted)
            }
            .padding(20)
        }
        .background(Theme.ground.ignoresSafeArea())
        .toolbar(.hidden, for: .navigationBar)
    }
}

struct SafariPage: View {
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                PageHeader(title: "Safari", back: { dismiss() }) { EmptyView() }
                Text("Asking about the text of a web page, from Safari's share sheet, is coming in the next version. For now, the Action Button screenshot works on any page.")
                    .font(.system(size: 15)).foregroundStyle(Theme.secondary)
            }
            .padding(20)
        }
        .background(Theme.ground.ignoresSafeArea())
        .toolbar(.hidden, for: .navigationBar)
    }
}

struct Steps: View {
    let lines: [String]

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            ForEach(Array(lines.enumerated()), id: \.offset) { index, line in
                HStack(alignment: .top, spacing: 12) {
                    Text("\(index + 1)").font(.system(size: 13, weight: .semibold)).foregroundStyle(Theme.onClay)
                        .frame(width: 24, height: 24).background(Circle().fill(Theme.clay))
                    Text(line).font(.system(size: 15)).foregroundStyle(Theme.text)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }
}

struct LinkCard: View {
    let link: String
    @State private var copied = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(link).font(.system(size: 13, design: .monospaced)).foregroundStyle(Theme.textBody)
                .textSelection(.enabled)
            Button {
                UIPasteboard.general.string = link
                copied = true
            } label: {
                Label(copied ? "Copied" : "Copy link", systemImage: copied ? "checkmark" : "doc.on.doc")
                    .font(.system(size: 13)).foregroundStyle(Theme.textBody)
                    .padding(.horizontal, 12).frame(height: 32)
                    .background(Capsule().fill(Theme.raised))
            }
            .buttonStyle(.plain)
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 16).fill(Theme.surface))
    }
}
