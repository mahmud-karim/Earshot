import SwiftUI

/// The Agent tab's home: ask Hermes to do something on the server, by voice or typing.
struct AgentHome: View {
    @EnvironmentObject var conversation: Conversation
    @EnvironmentObject var settings: AppSettings
    let openChats: () -> Void
    let openSettings: () -> Void

    @State private var typing = false

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                RoundIcon(systemImage: "clock", label: "Agent chats", action: openChats)
                Spacer()
                ModelPill(action: openSettings)
                Spacer()
                RoundIcon(systemImage: "gearshape", label: "Hermes settings", action: openSettings)
            }
            .frame(height: 44)

            if let error = conversation.error {
                ErrorBanner(text: error) { conversation.error = nil }.padding(.top, 12)
            }

            Spacer()
            VStack(spacing: 28) {
                Image(systemName: "terminal")
                    .font(.system(size: 44, weight: .light)).foregroundStyle(Theme.clay)
                    .frame(width: 120, height: 120)
                    .background(Circle().fill(Theme.clay.opacity(0.12)))
                    .overlay(Circle().stroke(Theme.clay.opacity(0.4), lineWidth: 1))
                VStack(spacing: 10) {
                    Text("What should Hermes do?")
                        .font(Theme.serif(32)).foregroundStyle(Theme.text).multilineTextAlignment(.center)
                    Text(settings.key(.hermes) == nil
                         ? "Add Hermes's key first: tap the gear."
                         : "It works on shopto-cloud with its own tools, tells you what it did, and asks before anything risky.")
                        .font(.system(size: 15)).foregroundStyle(Theme.muted)
                        .multilineTextAlignment(.center).frame(maxWidth: 300)
                }
            }
            Spacer()

            VStack(spacing: 16) {
                if let attachment = conversation.attachment {
                    AttachmentChip(attachment: attachment) { conversation.attachment = nil }
                }
                if typing {
                    TypeBar(isOpen: $typing)
                } else {
                    HStack(spacing: 12) {
                        RoundButton(systemImage: "keyboard", label: "Type instead") { typing = true }
                        PrimaryPill(title: "Tap to talk", systemImage: "mic") { conversation.startListening() }
                        PhotoButton()
                    }
                }
                Text("Ears: \(EarsText.name(settings.data.ears)) · Voice: \(settings.preset.voice.label)")
                    .font(.system(size: 13)).foregroundStyle(Theme.muted).lineLimit(1).minimumScaleFactor(0.8)
            }
        }
        .padding(.horizontal, 20)
        .padding(.top, 8)
        .padding(.bottom, 12)
    }
}

/// Hermes's address, key and connection test, as a sheet from the Agent tab.
struct AgentSettingsSheet: View {
    var body: some View {
        NavigationStack {
            HermesPage(showsBack: false)
                .navigationDestination(for: ModelsRoute.self) { route in
                    if route == .hermesKey { KeyPage(name: .hermes) }
                }
        }
        .presentationDragIndicator(.visible)
        .presentationBackground(Theme.ground)
    }
}
