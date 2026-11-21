import SwiftUI
import UIKit

/// Paste or type any text and hear it in the preset's voice.
struct ReadAloudSheet: View {
    @EnvironmentObject var conversation: Conversation
    @EnvironmentObject var settings: AppSettings
    @Environment(\.dismiss) private var dismiss
    @State private var text = ""
    @FocusState private var focused: Bool

    private var playing: Bool { conversation.readingAloud }
    private var hasWords: Bool { text.rangeOfCharacter(from: .alphanumerics) != nil }
    /// Something else, like a reply, is using the voice.
    private var busy: Bool { conversation.phase != .idle && !playing }

    var body: some View {
        VStack(spacing: 16) {
            HStack {
                Text("Read aloud").font(Theme.serif(28)).foregroundStyle(Theme.text)
                Spacer()
                Button("Done") { dismiss() }
                    .font(.system(size: 16, weight: .semibold)).foregroundStyle(Theme.clay)
                    .buttonStyle(.plain)
            }

            if let error = conversation.error {
                ErrorBanner(text: error) { conversation.error = nil }
            }

            ZStack(alignment: .topLeading) {
                TextEditor(text: $text)
                    .scrollContentBackground(.hidden)
                    .font(.system(size: 16))
                    .foregroundStyle(Theme.textBody)
                    .focused($focused)
                    .padding(10)
                if text.isEmpty {
                    Text("Paste or type something to hear it read out loud.")
                        .font(.system(size: 16)).foregroundStyle(Theme.muted)
                        .padding(.horizontal, 15).padding(.vertical, 18)
                        .allowsHitTesting(false)
                }
            }
            .frame(maxHeight: .infinity)
            .background(RoundedRectangle(cornerRadius: 18).fill(Theme.surface))
            .overlay(RoundedRectangle(cornerRadius: 18).stroke(Theme.border, lineWidth: 0.5))

            HStack(spacing: 12) {
                RoundButton(systemImage: "doc.on.clipboard", label: "Paste", action: paste)
                PrimaryPill(title: playing ? "Stop" : "Read aloud", systemImage: playing ? "stop.fill" : "play.fill", action: toggle)
                    .disabled(!playing && (!hasWords || busy))
                    .opacity(!playing && (!hasWords || busy) ? 0.5 : 1)
                RoundButton(systemImage: "xmark", label: "Clear") {
                    if playing { conversation.stopSpeaking() }
                    text = ""
                }
            }

            Text(conversation.loadingVoice.map { "Loading \($0)…" } ?? "Voice: \(settings.preset.voice.label)")
                .font(.system(size: 13)).foregroundStyle(Theme.muted).lineLimit(1).minimumScaleFactor(0.8)
        }
        .padding(.horizontal, 20)
        .padding(.top, 20)
        .padding(.bottom, 12)
        .background(Theme.ground.ignoresSafeArea())
        .presentationDragIndicator(.visible)
        .presentationBackground(Theme.ground)
        // Nothing on the home screen stops it, so closing the sheet does.
        .onDisappear { if playing { conversation.stopSpeaking() } }
    }

    private func paste() {
        guard let pasted = UIPasteboard.general.string, !pasted.isEmpty else { return }
        if playing { conversation.stopSpeaking() }
        text = pasted
        focused = false
    }

    private func toggle() {
        if playing {
            conversation.stopSpeaking()
        } else {
            focused = false
            conversation.error = nil
            conversation.readAloud(text)
        }
    }
}
