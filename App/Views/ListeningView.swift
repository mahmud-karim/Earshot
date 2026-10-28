import SwiftUI

/// Full screen while the user talks: their words as they come, and the send button.
struct ListeningView: View {
    @EnvironmentObject var conversation: Conversation
    let typeInstead: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                RoundIcon(systemImage: "xmark", label: "Cancel") { conversation.cancelListening() }
                Spacer()
                HStack(spacing: 8) {
                    Circle().fill(Theme.clay).frame(width: 8, height: 8)
                    Text(pillText).font(.system(size: 14, weight: .medium)).foregroundStyle(Theme.clayLight)
                }
                .padding(.horizontal, 14).frame(height: 32)
                .background(Capsule().fill(Theme.clay.opacity(0.14)))
                Spacer()
                Color.clear.frame(width: 44, height: 44)
            }
            .frame(height: 44)

            if let attachment = conversation.attachment {
                AttachmentChip(attachment: attachment) { conversation.attachment = nil }.padding(.top, 20)
            }

            Spacer()
            VStack(spacing: 36) {
                words
                LevelBars(levels: conversation.levels)
            }
            Spacer()

            VStack(spacing: 18) {
                Text(conversation.paused ? "Paused · tap play to keep going" : "Sends when you pause · tap to send now")
                    .font(.system(size: 13)).foregroundStyle(Theme.muted)
                HStack(spacing: 28) {
                    RoundButton(systemImage: "keyboard", label: "Type instead") {
                        conversation.cancelListening()
                        typeInstead()
                    }
                    Button(action: conversation.sendNow) {
                        Image(systemName: "arrow.up").font(.system(size: 28, weight: .semibold))
                            .foregroundStyle(Theme.onClay).frame(width: 80, height: 80)
                            .background(Circle().fill(Theme.clay))
                            .overlay(Circle().stroke(Theme.clay.opacity(0.16), lineWidth: 8).padding(-4))
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("Send now")
                    .disabled(conversation.phase != .listening)
                    RoundButton(systemImage: conversation.paused ? "play.fill" : "pause.fill",
                                label: conversation.paused ? "Keep listening" : "Pause listening") {
                        conversation.togglePause()
                    }
                }
            }
        }
        .padding(.horizontal, 20)
        .padding(.top, 8)
        .padding(.bottom, 12)
        .background(Theme.ground.ignoresSafeArea())
    }

    private var pillText: String {
        if conversation.phase == .transcribing { return "Sending" }
        if conversation.status != nil { return "Getting ready" }
        return conversation.paused ? "Paused" : "Listening"
    }

    @ViewBuilder private var words: some View {
        // The newest words matter; a long question shows its end.
        let whole = conversation.liveText
        let text = whole.count > 180 ? "…" + whole.suffix(170) : whole
        Group {
            if let status = conversation.status {
                Text(status).foregroundStyle(Theme.muted)
            } else if text.isEmpty {
                Text(conversation.transcriberGivesLiveText ? "Go ahead, I’m listening" : "Listening… your words appear when you pause")
                    .foregroundStyle(Theme.muted)
            } else {
                Text("\(Text(text).foregroundStyle(Theme.text))\(Text(" ▍").foregroundStyle(Theme.clay))")
            }
        }
        .font(.system(size: 28))
        .lineSpacing(4)
        .frame(maxWidth: .infinity, alignment: .leading)
        .animation(.easeOut(duration: 0.15), value: text)
    }
}
