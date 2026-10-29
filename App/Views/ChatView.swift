import SwiftUI
import UIKit

/// A conversation: the turns, the reply as it arrives, and the controls.
struct ChatView: View {
    @EnvironmentObject var conversation: Conversation
    @EnvironmentObject var store: ChatStore
    let openChats: () -> Void
    let openModels: () -> Void
    @Binding var typing: Bool

    var body: some View {
        VStack(spacing: 0) {
            header
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 22) {
                        ForEach(conversation.chat?.turns ?? []) { turn in
                            if turn.isUser {
                                UserBubble(turn: turn)
                            } else {
                                // An empty saved answer (a stopped reply, before this was handled) would show the waiting dots forever.
                                ReplyBlock(text: turn.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "No answer was saved." : turn.text,
                                           model: turn.modelLabel ?? "", failed: turn.failed,
                                           meta: meta(turn), retry: retryAction(turn),
                                           replay: replayControl(turn), sources: turn.sources ?? [])
                            }
                        }
                        if conversation.phase == .thinking || !conversation.reply.isEmpty
                            || (conversation.phase == .speaking && conversation.replaying == nil) {
                            ReplyBlock(text: conversation.reply, model: conversation.replyModel, failed: false, meta: nil, retry: nil, replay: nil,
                                       sources: conversation.replySources)
                        }
                        Color.clear.frame(height: 1).id("end")
                    }
                    .padding(.horizontal, 20)
                    .padding(.top, 16)
                }
                .scrollDismissesKeyboard(.interactively)
                .onChange(of: conversation.reply) { _, _ in proxy.scrollTo("end", anchor: .bottom) }
                .onChange(of: conversation.chat?.turns.count) { _, _ in
                    withAnimation { proxy.scrollTo("end", anchor: .bottom) }
                }
                .onAppear { proxy.scrollTo("end", anchor: .bottom) }
            }

            VStack(spacing: 14) {
                if let error = conversation.error {
                    ErrorBanner(text: error) { conversation.error = nil }
                }
                if let approval = conversation.approval {
                    ApprovalCard(approval: approval)
                }
                if conversation.chat?.pendingRun != nil && conversation.phase == .idle {
                    PendingRunCard()
                }
                if conversation.phase == .thinking || conversation.phase == .speaking {
                    StatusBar()
                }
                if let attachment = conversation.attachment {
                    AttachmentChip(attachment: attachment) { conversation.attachment = nil }
                }
                if typing {
                    TypeBar(isOpen: $typing)
                } else {
                    HStack(spacing: 28) {
                        RoundButton(systemImage: "keyboard", label: "Type instead", size: 52) { typing = true }
                        RoundButton(systemImage: "mic", label: "Talk", size: 64, filled: true) { conversation.startListening() }
                        PhotoButton(size: 52)
                    }
                }
            }
            .padding(.horizontal, 20)
            .padding(.top, 12)
            .padding(.bottom, 12)
        }
    }

    private var header: some View {
        HStack(spacing: 4) {
            Button(action: openChats) {
                Image(systemName: "chevron.left").font(.system(size: 19, weight: .medium))
                    .foregroundStyle(Theme.secondary).frame(width: 44, height: 44)
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Back to chats")
            Text(conversation.chat?.title ?? "").font(.system(size: 15, weight: .medium)).foregroundStyle(Theme.text)
                .lineLimit(1)
            Spacer(minLength: 8)
            ModelPill(action: openModels)
                .scaleEffect(0.9)
            Button {
                conversation.newChat()
            } label: {
                Image(systemName: "square.and.pencil").font(.system(size: 18)).foregroundStyle(Theme.secondary)
                    .frame(width: 40, height: 44)
            }
            .buttonStyle(.plain)
            .accessibilityLabel("New chat")
        }
        .padding(.leading, 8).padding(.trailing, 8)
        .frame(height: 44)
        .padding(.top, 8)
    }

    /// Only the last answer, when it failed, can be tried again.
    private func retryAction(_ turn: Turn) -> (() -> Void)? {
        guard turn.failed, turn.id == conversation.chat?.turns.last?.id else { return nil }
        let conversation = self.conversation
        return { conversation.retry() }
    }

    /// Any finished answer can be heard again. While one plays, its button stops it.
    private func replayControl(_ turn: Turn) -> ReplayControl? {
        guard !turn.failed, !turn.text.isEmpty else { return nil }
        let conversation = self.conversation
        let playing = conversation.replaying == turn.id
        return ReplayControl(playing: playing, enabled: playing || conversation.phase == .idle) {
            if playing { conversation.stopSpeaking() } else { conversation.replay(turn) }
        }
    }

    private func meta(_ turn: Turn) -> String? {
        guard !turn.failed else { return nil }
        var parts: [String] = []
        if let seconds = turn.firstWordSeconds { parts.append(String(format: "First word in %.1f s", seconds)) }
        if let cost = turn.cost { parts.append(Money.text(cost)) }
        return parts.isEmpty ? nil : parts.joined(separator: " · ")
    }
}

struct UserBubble: View {
    @EnvironmentObject var store: ChatStore
    let turn: Turn

    var body: some View {
        VStack(alignment: .trailing, spacing: 6) {
            if let file = turn.imageFile {
                HStack(spacing: 8) {
                    if let image = store.image(file) {
                        Image(uiImage: image).resizable().scaledToFill()
                            .frame(width: 22, height: 30).clipShape(RoundedRectangle(cornerRadius: 4))
                    }
                    Text(turn.attachmentLabel ?? "Image").font(.system(size: 12)).foregroundStyle(Theme.muted)
                }
                .padding(.leading, 4).padding(.trailing, 10).frame(height: 36)
                .background(Capsule().fill(Theme.surface))
            }
            Text(turn.text)
                .font(.system(size: 16)).lineSpacing(3).foregroundStyle(Theme.text)
                .padding(.horizontal, 16).padding(.vertical, 12)
                .background(UnevenRoundedRectangle(topLeadingRadius: 20, bottomLeadingRadius: 20,
                                                   bottomTrailingRadius: 6, topTrailingRadius: 20).fill(Theme.raised))
                .textSelection(.enabled)
        }
        .frame(maxWidth: 300, alignment: .trailing)
        .frame(maxWidth: .infinity, alignment: .trailing)
    }
}

/// An answer: serif paragraphs, with drafts and code in copyable cards.
struct ReplyBlock: View {
    let text: String
    let model: String
    let failed: Bool
    let meta: String?
    let retry: (() -> Void)?
    let replay: ReplayControl?
    var sources: [Source] = []

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 8) {
                Circle().fill(failed ? Theme.danger : Theme.clay).frame(width: 10, height: 10)
                Text(model).font(.system(size: 13)).foregroundStyle(Theme.muted)
            }
            if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                ThinkingDots()
            }
            ForEach(Array(ReplyBlock.segments(text).enumerated()), id: \.offset) { _, segment in
                switch segment {
                case .prose(let paragraph):
                    Text(ReplyBlock.markdown(paragraph))
                        .font(Theme.serif(18)).lineSpacing(5)
                        .foregroundStyle(failed ? Theme.secondary : Theme.text)
                        .tint(Theme.clayLight)
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                case .draft(let draft):
                    DraftCard(text: draft)
                }
            }
            if !sources.isEmpty {
                SourceList(sources: sources)
            }
            if failed, let retry {
                Button("Try again", action: retry)
                    .font(.system(size: 14, weight: .medium)).foregroundStyle(Theme.clay)
                    .buttonStyle(.plain)
            }
            if meta != nil || replay != nil {
                HStack(spacing: 12) {
                    if let replay {
                        Button(action: replay.action) {
                            Label(replay.playing ? "Stop" : "Play again",
                                  systemImage: replay.playing ? "stop.fill" : "speaker.wave.2")
                                .font(.system(size: 13)).foregroundStyle(Theme.textBody)
                                .padding(.horizontal, 12).frame(height: 32)
                                .background(Capsule().fill(Theme.raised))
                        }
                        .buttonStyle(.plain)
                        .disabled(!replay.enabled)
                        .opacity(replay.enabled ? 1 : 0.4)
                    }
                    if let meta {
                        Text(meta).font(.system(size: 12)).foregroundStyle(Theme.muted)
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    enum Segment {
        case prose(String)
        case draft(String)
    }

    /// Prose paragraphs and fenced blocks, in order. An unclosed fence still shows as a draft.
    static func segments(_ text: String) -> [Segment] {
        var result: [Segment] = []
        let parts = text.components(separatedBy: "```")
        for (index, part) in parts.enumerated() {
            if index % 2 == 1 {
                // Drop the language tag on the first line.
                var lines = part.components(separatedBy: "\n")
                if let first = lines.first, !first.contains(" "), first.count < 20 { lines.removeFirst() }
                let draft = lines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
                if !draft.isEmpty { result.append(.draft(draft)) }
            } else {
                for paragraph in part.components(separatedBy: "\n\n") {
                    let trimmed = paragraph.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !trimmed.isEmpty { result.append(.prose(trimmed)) }
                }
            }
        }
        return result
    }

    static func markdown(_ text: String) -> AttributedString {
        let options = AttributedString.MarkdownParsingOptions(interpretedSyntax: .inlineOnlyPreservingWhitespace)
        return (try? AttributedString(markdown: text, options: options)) ?? AttributedString(text)
    }
}

/// The web pages an answer drew on, as links.
struct SourceList: View {
    let sources: [Source]

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Sources").font(.system(size: 13)).foregroundStyle(Theme.muted)
            ForEach(sources, id: \.url) { source in
                if let url = URL(string: source.url) {
                    Link(destination: url) {
                        HStack(spacing: 8) {
                            Image(systemName: "globe").font(.system(size: 12)).foregroundStyle(Theme.muted)
                            Text(source.title.isEmpty ? (url.host() ?? source.url) : source.title)
                                .font(.system(size: 14)).foregroundStyle(Theme.textBody).lineLimit(1)
                            Spacer(minLength: 4)
                            Text(url.host()?.replacingOccurrences(of: "www.", with: "") ?? "")
                                .font(.system(size: 12)).foregroundStyle(Theme.muted).lineLimit(1)
                        }
                        .padding(.horizontal, 12).frame(height: 36)
                        .background(RoundedRectangle(cornerRadius: 10).fill(Theme.surface))
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }
}

/// The "Play again" button under an answer.
struct ReplayControl {
    let playing: Bool
    let enabled: Bool
    let action: () -> Void
}

struct DraftCard: View {
    let text: String
    @State private var copied = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(text).font(.system(size: 15)).lineSpacing(3).foregroundStyle(Theme.textBody)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
            HStack(spacing: 8) {
                Button {
                    UIPasteboard.general.string = text
                    copied = true
                    Haptics.tap()
                } label: {
                    Label(copied ? "Copied" : "Copy", systemImage: copied ? "checkmark" : "doc.on.doc")
                        .font(.system(size: 13)).foregroundStyle(Theme.textBody)
                        .padding(.horizontal, 12).frame(height: 32)
                        .background(Capsule().fill(Theme.raised))
                }
                .buttonStyle(.plain)
                ShareLink(item: text) {
                    Text("Share").font(.system(size: 13)).foregroundStyle(Theme.textBody)
                        .padding(.horizontal, 12).frame(height: 32)
                        .background(Capsule().fill(Theme.raised))
                }
            }
        }
        .padding(.horizontal, 16).padding(.top, 14).padding(.bottom, 10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 16).fill(Theme.surface))
        .overlay(RoundedRectangle(cornerRadius: 16).stroke(Theme.hairline, lineWidth: 0.5))
    }
}

struct ThinkingDots: View {
    @State private var on = false
    var body: some View {
        HStack(spacing: 6) {
            ForEach(0..<3) { index in
                Circle().fill(Theme.muted).frame(width: 7, height: 7)
                    .opacity(on ? 1 : 0.3)
                    .animation(.easeInOut(duration: 0.6).repeatForever().delay(Double(index) * 0.2), value: on)
            }
        }
        .onAppear { on = true }
        .accessibilityLabel("Thinking")
    }
}

/// A Hermes run still working on the server while Earshot can't follow it, e.g. offline.
struct PendingRunCard: View {
    @EnvironmentObject var conversation: Conversation

    var body: some View {
        HStack(spacing: 12) {
            ProgressView().tint(Theme.muted)
            VStack(alignment: .leading, spacing: 2) {
                Text("Hermes is still working on this").font(.system(size: 15, weight: .medium)).foregroundStyle(Theme.text)
                Text("The answer shows up here once Earshot can reach the server.")
                    .font(.system(size: 13)).foregroundStyle(Theme.muted).fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
            Button("Check") { conversation.resumePendingRun() }
                .font(.system(size: 14, weight: .semibold)).foregroundStyle(Theme.clay)
                .buttonStyle(.plain)
        }
        .padding(14)
        .background(RoundedRectangle(cornerRadius: 18).fill(Theme.surface))
        .overlay(RoundedRectangle(cornerRadius: 18).stroke(Theme.border, lineWidth: 0.5))
    }
}

/// Hermes asking before a risky step. Saying "yes" or "no" answers it too.
struct ApprovalCard: View {
    @EnvironmentObject var conversation: Conversation
    let approval: HermesApproval

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Hermes wants your OK").font(.system(size: 15, weight: .semibold)).foregroundStyle(Theme.text)
            if !approval.description.isEmpty {
                Text(approval.description).font(.system(size: 14)).foregroundStyle(Theme.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if !approval.command.isEmpty {
                Text(approval.command).font(.system(size: 13, design: .monospaced)).foregroundStyle(Theme.textBody)
                    .lineLimit(4).textSelection(.enabled)
                    .padding(10).frame(maxWidth: .infinity, alignment: .leading)
                    .background(RoundedRectangle(cornerRadius: 10).fill(Theme.raised))
            }
            HStack(spacing: 12) {
                Button { conversation.answerApproval("deny") } label: {
                    Text("Don't allow").font(.system(size: 15, weight: .medium)).foregroundStyle(Theme.text)
                        .frame(maxWidth: .infinity).frame(height: 44)
                        .overlay(Capsule().stroke(Theme.border, lineWidth: 0.5))
                }
                .buttonStyle(.plain)
                Button { conversation.answerApproval("once") } label: {
                    Text("Allow").font(.system(size: 15, weight: .semibold)).foregroundStyle(Theme.onClay)
                        .frame(maxWidth: .infinity).frame(height: 44)
                        .background(Capsule().fill(Theme.clay))
                }
                .buttonStyle(.plain)
            }
            Text("Or tap the mic and say yes or no. It's turned down if not answered within a minute.")
                .font(.system(size: 12)).foregroundStyle(Theme.muted)
        }
        .padding(16)
        .background(RoundedRectangle(cornerRadius: 22).fill(Theme.surface))
        .overlay(RoundedRectangle(cornerRadius: 22).stroke(Theme.clay.opacity(0.5), lineWidth: 1))
    }
}

/// "Speaking · tap to interrupt" with the stop button.
struct StatusBar: View {
    @EnvironmentObject var conversation: Conversation
    @State private var pulse: [Float] = Array(repeating: 0.3, count: 14)
    private let timer = Timer.publish(every: 0.12, on: .main, in: .common).autoconnect()

    var body: some View {
        HStack(spacing: 12) {
            ZStack {
                Circle().stroke(Theme.clay.opacity(0.5), lineWidth: 1).frame(width: 28, height: 28)
                Circle().fill(Theme.clay).frame(width: 14, height: 14)
            }
            if conversation.phase == .speaking {
                // Moving only while sound is coming out; flat in the pauses while the next sentence is made.
                LevelBars(levels: conversation.sounding ? pulse : Array(repeating: 0.15, count: pulse.count),
                          barWidth: 3, maxHeight: 24, spacing: 3)
            }
            Text(statusText)
                .font(.system(size: 14)).foregroundStyle(Theme.secondary).lineLimit(1)
            Spacer(minLength: 0)
            Button(action: conversation.stopSpeaking) {
                Image(systemName: "stop.fill").font(.system(size: 15)).foregroundStyle(Theme.text)
                    .frame(width: 44, height: 44).background(Circle().fill(Theme.high))
            }
            .buttonStyle(.plain)
            .accessibilityLabel(conversation.phase == .speaking ? "Stop speaking" : "Stop")
        }
        .padding(.leading, 16).padding(.trailing, 8)
        .frame(height: 60)
        .background(RoundedRectangle(cornerRadius: 22).fill(Theme.surface))
        .overlay(RoundedRectangle(cornerRadius: 22).stroke(Theme.border, lineWidth: 0.5))
        .contentShape(Rectangle())
        .onTapGesture {
            if conversation.phase == .speaking { conversation.startListening(interrupting: true) }
        }
        .onReceive(timer) { _ in
            if conversation.sounding {
                pulse = pulse.map { _ in Float.random(in: 0.25...1) }
            }
        }
    }

    private var statusText: String {
        if conversation.searching { return "Searching the web…" }
        if conversation.approval != nil { return "Waiting for your OK" }
        if conversation.sounding { return "Speaking · tap to interrupt" }
        // Hermes keeps working after its short "On it.": say what it's doing while nothing plays.
        if let step = conversation.working { return "Working · \(step)" }
        if conversation.phase == .speaking { return "Speaking · tap to interrupt" }
        if let voice = conversation.loadingVoice { return "Loading \(voice) voice…" }
        return conversation.replaying != nil ? "Getting the voice ready…" : "Thinking…"
    }
}
