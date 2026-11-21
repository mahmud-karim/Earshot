import SwiftUI
import UIKit

/// Paste or type any text and hear it in the preset's voice, with the word being said lit up.
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

            if let session = conversation.readAloudSession {
                ReadAloudText(session: session)
                    .frame(maxHeight: .infinity)
                    .background(RoundedRectangle(cornerRadius: 18).fill(Theme.surface))
                    .overlay(RoundedRectangle(cornerRadius: 18).stroke(Theme.border, lineWidth: 0.5))
                ReadAloudStatus(session: session, loadingVoice: conversation.loadingVoice)
            } else {
                editor
            }

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

            if !playing {
                Text("Voice: \(settings.preset.voice.label)")
                    .font(.system(size: 13)).foregroundStyle(Theme.muted).lineLimit(1).minimumScaleFactor(0.8)
            }
        }
        .padding(.horizontal, 20)
        .padding(.top, 20)
        .padding(.bottom, 12)
        .background(Theme.ground.ignoresSafeArea())
        .animation(.easeOut(duration: 0.2), value: playing)
        .presentationDragIndicator(.visible)
        .presentationBackground(Theme.ground)
        // Nothing on the home screen stops it, so closing the sheet does.
        .onDisappear { if playing { conversation.stopSpeaking() } }
    }

    private var editor: some View {
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

/// What the voice is doing right now, and how far through the text it is.
struct ReadAloudStatus: View {
    @ObservedObject var session: ReadAloudSession
    let loadingVoice: String?

    var body: some View {
        TimelineView(.periodic(from: .now, by: 0.1)) { context in
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 10) {
                    if session.playing == nil {
                        ProgressView().controlSize(.small).tint(Theme.clay)
                    } else {
                        Image(systemName: "waveform")
                            .font(.system(size: 15, weight: .medium)).foregroundStyle(Theme.clay)
                            .symbolEffect(.variableColor.iterative)
                    }
                    Text(message)
                        .font(.system(size: 14)).foregroundStyle(Theme.secondary)
                        .lineLimit(2).fixedSize(horizontal: false, vertical: true)
                    Spacer(minLength: 0)
                }
                ProgressView(value: session.progress(at: context.date))
                    .tint(Theme.clay)
            }
            .padding(14)
            .background(RoundedRectangle(cornerRadius: 16).fill(Theme.surface))
            .overlay(RoundedRectangle(cornerRadius: 16).stroke(Theme.hairline, lineWidth: 0.5))
        }
        .accessibilityElement(children: .combine)
    }

    private var message: String {
        let total = session.total
        if let loadingVoice {
            return "Loading the \(loadingVoice) voice model. The first time can take a little while."
        }
        if let playing = session.playing {
            return "Reading sentence \(session.number(playing)) of \(total)"
        }
        if let making = session.making {
            return "Making the audio for sentence \(session.number(making)) of \(total) with \(session.voiceName)…"
        }
        if session.heard >= total {
            return "Finishing…"
        }
        return "Getting \(session.voiceName) ready…"
    }
}

/// The text being read, with the sentence playing brighter and the word being said lit up.
struct ReadAloudText: View {
    @ObservedObject var session: ReadAloudSession

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                TimelineView(.periodic(from: .now, by: 0.05)) { context in
                    paragraphs(at: context.date)
                }
            }
            .onChange(of: session.playingParagraph) { _, paragraph in
                guard let paragraph else { return }
                withAnimation(.easeOut(duration: 0.3)) { proxy.scrollTo(paragraph, anchor: .center) }
            }
        }
    }

    private func paragraphs(at date: Date) -> some View {
        let sentence = session.playing.map { session.sentences[$0] }
        var word: Range<String.Index>?
        if let current = session.word(at: date), current.word < session.words[current.sentence].count {
            word = session.words[current.sentence][current.word]
        }
        let position = session.position
        return VStack(alignment: .leading, spacing: 14) {
            ForEach(session.paragraphs.indices, id: \.self) { index in
                let paragraph = session.paragraphs[index]
                ReadAloudParagraph(
                    text: session.text,
                    range: paragraph,
                    position: ReadAloudParagraph.clamp(position, to: paragraph),
                    sentence: sentence.flatMap { ReadAloudParagraph.clip($0, to: paragraph) },
                    word: word.flatMap { ReadAloudParagraph.clip($0, to: paragraph) }
                )
                .equatable()
                .id(index)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(16)
    }
}

/// One line of the text. Only the line being read changes as the words go by, so the others are
/// left alone between sentences.
struct ReadAloudParagraph: View, Equatable {
    let text: String
    let range: Range<String.Index>
    /// Everything before this has been read.
    let position: String.Index
    let sentence: Range<String.Index>?
    let word: Range<String.Index>?

    static func == (lhs: ReadAloudParagraph, rhs: ReadAloudParagraph) -> Bool {
        lhs.range == rhs.range && lhs.position == rhs.position && lhs.sentence == rhs.sentence && lhs.word == rhs.word
    }

    var body: some View {
        Text(attributed)
            .font(.system(size: 19))
            .lineSpacing(6)
            .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var attributed: AttributedString {
        // Cut where the read part, the sentence and the word begin and end; colour each piece.
        var cuts = [range.lowerBound, position, range.upperBound]
        if let sentence { cuts += [sentence.lowerBound, sentence.upperBound] }
        if let word { cuts += [word.lowerBound, word.upperBound] }
        let points = Array(Set(cuts)).sorted()
        var result = AttributedString()
        for (start, end) in zip(points, points.dropFirst()) where start < end {
            var piece = AttributedString(String(text[start..<end]))
            if let word, word.lowerBound <= start, end <= word.upperBound {
                piece.foregroundColor = Theme.onClay
                piece.backgroundColor = Theme.clay
            } else if let sentence, sentence.lowerBound <= start, end <= sentence.upperBound {
                piece.foregroundColor = Theme.text
            } else if end <= position {
                piece.foregroundColor = Theme.muted
            } else {
                piece.foregroundColor = Theme.secondary
            }
            result += piece
        }
        return result
    }

    static func clamp(_ index: String.Index, to range: Range<String.Index>) -> String.Index {
        min(max(index, range.lowerBound), range.upperBound)
    }

    static func clip(_ part: Range<String.Index>, to range: Range<String.Index>) -> Range<String.Index>? {
        let lower = max(part.lowerBound, range.lowerBound)
        let upper = min(part.upperBound, range.upperBound)
        return lower < upper ? lower..<upper : nil
    }
}
