import Foundation

/// One pass of Read aloud, for its screen: the text cut into sentences and words, what the voice is
/// doing now, and which sentence is playing.
///
/// None of the voices say when each word is spoken, so a word's moment is worked out from its
/// sentence: the sentence's start and length come from the audio actually playing, and its words
/// share that time by their length.
@MainActor
final class ReadAloudSession: ObservableObject {
    /// The text as it is read and shown.
    let text: String
    /// Each sentence's place in `text`.
    let sentences: [Range<String.Index>]
    /// Each sentence's words, in order.
    let words: [[Range<String.Index>]]
    /// Where each word ends, as a fraction of its sentence's audio.
    let wordEnds: [[Double]]
    /// The sentences that have something to say; the others are never played.
    let speakable: [Bool]
    /// The voice, as the status line names it: "OpenAI", "Kokoro on your PC", ….
    let voiceName: String
    /// Each line of `text`, shown as its own paragraph.
    let paragraphs: [Range<String.Index>]

    /// The sentence whose audio is being made, if the queue is waiting on one.
    @Published var making: Int?
    /// The sentence playing now, with when it started and how long its audio lasts.
    @Published private(set) var playing: Int?
    @Published private(set) var playingStart = Date()
    @Published private(set) var playingDuration: TimeInterval = 0
    /// Sentences heard to the end.
    @Published private(set) var heard = 0
    /// The end of the last sentence heard: everything before it has been read.
    @Published private(set) var readUpTo: String.Index

    var total: Int { max(speakable.filter { $0 }.count, 1) }

    init(text: String, sentences: [String], voiceName: String) {
        self.text = text
        self.voiceName = voiceName
        readUpTo = text.startIndex
        var lines: [Range<String.Index>] = []
        var lineStart = text.startIndex
        for index in text.indices where text[index].isNewline {
            lines.append(lineStart..<index)
            lineStart = text.index(after: index)
        }
        lines.append(lineStart..<text.endIndex)
        paragraphs = lines.filter { !text[$0].trimmingCharacters(in: .whitespaces).isEmpty }
        var ranges: [Range<String.Index>] = []
        var cursor = text.startIndex
        for sentence in sentences {
            if let found = text.range(of: sentence, range: cursor..<text.endIndex) {
                ranges.append(found)
                cursor = found.upperBound
            } else {
                ranges.append(cursor..<cursor)
            }
        }
        self.sentences = ranges
        speakable = sentences.map { SpeechText.clean($0).rangeOfCharacter(from: .alphanumerics) != nil }
        var allWords: [[Range<String.Index>]] = []
        var allEnds: [[Double]] = []
        for range in ranges {
            let found = ReadAloudSession.words(in: text, range: range)
            // Longer words take longer to say; every word also gets a little time, and a word that
            // ends a clause is followed by a pause.
            let weights = found.map { word -> Double in
                let piece = text[word]
                let letters = piece.unicodeScalars.filter { CharacterSet.alphanumerics.contains($0) }.count
                let pause = piece.last.map { ",;:.!?".contains($0) } == true ? 2.0 : 0
                return Double(letters) + 1.5 + pause
            }
            let sum = max(weights.reduce(0, +), 1)
            var running = 0.0
            allEnds.append(weights.map { running += $0; return running / sum })
            allWords.append(found)
        }
        words = allWords
        wordEnds = allEnds
    }

    /// The sentence's position among those that are read, from 1.
    func number(_ sentence: Int) -> Int {
        speakable.prefix(sentence + 1).filter { $0 }.count
    }

    func started(_ sentence: Int, duration: TimeInterval) {
        playing = sentence
        playingStart = Date()
        playingDuration = duration
    }

    func ended(_ sentence: Int) {
        heard = min(heard + 1, total)
        if sentence < sentences.count, sentences[sentence].upperBound > readUpTo { readUpTo = sentences[sentence].upperBound }
        if playing == sentence { playing = nil }
    }

    /// The word being said at `date`, as a sentence and a word index.
    func word(at date: Date) -> (sentence: Int, word: Int)? {
        guard let sentence = playing, sentence < wordEnds.count, !wordEnds[sentence].isEmpty else { return nil }
        let fraction = playingDuration > 0 ? date.timeIntervalSince(playingStart) / playingDuration : 0
        let ends = wordEnds[sentence]
        let index = ends.firstIndex(where: { $0 > fraction }) ?? ends.count - 1
        return (sentence, index)
    }

    /// Where reading is: the start of the sentence playing, else the end of the last one heard.
    var position: String.Index {
        playing.map { sentences[$0].lowerBound } ?? readUpTo
    }

    /// The paragraph holding the sentence playing.
    var playingParagraph: Int? {
        guard let playing else { return nil }
        let start = sentences[playing].lowerBound
        return paragraphs.firstIndex(where: { $0.contains(start) || $0.lowerBound >= start })
    }

    /// How far through the whole text, from 0 to 1.
    func progress(at date: Date) -> Double {
        var done = Double(heard)
        if playing != nil, playingDuration > 0 {
            done += min(max(date.timeIntervalSince(playingStart) / playingDuration, 0), 1)
        }
        return min(done / Double(total), 1)
    }

    private static func words(in text: String, range: Range<String.Index>) -> [Range<String.Index>] {
        var result: [Range<String.Index>] = []
        var start: String.Index?
        var index = range.lowerBound
        while index < range.upperBound {
            if text[index].isWhitespace {
                if let begun = start { result.append(begun..<index); start = nil }
            } else if start == nil {
                start = index
            }
            index = text.index(after: index)
        }
        if let begun = start { result.append(begun..<range.upperBound) }
        return result
    }
}
