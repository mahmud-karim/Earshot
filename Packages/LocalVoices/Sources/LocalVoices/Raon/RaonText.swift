import Foundation

/// Raon's text side: the character vocabulary, the character list the official code builds, and the
/// sentence chunking it uses for long input. English only: Chinese pinyin conversion is not ported.
struct RaonTokenizer {
    /// Unicode scalar to vocabulary index. Index 0 is the space, which also stands in for unknown characters.
    private let vocabulary: [UInt32: Int32]

    /// `usableCount` is the checkpoint's embedding rows minus the filler row. The shipped vocab.txt lists
    /// four characters more than the checkpoints embed (U+FDFA, U+FDFB, U+FFFD, U+1F3B5); those are
    /// dropped and read as unknown, as the ComfyUI pack does.
    init(vocabURL: URL, usableCount: Int) throws {
        let text = try String(contentsOf: vocabURL, encoding: .utf8)
        var vocabulary: [UInt32: Int32] = [:]
        for (index, line) in text.components(separatedBy: "\n").enumerated() where index < usableCount {
            let scalars = Array(line.unicodeScalars)
            if scalars.count == 1 { vocabulary[scalars[0].value] = Int32(index) }
        }
        guard vocabulary[32] == 0 else { throw LocalVoiceError("Raon's vocab.txt must start with a space.") }
        self.vocabulary = vocabulary
    }

    /// Token ids for an already prepared string, one per Unicode scalar as Python indexes strings.
    func encode(_ text: String) -> [Int32] {
        Self.characters(text).map { vocabulary[$0.value] ?? 0 }
    }

    /// The English half of f5_tts `convert_char_to_pinyin`: jieba splits the text into runs of letters
    /// and digits (keeping decimals and a trailing %), runs of `+#&._%-`, and single other characters;
    /// a space is then inserted before any multi-character ASCII piece that does not follow a space,
    /// colon or quote. That is why "state-of-the-art" reads "state- of- the- art", as in training.
    static func characters(_ text: String) -> [Unicode.Scalar] {
        let replacements: [Unicode.Scalar: Unicode.Scalar] = [";": ",", "\u{201C}": "\"", "\u{201D}": "\"", "\u{2018}": "'", "\u{2019}": "'"]
        let input = text.unicodeScalars.map { replacements[$0] ?? $0 }
        func isDigit(_ c: Unicode.Scalar) -> Bool { (48 ... 57).contains(c.value) }
        func isAlnum(_ c: Unicode.Scalar) -> Bool { isDigit(c) || (65 ... 90).contains(c.value) || (97 ... 122).contains(c.value) }
        let joiners = Set("+#&._%-".unicodeScalars)

        var output: [Unicode.Scalar] = []
        var index = 0
        while index < input.count {
            var end = index + 1
            let c = input[index]
            if c == "\r", end < input.count, input[end] == "\n" {
                end += 1
            } else if isAlnum(c) {
                while end < input.count, isAlnum(input[end]) { end += 1 }
                if end + 1 < input.count, input[end] == ".", isDigit(input[end + 1]) {
                    end += 2
                    while end < input.count, isDigit(input[end]) { end += 1 }
                }
                if end < input.count, input[end] == "%" { end += 1 }
            } else if joiners.contains(c) {
                while end < input.count, joiners.contains(input[end]) { end += 1 }
            }
            let piece = input[index ..< end]
            if piece.count > 1, piece.allSatisfy(\.isASCII), let last = output.last, !" :'\"".unicodeScalars.contains(last) {
                output.append(" ")
            }
            output.append(contentsOf: piece)
            index = end
        }
        return output
    }
}

enum RaonText {
    /// f5_tts `fix_ref_text_ending`: the reference transcript ends in ". " so the new words start a sentence.
    static func referenceEnding(_ text: String) -> String {
        var text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return text }
        if !text.hasSuffix(". "), !text.hasSuffix("。") {
            text += text.hasSuffix(".") ? " " : ". "
        }
        return text
    }

    /// f5_tts `chunk_text`: splits after sentence punctuation and packs pieces up to `maxBytes` of UTF-8.
    static func chunks(_ text: String, maxBytes: Int) -> [String] {
        let pattern = try! NSRegularExpression(pattern: "(?<=[;:,.!?])\\s+|(?<=[；：，。！？])")
        let ns = text as NSString
        var sentences: [String] = []
        var start = 0
        for match in pattern.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
            sentences.append(ns.substring(with: NSRange(location: start, length: match.range.location - start)))
            start = match.range.location + match.range.length
        }
        sentences.append(ns.substring(from: start))

        var chunks: [String] = []
        var current = ""
        for sentence in sentences {
            let piece = sentence.unicodeScalars.last?.isASCII == true ? sentence + " " : sentence
            if current.utf8.count + sentence.utf8.count <= maxBytes {
                current += piece
            } else {
                if !current.isEmpty { chunks.append(current.trimmingCharacters(in: .whitespaces)) }
                current = piece
            }
        }
        if !current.isEmpty { chunks.append(current.trimmingCharacters(in: .whitespaces)) }
        return chunks.filter { !$0.isEmpty }
    }
}
