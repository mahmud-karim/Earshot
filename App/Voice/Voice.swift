import AVFoundation
import Foundation

// The part that speaks. A reply is spoken sentence by sentence while it is still being written: each
// sentence is synthesised as soon as it is complete, and played in order through AudioIO.

/// Turns one sentence into audio.
protocol Synthesizer {
    func audio(for text: String) async throws -> [AVAudioPCMBuffer]
}

/// Plays sentences in order, fetching each as soon as it is queued.
@MainActor
final class SpeechQueue {
    var onStart: (() -> Void)?
    var onDone: (() -> Void)?
    var onError: ((String) -> Void)?

    private let synthesizer: Synthesizer
    private var pending: [Task<[AVAudioPCMBuffer], Error>] = []
    private var inputFinished = false
    private var loop: Task<Void, Never>?
    private var started = false
    private var stopped = false

    init(synthesizer: Synthesizer) {
        self.synthesizer = synthesizer
    }

    func enqueue(_ sentence: String) {
        guard !stopped else { return }
        let text = SpeechText.clean(sentence)
        guard text.rangeOfCharacter(from: .alphanumerics) != nil else { return }
        let synthesizer = self.synthesizer
        pending.append(Task { try await synthesizer.audio(for: text) })
        if loop == nil {
            loop = Task { await self.run() }
        }
    }

    func finishInput() {
        inputFinished = true
        if loop == nil {
            finish()
        }
    }

    func stop() {
        stopped = true
        pending.forEach { $0.cancel() }
        pending.removeAll()
        loop?.cancel()
        AudioIO.shared.stopPlayback()
        drained?.resume()
        drained = nil
    }

    /// Buffers scheduled and not yet heard.
    private var outstanding = 0
    private var drained: CheckedContinuation<Void, Never>?

    private func run() async {
        while !stopped {
            if pending.isEmpty {
                if inputFinished { break }
                try? await Task.sleep(nanoseconds: 30_000_000)
                continue
            }
            let next = pending.removeFirst()
            do {
                let buffers = try await next.value
                guard !stopped else { break }
                if !started {
                    started = true
                    onStart?()
                }
                // Scheduled at once rather than one by one, so there is no gap between buffers.
                for buffer in buffers {
                    outstanding += 1
                    AudioIO.shared.play(buffer) {
                        DispatchQueue.main.async { self.bufferHeard() }
                    }
                }
            } catch {
                if !stopped, !(error is CancellationError) {
                    onError?(error.localizedDescription)
                }
            }
        }
        if !stopped && outstanding > 0 {
            await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                drained = continuation
            }
        }
        finish()
    }

    private func bufferHeard() {
        outstanding = max(0, outstanding - 1)
        if outstanding == 0, let drained {
            self.drained = nil
            drained.resume()
        }
    }

    private func finish() {
        guard !stopped else { return }
        stopped = true
        onDone?()
    }
}

/// What a sentence sounds like without its markdown.
enum SpeechText {
    static func clean(_ text: String) -> String {
        var result = text
        for symbol in ["**", "__", "`", "#", "> ", "* ", "- "] {
            result = result.replacingOccurrences(of: symbol, with: symbol.hasSuffix(" ") ? " " : "")
        }
        // [label](url) reads as its label.
        result = result.replacingOccurrences(of: #"\[([^\]]+)\]\([^)]+\)"#, with: "$1", options: .regularExpression)
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

/// Cuts streamed text into sentences. Fenced blocks are shown on screen but never spoken.
struct SentenceSplitter {
    private var buffer = ""
    private var inFence = false

    mutating func push(_ text: String) -> [String] {
        buffer += text
        var sentences: [String] = []
        while true {
            if inFence {
                guard let end = buffer.range(of: "```") else {
                    // Keep only enough to spot a closing fence split across deltas.
                    if buffer.count > 2 { buffer = String(buffer.suffix(2)) }
                    break
                }
                buffer = String(buffer[end.upperBound...])
                inFence = false
                continue
            }
            if let fence = buffer.range(of: "```") {
                let before = String(buffer[..<fence.lowerBound])
                sentences.append(contentsOf: SentenceSplitter.split(before, keepTail: false).sentences)
                buffer = String(buffer[fence.upperBound...])
                inFence = true
                continue
            }
            let (found, tail) = SentenceSplitter.split(buffer, keepTail: true)
            sentences.append(contentsOf: found)
            buffer = tail
            break
        }
        return sentences
    }

    /// Whatever is left when the reply ends.
    mutating func flush() -> [String] {
        defer { buffer = "" }
        return inFence ? [] : SentenceSplitter.split(buffer, keepTail: false).sentences
    }

    private static func split(_ text: String, keepTail: Bool) -> (sentences: [String], tail: String) {
        var sentences: [String] = []
        var current = ""
        let characters = Array(text)
        var index = 0
        while index < characters.count {
            let character = characters[index]
            current.append(character)
            let next: Character? = index + 1 < characters.count ? characters[index + 1] : nil
            let endsSentence = (".!?".contains(character) && (next == " " || next == "\n"))
                || character == "\n"
            if endsSentence && current.trimmingCharacters(in: .whitespacesAndNewlines).count > 1 {
                sentences.append(current)
                current = ""
            }
            index += 1
        }
        if keepTail {
            return (sentences, current)
        }
        if !current.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            sentences.append(current)
        }
        return (sentences, "")
    }
}

// MARK: - Apple voices

enum AppleVoices {
    /// English voices, best quality first.
    static func english() -> [AVSpeechSynthesisVoice] {
        AVSpeechSynthesisVoice.speechVoices()
            .filter { $0.language.hasPrefix("en") }
            .sorted { lhs, rhs in
                if lhs.quality.rawValue != rhs.quality.rawValue { return lhs.quality.rawValue > rhs.quality.rawValue }
                return lhs.name < rhs.name
            }
    }

    static func best(_ identifier: String) -> AVSpeechSynthesisVoice? {
        if !identifier.isEmpty, let chosen = AVSpeechSynthesisVoice(identifier: identifier) {
            return chosen
        }
        let voices = english()
        let local = voices.filter { $0.language == AVSpeechSynthesisVoice.currentLanguageCode() }
        return (local.first(where: { $0.name.hasPrefix("Ava") }) ?? local.first ?? voices.first)
    }

    static func name(for identifier: String) -> String? {
        best(identifier)?.name
    }

    static func quality(_ voice: AVSpeechSynthesisVoice) -> String {
        switch voice.quality {
        case .premium: return "Premium"
        case .enhanced: return "Enhanced"
        default: return "Compact"
        }
    }
}

final class AppleSynthesizer: Synthesizer {
    let voice: AVSpeechSynthesisVoice?

    init(identifier: String) {
        voice = AppleVoices.best(identifier)
    }

    func audio(for text: String) async throws -> [AVAudioPCMBuffer] {
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = voice
        let synthesizer = AVSpeechSynthesizer()
        return await withCheckedContinuation { (continuation: CheckedContinuation<[AVAudioPCMBuffer], Never>) in
            var buffers: [AVAudioPCMBuffer] = []
            var resumed = false
            synthesizer.write(utterance) { buffer in
                guard !resumed else { return }
                guard let pcm = buffer as? AVAudioPCMBuffer, pcm.frameLength > 0 else {
                    resumed = true
                    // Retained until here so the synthesizer is not freed mid-sentence.
                    _ = synthesizer
                    continuation.resume(returning: buffers)
                    return
                }
                buffers.append(pcm)
            }
        }
    }
}

// MARK: - Voices over HTTP

/// OpenAI's speech endpoint, and the same endpoint served by Kokoro on the PC.
struct OpenAISpeechSynthesizer: Synthesizer {
    let baseURL: URL
    let apiKey: String?
    let model: String
    let voice: String

    func audio(for text: String) async throws -> [AVAudioPCMBuffer] {
        var request = URLRequest(url: baseURL.appendingPathComponent("audio/speech"))
        request.httpMethod = "POST"
        request.timeoutInterval = 60
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        if let apiKey {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "authorization")
        }
        request.httpBody = try JSONSerialization.data(withJSONObject: [
            "model": model, "voice": voice, "input": text, "response_format": "mp3"
        ])
        return try await AudioDecoder.fetch(request)
    }
}

struct ElevenLabsSynthesizer: Synthesizer {
    let apiKey: String
    let voiceID: String
    let model: String

    func audio(for text: String) async throws -> [AVAudioPCMBuffer] {
        let voice = voiceID.isEmpty ? "21m00Tcm4TlvDq8ikWAM" : voiceID
        var request = URLRequest(url: URL(string: "https://api.elevenlabs.io/v1/text-to-speech/\(voice)?output_format=mp3_44100_128")!)
        request.httpMethod = "POST"
        request.timeoutInterval = 60
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.setValue(apiKey, forHTTPHeaderField: "xi-api-key")
        request.httpBody = try JSONSerialization.data(withJSONObject: ["text": text, "model_id": model])
        return try await AudioDecoder.fetch(request)
    }
}

enum AudioDecoder {
    static func fetch(_ request: URLRequest) async throws -> [AVAudioPCMBuffer] {
        let (data, response) = try await URLSession.shared.data(for: request)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard status == 200 else {
            throw BrainError(message: "The voice failed. " + SSE.readableError(status: status, body: String(decoding: data, as: UTF8.self)))
        }
        return try decode(data)
    }

    /// Compressed audio (mp3, wav) to one PCM buffer.
    static func decode(_ data: Data) throws -> [AVAudioPCMBuffer] {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".mp3")
        try data.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let file = try AVAudioFile(forReading: url)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)) else {
            return []
        }
        try file.read(into: buffer)
        return [buffer]
    }
}
