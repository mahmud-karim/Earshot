import AVFoundation
import CoreMedia
import Foundation
import Speech

// The part that listens. Audio arrives as 16 kHz mono samples on the audio thread.

protocol Transcriber: AnyObject {
    /// The words so far, on the main thread. Providers that only transcribe at the end never call it.
    var onText: ((String) -> Void)? { get set }
    /// Set while the transcriber gets ready, e.g. "Downloading speech model…".
    var onStatus: ((String?) -> Void)? { get set }
    var givesLiveText: Bool { get }
    func start() async throws
    /// Thread-safe; called on the audio thread.
    func feed(_ samples: [Float])
    func finish() async throws -> String
    func cancel()
}

// MARK: - Apple, on the device

/// The iOS 26 recogniser. It runs on the device and asks for no speech-recognition permission.
final class AppleTranscriber: Transcriber {
    var onText: ((String) -> Void)?
    var onStatus: ((String?) -> Void)?
    let givesLiveText = true

    private var analyzer: SpeechAnalyzer?
    private var input: AsyncStream<AnalyzerInput>.Continuation?
    private var relay: Task<Void, Never>?
    private var analyzerFormat: AVAudioFormat?
    private var converter: AVAudioConverter?
    private let lock = NSLock()
    private var finalText = ""
    private var volatileText = ""
    private let captured = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false)!

    func start() async throws {
        guard SpeechTranscriber.isAvailable else {
            throw BrainError(message: "Apple speech isn't supported on this iPhone. Pick other Ears in Models.")
        }
        guard let locale = await AppleTranscriber.locale() else {
            throw BrainError(message: "English isn't available for Apple speech here.")
        }
        let transcriber = SpeechTranscriber(locale: locale, transcriptionOptions: [],
                                            reportingOptions: [.volatileResults, .fastResults], attributeOptions: [])
        if let request = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
            await MainActor.run { self.onStatus?("Downloading the speech model…") }
            try await request.downloadAndInstall()
            await MainActor.run { self.onStatus?(nil) }
        }

        let format = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [transcriber], considering: captured) ?? captured
        lock.withLock {
            analyzerFormat = format
            converter = format == captured ? nil : AVAudioConverter(from: captured, to: format)
            finalText = ""
            volatileText = ""
        }

        let analyzer = SpeechAnalyzer(modules: [transcriber])
        let (inputs, continuation) = AsyncStream<AnalyzerInput>.makeStream()
        relay = Task { [weak self] in
            do {
                for try await result in transcriber.results {
                    self?.take(String(result.text.characters), isFinal: result.isFinal)
                }
            } catch {}
        }
        try await analyzer.start(inputSequence: inputs)
        lock.withLock {
            self.analyzer = analyzer
            self.input = continuation
        }
    }

    func feed(_ samples: [Float]) {
        lock.lock()
        let input = self.input
        let format = analyzerFormat
        let converter = self.converter
        lock.unlock()
        guard let input, let format, !samples.isEmpty,
              let source = AVAudioPCMBuffer(pcmFormat: captured, frameCapacity: AVAudioFrameCount(samples.count)) else { return }
        source.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { pointer in
            source.floatChannelData![0].update(from: pointer.baseAddress!, count: pointer.count)
        }
        guard let converter else {
            input.yield(AnalyzerInput(buffer: source))
            return
        }
        let ratio = format.sampleRate / captured.sampleRate
        guard let output = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(Double(samples.count) * ratio) + 64) else { return }
        var given = false
        var error: NSError?
        converter.convert(to: output, error: &error) { _, status in
            if given {
                status.pointee = .noDataNow
                return nil
            }
            given = true
            status.pointee = .haveData
            return source
        }
        if error == nil, output.frameLength > 0 {
            input.yield(AnalyzerInput(buffer: output))
        }
    }

    func finish() async throws -> String {
        let (analyzer, input) = lock.withLock { () -> (SpeechAnalyzer?, AsyncStream<AnalyzerInput>.Continuation?) in
            let pair = (self.analyzer, self.input)
            self.input = nil
            return pair
        }
        input?.finish()
        if let analyzer {
            try? await analyzer.finalizeAndFinishThroughEndOfInput()
        }
        await relay?.value
        return lock.withLock { (finalText + volatileText).trimmingCharacters(in: .whitespacesAndNewlines) }
    }

    func cancel() {
        lock.lock()
        let analyzer = self.analyzer
        input?.finish()
        input = nil
        self.analyzer = nil
        lock.unlock()
        relay?.cancel()
        Task { await analyzer?.cancelAndFinishNow() }
    }

    private func take(_ text: String, isFinal: Bool) {
        lock.lock()
        if isFinal {
            finalText += text
            volatileText = ""
        } else {
            volatileText = text
        }
        let whole = (finalText + volatileText).trimmingCharacters(in: .whitespacesAndNewlines)
        lock.unlock()
        DispatchQueue.main.async { self.onText?(whole) }
    }

    private static func locale() async -> Locale? {
        let device = Locale.current
        if device.language.languageCode?.identifier == "en",
           let match = await SpeechTranscriber.supportedLocale(equivalentTo: device) {
            return match
        }
        return await SpeechTranscriber.supportedLocale(equivalentTo: Locale(identifier: "en_US"))
    }
}

// MARK: - Whisper-style servers

/// Records the whole question, then sends it to an OpenAI-compatible transcription endpoint: OpenAI
/// itself, or a Whisper server on the PC.
final class UploadTranscriber: Transcriber {
    var onText: ((String) -> Void)?
    var onStatus: ((String?) -> Void)?
    let givesLiveText = false

    let baseURL: URL
    let apiKey: String?
    let model: String
    /// Sent when known: skips the server's language detection.
    let language: String?
    /// Turns a failed status and its body into a sentence; the generic one fits OpenAI.
    let describe: (Int, String) -> String

    private let lock = NSLock()
    private var samples: [Float] = []
    private var cancelled = false

    init(baseURL: URL, apiKey: String?, model: String, language: String? = nil,
         describe: @escaping (Int, String) -> String = { "Transcription failed. " + SSE.readableError(status: $0, body: $1) }) {
        self.baseURL = baseURL
        self.apiKey = apiKey
        self.model = model
        self.language = language
        self.describe = describe
    }

    func start() async throws {
        lock.withLock {
            samples = []
            cancelled = false
        }
    }

    func feed(_ new: [Float]) {
        lock.lock()
        if !cancelled { samples.append(contentsOf: new) }
        lock.unlock()
    }

    func cancel() {
        lock.lock()
        cancelled = true
        samples = []
        lock.unlock()
    }

    func finish() async throws -> String {
        let recorded = lock.withLock { samples }
        guard recorded.count > 8_000 else { return "" }

        let boundary = "earshot-\(UUID().uuidString)"
        var body = Data()
        func field(_ name: String, _ value: String) {
            body.append("--\(boundary)\r\nContent-Disposition: form-data; name=\"\(name)\"\r\n\r\n\(value)\r\n".data(using: .utf8)!)
        }
        field("model", model)
        field("response_format", "json")
        if let language { field("language", language) }
        body.append("--\(boundary)\r\nContent-Disposition: form-data; name=\"file\"; filename=\"speech.wav\"\r\nContent-Type: audio/wav\r\n\r\n".data(using: .utf8)!)
        body.append(WAV.encode(recorded, sampleRate: 16_000))
        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)

        var request = URLRequest(url: baseURL.appendingPathComponent("audio/transcriptions"))
        request.httpMethod = "POST"
        request.timeoutInterval = 60
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "content-type")
        if let apiKey {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "authorization")
        }
        let (data, response) = try await URLSession.shared.upload(for: request, from: body)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard status == 200 else {
            throw BrainError(message: describe(status, String(decoding: data, as: UTF8.self)))
        }
        let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        return ((object?["text"] as? String) ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

enum WAV {
    /// 16-bit PCM mono.
    static func encode(_ samples: [Float], sampleRate: Int) -> Data {
        var data = Data()
        func append<T: FixedWidthInteger>(_ value: T) {
            var little = value.littleEndian
            withUnsafeBytes(of: &little) { data.append(contentsOf: $0) }
        }
        let byteCount = samples.count * 2
        data.append("RIFF".data(using: .ascii)!)
        append(UInt32(36 + byteCount))
        data.append("WAVEfmt ".data(using: .ascii)!)
        append(UInt32(16))
        append(UInt16(1))
        append(UInt16(1))
        append(UInt32(sampleRate))
        append(UInt32(sampleRate * 2))
        append(UInt16(2))
        append(UInt16(16))
        data.append("data".data(using: .ascii)!)
        append(UInt32(byteCount))
        data.reserveCapacity(data.count + byteCount)
        for sample in samples {
            append(Int16(max(-1, min(1, sample)) * 32_767))
        }
        return data
    }
}
