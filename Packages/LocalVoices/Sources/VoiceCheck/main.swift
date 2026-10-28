import Foundation
import LocalVoices
import MLX
import MLXAudioCore

// voice-check <model> [--voice id] [--ref clip.wav --ref-text "words"] [--text "one sentence"]...
//             [--out folder] [--models folder] [--name prefix] [--cpu]
//
// Downloads the model, loads it, speaks each sentence and writes one WAV per sentence plus
// report.json with the timings. CI runs it for every local voice; nothing here runs on the phone.

struct Options {
    var model: LocalVoiceModel
    var voice: String?
    var reference: VoiceReference?
    var texts: [String] = []
    var out = URL(fileURLWithPath: "voice-out", isDirectory: true)
    var models = URL(fileURLWithPath: "models", isDirectory: true)
    var name: String?
    var cpu = false
}

let defaultTexts = [
    "Hi, I'm Earshot. What can I help you with today?",
    "The weather in Boston is fifty eight degrees and cloudy, with light rain after six.",
    "Sure, I've set a timer for twelve minutes."
]

func parse() -> Options {
    var arguments = Array(CommandLine.arguments.dropFirst())
    guard let first = arguments.first, let model = LocalVoiceModel(rawValue: first) else {
        let names = LocalVoiceModel.allCases.map(\.rawValue).joined(separator: ", ")
        FileHandle.standardError.write("usage: voice-check <\(names)> [options]\n".data(using: .utf8)!)
        exit(2)
    }
    arguments.removeFirst()
    var options = Options(model: model)
    var refAudio: String?
    var refText: String?
    while !arguments.isEmpty {
        let flag = arguments.removeFirst()
        func value() -> String {
            guard !arguments.isEmpty else { fatalError("\(flag) needs a value") }
            return arguments.removeFirst()
        }
        switch flag {
        case "--voice": options.voice = value()
        case "--ref": refAudio = value()
        case "--ref-text": refText = value()
        case "--text": options.texts.append(value())
        case "--out": options.out = URL(fileURLWithPath: value(), isDirectory: true)
        case "--models": options.models = URL(fileURLWithPath: value(), isDirectory: true)
        case "--name": options.name = value()
        case "--cpu": options.cpu = true
        default: fatalError("unknown option \(flag)")
        }
    }
    if let refAudio {
        options.reference = VoiceReference(audio: URL(fileURLWithPath: refAudio), transcript: refText ?? "")
    }
    if options.texts.isEmpty { options.texts = defaultTexts }
    return options
}

func seconds(since start: Date) -> Double {
    (Date().timeIntervalSince(start) * 1000).rounded() / 1000
}

let options = parse()
if options.cpu { Device.setDefault(device: .cpu) }
print("device: \(Device.defaultDevice())")

let store = LocalModelStore(root: options.models)
let engine = LocalVoiceEngine(store: store)
try FileManager.default.createDirectory(at: options.out, withIntermediateDirectories: true)

var report: [String: Any] = ["model": options.model.rawValue, "voice": options.voice ?? NSNull()]
let downloadStart = Date()
var lastShown = Date.distantPast
try await store.download(options.model) { done, total in
    if Date().timeIntervalSince(lastShown) > 10 {
        lastShown = Date()
        print(String(format: "download %.0f / %.0f MB", Double(done) / 1e6, Double(total) / 1e6))
    }
}
report["download_seconds"] = seconds(since: downloadStart)
report["bytes_on_disk"] = store.bytesOnDisk(options.model)

let loadStart = Date()
try await engine.prepare(options.model)
report["load_seconds"] = seconds(since: loadStart)
print("loaded in \(report["load_seconds"]!) s")

var sentences: [[String: Any]] = []
let prefix = options.name ?? options.model.rawValue
for (index, text) in options.texts.enumerated() {
    let start = Date()
    let (samples, rate) = try await engine.speak(text, model: options.model, voice: options.voice, reference: options.reference)
    let took = seconds(since: start)
    let audioSeconds = Double(samples.count) / Double(rate)
    let file = options.out.appendingPathComponent("\(prefix)-\(index + 1).wav")
    try AudioUtils.writeWavFile(samples: samples, sampleRate: rate, fileURL: file)
    let peak = samples.reduce(Float(0)) { max($0, abs($1)) }
    print(String(format: "%@: %.2f s of audio in %.2f s (%.2fx real time), peak %.3f", file.lastPathComponent, audioSeconds, took, took / max(audioSeconds, 0.001), peak))
    sentences.append([
        "text": text, "file": file.lastPathComponent, "sample_rate": rate,
        "audio_seconds": audioSeconds, "generate_seconds": took, "peak": peak
    ])
}
report["sentences"] = sentences
report["peak_memory_mb"] = Memory.peakMemory / 1_000_000
let json = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
try json.write(to: options.out.appendingPathComponent("\(prefix)-report.json"))
print(String(decoding: json, as: UTF8.self))
