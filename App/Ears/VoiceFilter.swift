import AVFoundation
import Foundation
import LocalVoices
import SwiftUI

/// How alike the speech must sound to the user's recording, as cosine similarity of CAM++ voiceprints.
/// Measured on the phone (2026-10-28): the two halves of the user's recording 0.90, the Alba and
/// Kyon voices 0.56 and 0.47. Speech at another moment scores below the 0.90, so the bars sit between.
enum VoiceFilterLevel: String, Codable, CaseIterable {
    case relaxed, normal, strict

    var label: String {
        switch self {
        case .relaxed: return "Relaxed"
        case .normal: return "Normal"
        case .strict: return "Strict"
        }
    }

    var threshold: Float {
        switch self {
        case .relaxed: return 0.58
        case .normal: return 0.65
        case .strict: return 0.72
        }
    }

    var note: String {
        switch self {
        case .relaxed: return "Lets through anything close to your voice. Use it if you're being ignored."
        case .normal: return "Ignores other people and the TV, but lets you through when you sound a little different."
        case .strict: return "Only speech that sounds clearly like your recording."
        }
    }
}

/// "Only my voice": compares what was heard with the voiceprint of the user's "My voice" recording,
/// so that speech from someone else or the TV is not sent as a question.
@MainActor
final class VoiceFilter: ObservableObject {
    static let shared = VoiceFilter()

    @Published private(set) var progress: Double?
    @Published private(set) var failure: String?
    /// Bumped when the model is added or removed so views re-read the disk.
    @Published private(set) var version = 0
    private var task: Task<Void, Never>?
    /// The recording's print, with the file date it was made from.
    private var enrolled: (date: Date, print: [Float])?

    var isDownloaded: Bool {
        _ = version
        return LocalModelStore.shared.isDownloaded(CamPlusModel.model)
    }

    var hasRecording: Bool { FileManager.default.fileExists(atPath: VoiceSamples.myFile.path) }

    func isOn(_ settings: AppSettings) -> Bool {
        settings.data.voiceFilter && isDownloaded && hasRecording
    }

    /// Whether `samples` sound like the user. Nil when it can't tell (too little speech, no recording,
    /// a failure); the speech is then let through rather than lost.
    func match(_ samples: [Float], settings: AppSettings) async -> (isYou: Bool, score: Float)? {
        guard let score = await score(samples) else { return nil }
        return (score >= settings.data.voiceFilterLevel.threshold, score)
    }

    func score(_ samples: [Float]) async -> Float? {
        do {
            guard let mine = try await recordingPrint(),
                  let heard = try await SpeakerPrint.shared.print(samples) else { return nil }
            return SpeakerPrint.similarity(mine, heard)
        } catch {
            talkLog.notice("voice filter failed: \(error.localizedDescription, privacy: .public)")
            return nil
        }
    }

    /// The scores of the voices that come with Earshot against the user's recording: none of them are
    /// the user, so all should be well under the threshold.
    func otherVoices() async -> [(name: String, score: Float)] {
        var result: [(String, Float)] = []
        // The same person for comparison: the two halves of the user's own recording.
        if let mine = FluidSample.read(VoiceSamples.myFile), mine.count >= 2 * SpeakerPrint.minimumSamples,
           let first = try? await SpeakerPrint.shared.print(Array(mine[..<(mine.count / 2)])),
           let second = try? await SpeakerPrint.shared.print(Array(mine[(mine.count / 2)...])) {
            result.append(("You vs you", SpeakerPrint.similarity(first, second)))
        }
        for name in ["alba", "kyon"] {
            guard let url = Bundle.main.url(forResource: name, withExtension: "wav"),
                  let samples = FluidSample.read(url), let score = await score(samples) else { continue }
            result.append((name.capitalized, score))
        }
        return result
    }

    private func recordingPrint() async throws -> [Float]? {
        let file = VoiceSamples.myFile
        guard let date = (try? FileManager.default.attributesOfItem(atPath: file.path))?[.modificationDate] as? Date else { return nil }
        if let enrolled, enrolled.date == date { return enrolled.print }
        guard let samples = FluidSample.read(file), let print = try await SpeakerPrint.shared.print(samples) else { return nil }
        enrolled = (date, print)
        return print
    }

    func start() {
        guard task == nil else { return }
        failure = nil
        progress = 0
        task = Task {
            do {
                try await LocalModelStore.shared.download(CamPlusModel.model) { done, total in
                    Task { @MainActor in self.progress = total > 0 ? Double(done) / Double(total) : 0 }
                }
            } catch is CancellationError {
            } catch {
                failure = error.localizedDescription
            }
            task = nil
            progress = nil
            version += 1
        }
    }

    func delete() {
        Task {
            await SpeakerPrint.shared.unload()
            try? LocalModelStore.shared.delete(CamPlusModel.model)
            enrolled = nil
            version += 1
        }
    }
}

/// Apple's microphone modes. Voice Isolation is picked by the user in Control Center; apps can only
/// read the mode and open the picker.
enum MicMode {
    static var current: String {
        switch AVCaptureDevice.activeMicrophoneMode {
        case .voiceIsolation: return "Voice Isolation"
        case .wideSpectrum: return "Wide Spectrum"
        default: return "Standard"
        }
    }

    static func showPicker() {
        AVCaptureDevice.showSystemUserInterface(.microphoneModes)
    }
}

/// The "Only my voice" page: Voice Isolation, and the voiceprint filter with a test.
struct VoiceFilterPage: View {
    @EnvironmentObject var settings: AppSettings
    @EnvironmentObject var conversation: Conversation
    @ObservedObject private var filter = VoiceFilter.shared
    @Environment(\.dismiss) private var dismiss
    @State private var micMode = MicMode.current
    @State private var recorder: SampleRecorder?
    @State private var result: String?
    @State private var others: String?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                PageHeader(title: "Only my voice", back: { dismiss() }) { EmptyView() }

                VStack(alignment: .leading, spacing: 10) {
                    GroupLabel(text: "Voice Isolation · Apple")
                    RowGroup {
                        Button {
                            MicMode.showPicker()
                        } label: {
                            RowLabel(title: "Microphone mode", subtitle: "Voice Isolation cuts background noise and other sounds. Pick it while Earshot is listening.") {
                                Text(micMode).font(.system(size: 14)).foregroundStyle(Theme.secondary)
                                Chevron()
                            }
                        }
                        .buttonStyle(.plain)
                    }
                }

                VStack(alignment: .leading, spacing: 10) {
                    GroupLabel(text: "Voiceprint · ignores other voices")
                    RowGroup {
                        RowLabel(title: "Only answer my voice",
                                 subtitle: filter.hasRecording ? "Speech that doesn't sound like your recording isn't sent." : "Record \"My voice\" first: Models › Voice › On this iPhone.") {
                            ClayToggle(isOn: Binding(get: { settings.data.voiceFilter }, set: { on in
                                settings.data.voiceFilter = on
                                if on && !filter.isDownloaded { filter.start() }
                            }))
                            .disabled(!filter.hasRecording)
                        }
                        Hairline()
                        RowLabel(title: filter.isDownloaded ? "Voiceprint model" : "Voiceprint model · not downloaded",
                                 subtitle: filter.failure ?? "CAM++ · \(OnDeviceVoicesPage.size(CamPlusModel.approximateBytes)) · on this iPhone") {
                            model
                        }
                    }
                    if settings.data.voiceFilter {
                        VStack(alignment: .leading, spacing: 8) {
                            Segmented(options: VoiceFilterLevel.allCases.map { ($0, $0.label) },
                                      selection: $settings.data.voiceFilterLevel, height: 36)
                            Text(settings.data.voiceFilterLevel.note).font(.system(size: 13)).foregroundStyle(Theme.muted)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                    Text("It checks a question that sends itself when you pause. Tapping send always sends. It can't pull your voice out of someone talking over you; Voice Isolation helps with that.")
                        .font(.system(size: 13)).foregroundStyle(Theme.muted).fixedSize(horizontal: false, vertical: true)
                }

                if filter.isDownloaded && filter.hasRecording {
                    VStack(alignment: .leading, spacing: 10) {
                        GroupLabel(text: "Test it")
                        PrimaryPill(title: recorder == nil ? "Say something" : "Stop and check",
                                    systemImage: recorder == nil ? "mic" : "stop.fill") {
                            Task { await toggleTest() }
                        }
                        if let result {
                            Text(result).font(.system(size: 15)).foregroundStyle(Theme.text)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        Button {
                            Task { await checkOthers() }
                        } label: {
                            Text("Check against the built-in voices").font(.system(size: 15, weight: .medium)).foregroundStyle(Theme.clay)
                        }
                        .buttonStyle(.plain)
                        if let others {
                            Text(others).font(.system(size: 14)).foregroundStyle(Theme.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                }

                Text("CAM++ by FunASR (Alibaba DAMO), converted to Core ML by FluidInference.")
                    .font(.system(size: 12)).foregroundStyle(Theme.muted)
            }
            .padding(20)
        }
        .background(Theme.ground.ignoresSafeArea())
        .toolbar(.hidden, for: .navigationBar)
        .onAppear { micMode = MicMode.current }
        .onDisappear {
            if let recorder { Task { try? await conversation.stopSampleRecording(recorder, save: false) } }
        }
    }

    @ViewBuilder private var model: some View {
        if let progress = filter.progress {
            Text("\(Int(progress * 100))%").font(.system(size: 14)).foregroundStyle(Theme.secondary)
        } else if filter.isDownloaded {
            Button { filter.delete(); settings.data.voiceFilter = false } label: {
                Image(systemName: "trash").font(.system(size: 13)).foregroundStyle(Theme.muted).frame(width: 32, height: 32)
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Delete the voiceprint model")
        } else {
            Button { filter.start() } label: {
                Text("Get").font(.system(size: 14, weight: .semibold)).foregroundStyle(Theme.onClay)
                    .padding(.horizontal, 12).frame(height: 32).background(Capsule().fill(Theme.clay))
            }
            .buttonStyle(.plain)
        }
    }

    private func toggleTest() async {
        if let current = recorder {
            recorder = nil
            let samples = current.recorded
            try? await conversation.stopSampleRecording(current, save: false)
            guard let score = await filter.score(samples) else {
                result = "Too little speech to tell. Talk for at least two seconds."
                return
            }
            let threshold = settings.data.voiceFilterLevel.threshold
            result = "Match \(Int(score * 100))% · " + (score >= threshold
                ? "that's you. It would be sent."
                : "below \(Int(threshold * 100))%, so it would be ignored.")
        } else {
            result = nil
            recorder = await conversation.startSampleRecording()
            micMode = MicMode.current
        }
    }

    private func checkOthers() async {
        others = "Checking…"
        let scores = await filter.otherVoices()
        let threshold = settings.data.voiceFilterLevel.threshold
        others = scores.isEmpty ? "Couldn't check."
            : scores.map { "\($0.name) \(Int($0.score * 100))%" }.joined(separator: " · ")
              + " · needed \(Int(threshold * 100))%. These aren't you, so they should be lower."
    }
}
