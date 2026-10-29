import AVFoundation
import Foundation
import LocalVoices
import SwiftUI

/// Fluid Speech Mini on the phone. It has no streaming mode, so the live words come from transcribing
/// what was heard so far about once a second; the final text is one pass over the whole question.
final class FluidTranscriber: Transcriber {
    var onText: ((String) -> Void)?
    var onStatus: ((String?) -> Void)?
    let givesLiveText = true

    private let lock = NSLock()
    private var samples: [Float] = []
    private var live: Task<Void, Never>?

    func start() async throws {
        lock.withLock { samples = [] }
        if !(await FluidSpeech.shared.isLoaded) {
            await MainActor.run { self.onStatus?("Loading Fluid Speech Mini…") }
            do {
                try await FluidSpeech.shared.load()
            } catch {
                await MainActor.run { self.onStatus?(nil) }
                throw error
            }
            await MainActor.run { self.onStatus?(nil) }
        }
        live = Task { [weak self] in
            var transcribed = 0
            var shown = ""
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 700_000_000)
                guard let self else { return }
                let heard = self.lock.withLock { self.samples }
                // A pass only when there is new sound. The window stops at 30 seconds back, so a long
                // question keeps updating quickly; the final pass reads all of it.
                guard heard.count >= transcribed + 8_000 else { continue }
                transcribed = heard.count
                guard let text = try? await FluidSpeech.shared.transcribe(Array(heard.suffix(16_000 * 30))),
                      !text.isEmpty, text != shown, !Task.isCancelled else { continue }
                shown = text
                // Only changed text is passed on: the conversation takes every update as the user still talking.
                await MainActor.run { self.onText?(text) }
            }
        }
    }

    func feed(_ samples: [Float]) {
        lock.withLock { self.samples.append(contentsOf: samples) }
    }

    func finish() async throws -> String {
        live?.cancel()
        live = nil
        let heard = lock.withLock { samples }
        return try await FluidSpeech.shared.transcribe(heard)
    }

    func cancel() {
        live?.cancel()
        live = nil
        lock.withLock { samples = [] }
    }
}

/// Download state of Fluid Speech Mini, for the Ears page.
@MainActor
final class FluidDownload: ObservableObject {
    static let shared = FluidDownload()

    @Published private(set) var progress: Double?
    @Published private(set) var failure: String?
    /// Bumped when the model is added or removed so views re-read the disk.
    @Published private(set) var version = 0
    private var task: Task<Void, Never>?

    var isDownloaded: Bool {
        _ = version
        return LocalModelStore.shared.isDownloaded(FluidSpeechMini.model)
    }

    func start() {
        guard task == nil else { return }
        failure = nil
        progress = 0
        task = Task {
            do {
                try await LocalModelStore.shared.download(FluidSpeechMini.model) { done, total in
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

    func cancel() {
        task?.cancel()
    }

    func delete() {
        Task {
            await FluidSpeech.shared.unload()
            try? LocalModelStore.shared.delete(FluidSpeechMini.model)
            version += 1
        }
    }
}

/// Fluid Speech Mini: download it, try it on a recording, remove it.
struct FluidEarsPage: View {
    @EnvironmentObject var settings: AppSettings
    @ObservedObject private var download = FluidDownload.shared
    @Environment(\.dismiss) private var dismiss
    @State private var trial: String?
    @State private var trying = false
    @State private var deleting = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                PageHeader(title: "Fluid Speech Mini", back: { dismiss() }) { EmptyView() }
                Text("Speech to text on your iPhone's Neural Engine: free, private, and it works without internet. English only. It downloads once, about \(OnDeviceVoicesPage.size(FluidSpeechMini.approximateBytes)); Wi-Fi is best.")
                    .font(.system(size: 14)).foregroundStyle(Theme.muted)
                RowGroup {
                    RowLabel(title: download.isDownloaded ? "On this iPhone" : "Not downloaded",
                             subtitle: download.failure ?? (download.isDownloaded ? "Ready to listen" : "Needed before it can listen")) {
                        control
                    }
                    if download.isDownloaded {
                        Hairline()
                        Button {
                            Task { await tryIt() }
                        } label: {
                            RowLabel(title: trying ? "Listening to a sample…" : "Try it",
                                     subtitle: trial ?? "Transcribes a short recording that comes with Earshot", titleColor: Theme.clay) {
                                EmptyView()
                            }
                        }
                        .buttonStyle(.plain)
                        .disabled(trying)
                    }
                }
                Text("Fluid Speech Mini by FluidVoice Labs, licensed CC BY 4.0. Runs with FluidAudio by FluidInference.")
                    .font(.system(size: 12)).foregroundStyle(Theme.muted)
            }
            .padding(20)
        }
        .background(Theme.ground.ignoresSafeArea())
        .toolbar(.hidden, for: .navigationBar)
        .alert("Delete Fluid Speech Mini?", isPresented: $deleting) {
            Button("Delete", role: .destructive) {
                download.delete()
                if settings.data.ears == .fluid { settings.data.ears = .apple }
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("It frees \(OnDeviceVoicesPage.size(FluidSpeechMini.approximateBytes)). Ears go back to Apple on-device.")
        }
    }

    @ViewBuilder private var control: some View {
        if let progress = download.progress {
            HStack(spacing: 8) {
                VStack(alignment: .trailing, spacing: 4) {
                    ProgressView(value: progress).tint(Theme.clay).frame(width: 70)
                    Text("\(Int(progress * 100))%").font(.system(size: 12)).foregroundStyle(Theme.muted)
                }
                Button { download.cancel() } label: {
                    Image(systemName: "xmark").font(.system(size: 12, weight: .semibold)).foregroundStyle(Theme.secondary)
                        .frame(width: 28, height: 28).background(Circle().fill(Theme.raised))
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Stop downloading")
            }
        } else if download.isDownloaded {
            Button { deleting = true } label: {
                Image(systemName: "trash").font(.system(size: 13)).foregroundStyle(Theme.muted).frame(width: 32, height: 32)
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Delete Fluid Speech Mini from this iPhone")
        } else {
            Button { download.start() } label: {
                Text("Get · " + OnDeviceVoicesPage.size(FluidSpeechMini.approximateBytes))
                    .font(.system(size: 14, weight: .semibold)).foregroundStyle(Theme.onClay)
                    .padding(.horizontal, 12).frame(height: 32)
                    .background(Capsule().fill(Theme.clay))
            }
            .buttonStyle(.plain)
        }
    }

    /// Reads the bundled Alba clip at 16 kHz and shows what the model heard, with the time it took.
    private func tryIt() async {
        trying = true
        defer { trying = false }
        guard let url = Bundle.main.url(forResource: "alba", withExtension: "wav"),
              let samples = FluidSample.read(url) else {
            trial = "The sample recording is missing."
            return
        }
        do {
            let started = Date()
            try await FluidSpeech.shared.load()
            let loaded = Date()
            let text = try await FluidSpeech.shared.transcribe(samples)
            let seconds = Date().timeIntervalSince(loaded)
            trial = "“\(text)” · \(String(format: "%.2f", seconds)) s for \(String(format: "%.1f", Double(samples.count) / 16_000)) s of audio"
                + (loaded.timeIntervalSince(started) > 0.5 ? String(format: " · loaded in %.1f s", loaded.timeIntervalSince(started)) : "")
        } catch {
            trial = error.localizedDescription
        }
    }
}

/// Reads a recording as 16 kHz mono samples.
enum FluidSample {
    static func read(_ url: URL) -> [Float]? {
        guard let file = try? AVAudioFile(forReading: url),
              let source = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)),
              (try? file.read(into: source)) != nil,
              let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false),
              let converted = AudioIO.convert(source, to: target),
              let channel = converted.floatChannelData?[0] else { return nil }
        return Array(UnsafeBufferPointer(start: channel, count: Int(converted.frameLength)))
    }
}
