import LocalVoices
import SwiftUI

/// The voice models that run on the phone: download, pick a voice, hear it.
struct OnDeviceVoicesPage: View {
    @EnvironmentObject var settings: AppSettings
    @EnvironmentObject var conversation: Conversation
    @ObservedObject private var downloads = LocalVoiceDownloads.shared
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        ScrollView {
            VStack(spacing: 18) {
                PageHeader(title: "On this iPhone", back: { dismiss() }) { EmptyView() }
                Text("These voices run on your iPhone: free, private, and they work without internet. Each one downloads once; Wi-Fi is best. While Earshot is in the background, Apple's voice speaks instead.")
                    .font(.system(size: 14)).foregroundStyle(Theme.muted).frame(maxWidth: .infinity, alignment: .leading)
                ForEach(LocalVoiceModel.allCases, id: \.self) { model in
                    section(model)
                }
            }
            .padding(20)
        }
        .background(Theme.ground.ignoresSafeArea())
        .toolbar(.hidden, for: .navigationBar)
    }

    private var current: OnDeviceVoice? {
        settings.preset.voice.kind == .onDevice ? OnDeviceVoice(settings.preset.voice.voice) : nil
    }

    private func section(_ model: LocalVoiceModel) -> some View {
        let info = model.info
        let downloaded = downloads.isDownloaded(model)
        return VStack(spacing: 10) {
            GroupLabel(text: "\(info.name) · \(info.maker)")
            RowGroup {
                RowLabel(title: "\(info.parameters) · \(info.license)", subtitle: info.note) {
                    downloadControl(model, downloaded: downloaded)
                }
                if let failure = downloads.failures[model] {
                    Hairline()
                    Text(failure).font(.system(size: 13)).foregroundStyle(Theme.danger)
                        .frame(maxWidth: .infinity, alignment: .leading).padding(.horizontal, 16).padding(.vertical, 10)
                }
                if downloaded {
                    ForEach(voices(model), id: \.voice) { voice in
                        Hairline()
                        voiceRow(voice)
                    }
                    if info.clonesVoices && VoiceSamples.mine == nil {
                        Hairline()
                        NavigationLink(value: ModelsRoute.myVoice) {
                            RowLabel(title: "Record my voice", subtitle: "Read one sentence; this model copies how you sound", titleColor: Theme.clay) {
                                Chevron()
                            }
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
        }
    }

    private func voices(_ model: LocalVoiceModel) -> [OnDeviceVoice] {
        var list = model.info.voices.map { OnDeviceVoice(model: model, voice: $0.id) }
        if model.info.needsReference {
            list.append(OnDeviceVoice(model: model, voice: OnDeviceVoice.builtInReference))
        }
        if model.info.clonesVoices && VoiceSamples.mine != nil {
            list.append(OnDeviceVoice(model: model, voice: OnDeviceVoice.myReference))
        }
        return list
    }

    private func voiceRow(_ voice: OnDeviceVoice) -> some View {
        let detail = voice.model.info.voices.first(where: { $0.id == voice.voice })?.detail ?? ""
        return HStack(spacing: 0) {
            Button {
                settings.data.onDeviceVoice = voice.stored
                settings.preset.voice = VoiceChoice(kind: .onDevice, voice: voice.stored)
            } label: {
                RowLabel(title: voice.voiceName, subtitle: detail) {
                    if current == voice { Checkmark() }
                }
            }
            .buttonStyle(.plain)
            PlayButton { conversation.preview(VoiceChoice(kind: .onDevice, voice: voice.stored)) }
                .padding(.trailing, 10)
        }
    }

    @ViewBuilder private func downloadControl(_ model: LocalVoiceModel, downloaded: Bool) -> some View {
        if let progress = downloads.progress[model] {
            HStack(spacing: 8) {
                VStack(alignment: .trailing, spacing: 4) {
                    ProgressView(value: progress.fraction).tint(Theme.clay).frame(width: 70)
                    Text("\(Int(progress.fraction * 100))%").font(.system(size: 12)).foregroundStyle(Theme.muted)
                }
                Button { downloads.cancel(model) } label: {
                    Image(systemName: "xmark").font(.system(size: 12, weight: .semibold)).foregroundStyle(Theme.secondary)
                        .frame(width: 28, height: 28).background(Circle().fill(Theme.raised))
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Stop downloading")
            }
        } else if downloaded {
            Button { downloads.delete(model) } label: {
                Image(systemName: "trash").font(.system(size: 13)).foregroundStyle(Theme.muted)
                    .frame(width: 32, height: 32)
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Delete \(model.info.name) from this iPhone")
        } else {
            Button { downloads.start(model) } label: {
                Text("Get · " + Self.size(model.info.approximateBytes))
                    .font(.system(size: 14, weight: .semibold)).foregroundStyle(Theme.onClay)
                    .padding(.horizontal, 12).frame(height: 32)
                    .background(Capsule().fill(Theme.clay))
            }
            .buttonStyle(.plain)
        }
    }

    static func size(_ bytes: Int64) -> String {
        bytes >= 1_000_000_000 ? String(format: "%.1f GB", Double(bytes) / 1e9) : "\(bytes / 1_000_000) MB"
    }
}

/// Records the user reading one sentence, for the models that copy a voice.
struct MyVoicePage: View {
    @EnvironmentObject var conversation: Conversation
    @Environment(\.dismiss) private var dismiss
    @State private var recorder: SampleRecorder?
    @State private var seconds = 0.0
    @State private var message: String?
    @State private var saved = VoiceSamples.mine != nil
    private let timer = Timer.publish(every: 0.2, on: .main, in: .common).autoconnect()

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            PageHeader(title: "My voice", back: { stop(save: false); dismiss() }) { EmptyView() }
            Text("Read this out loud at your normal pace, in a quiet room:")
                .font(.system(size: 14)).foregroundStyle(Theme.muted)
            Text("“\(VoiceSamples.readAloud)”")
                .font(Theme.serif(21)).foregroundStyle(Theme.text)
                .padding(16).frame(maxWidth: .infinity, alignment: .leading)
                .background(RoundedRectangle(cornerRadius: 16).fill(Theme.surface))
            if recorder != nil {
                PrimaryPill(title: String(format: "Stop · %.0f s", seconds), systemImage: "stop.fill") { stop(save: true) }
            } else {
                PrimaryPill(title: saved ? "Record again" : "Record", systemImage: "mic.fill") { start() }
            }
            if let message {
                Text(message).font(.system(size: 14)).foregroundStyle(Theme.textBody)
            }
            if saved && recorder == nil {
                Button("Delete my recording", role: .destructive) {
                    try? FileManager.default.removeItem(at: VoiceSamples.myFile)
                    saved = false
                    message = nil
                }
                .font(.system(size: 15)).foregroundStyle(Theme.danger)
            }
            Spacer()
        }
        .padding(20)
        .background(Theme.ground.ignoresSafeArea())
        .toolbar(.hidden, for: .navigationBar)
        .onReceive(timer) { _ in
            seconds = recorder?.seconds ?? 0
        }
        .onDisappear { stop(save: false) }
    }

    private func start() {
        message = nil
        Task {
            if let made = await conversation.startSampleRecording() {
                recorder = made
            } else {
                message = conversation.error ?? "The microphone could not start."
            }
        }
    }

    private func stop(save: Bool) {
        guard let recorder else { return }
        self.recorder = nil
        Task {
            do {
                try await conversation.stopSampleRecording(recorder, save: save)
                if save {
                    saved = true
                    message = "Saved. Pick “My voice” under any model that copies voices."
                }
            } catch {
                message = error.localizedDescription
            }
        }
    }
}
