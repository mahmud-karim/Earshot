import PhotosUI
import SwiftUI

/// Home: the orb, the preset switch and "Tap to talk".
struct TalkView: View {
    @EnvironmentObject var conversation: Conversation
    @EnvironmentObject var settings: AppSettings
    let openChats: () -> Void
    let openSettings: () -> Void
    let openModels: () -> Void

    @State private var typing = false

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                RoundIcon(systemImage: "clock", label: "Chats", action: openChats)
                Spacer()
                ModelPill(action: openModels)
                Spacer()
                RoundIcon(systemImage: "slider.horizontal.3", label: "Settings", action: openSettings)
            }
            .frame(height: 44)

            if let error = conversation.error {
                ErrorBanner(text: error) { conversation.error = nil }.padding(.top, 12)
            }

            Spacer()
            VStack(spacing: 28) {
                Orb()
                VStack(spacing: 10) {
                    Text("What’s on your mind?")
                        .font(Theme.serif(32)).foregroundStyle(Theme.text).multilineTextAlignment(.center)
                    Text("Tap to talk. Press the Action Button in any app to ask about what’s on screen.")
                        .font(.system(size: 15)).foregroundStyle(Theme.muted)
                        .multilineTextAlignment(.center).frame(maxWidth: 290)
                }
            }
            Spacer()

            VStack(spacing: 16) {
                if let attachment = conversation.attachment {
                    AttachmentChip(attachment: attachment) { conversation.attachment = nil }
                }
                PresetSwitch()
                if typing {
                    TypeBar(isOpen: $typing)
                } else {
                    HStack(spacing: 12) {
                        RoundButton(systemImage: "keyboard", label: "Type instead") { typing = true }
                        PrimaryPill(title: "Tap to talk", systemImage: "mic") { conversation.startListening() }
                        PhotoButton()
                    }
                }
                Text("Ears: \(EarsText.name(settings.data.ears)) · Voice: \(settings.preset.voice.label)")
                    .font(.system(size: 13)).foregroundStyle(Theme.muted).lineLimit(1).minimumScaleFactor(0.8)
            }
        }
        .padding(.horizontal, 20)
        .padding(.top, 8)
        .padding(.bottom, 12)
    }
}

enum EarsText {
    static func name(_ kind: EarsKind) -> String {
        switch kind {
        case .apple: return "Apple on-device"
        case .fluid: return "Fluid Speech Mini"
        case .server: return "Qwen3-ASR on your PC"
        case .openAI: return "OpenAI"
        case .pc: return "Whisper on your PC"
        }
    }
}

struct RoundIcon: View {
    let systemImage: String
    let label: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: systemImage).font(.system(size: 19, weight: .regular))
                .foregroundStyle(Theme.secondary).frame(width: 44, height: 44)
        }
        .buttonStyle(.plain)
        .accessibilityLabel(label)
    }
}

/// "Sonnet 5.5 · Balanced ⌄" — opens Models.
struct ModelPill: View {
    @EnvironmentObject var settings: AppSettings
    @EnvironmentObject var conversation: Conversation
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 6) {
                Text(conversation.isAgent ? "Hermes" : settings.preset.brain.label).font(.system(size: 14, weight: .medium)).foregroundStyle(Theme.text)
                Text("· " + (conversation.isAgent ? "shopto-cloud" : settings.preset.name)).font(.system(size: 14)).foregroundStyle(Theme.muted)
                Image(systemName: "chevron.down").font(.system(size: 11, weight: .semibold)).foregroundStyle(Theme.muted)
            }
            .lineLimit(1)
            .padding(.leading, 14).padding(.trailing, 12)
            .frame(height: 36)
            .background(Capsule().fill(Theme.surface))
            .overlay(Capsule().stroke(Theme.border, lineWidth: 0.5))
        }
        .buttonStyle(.plain)
    }
}

/// The presets as a segmented control.
struct PresetSwitch: View {
    @EnvironmentObject var settings: AppSettings

    var body: some View {
        let presets = Array(settings.data.presets.prefix(4))
        Segmented(options: presets.map { ($0.id, PresetSwitch.short($0.name)) },
                  selection: Binding(get: { settings.data.currentPresetID ?? presets[0].id },
                                     set: { settings.data.currentPresetID = $0 }))
    }

    static func short(_ name: String) -> String {
        name == "Quick question" ? "Quick" : name
    }
}

struct PhotoButton: View {
    @EnvironmentObject var conversation: Conversation
    @State private var item: PhotosPickerItem?
    var size: CGFloat = 56

    var body: some View {
        PhotosPicker(selection: $item, matching: .images) {
            Image(systemName: "photo")
                .font(.system(size: size * 0.36, weight: .medium))
                .frame(width: size, height: size)
                .foregroundStyle(Theme.secondary)
                .background(Circle().fill(Theme.surface))
                .overlay(Circle().stroke(Theme.border, lineWidth: 0.5))
        }
        .accessibilityLabel("Attach a photo")
        .onChange(of: item) { _, newItem in
            guard let newItem else { return }
            Task {
                if let data = try? await newItem.loadTransferable(type: Data.self), let image = UIImage(data: data) {
                    conversation.attach(image, label: "Photo")
                }
                item = nil
            }
        }
    }
}

/// A text field in place of the talk button.
struct TypeBar: View {
    @EnvironmentObject var conversation: Conversation
    @Binding var isOpen: Bool
    @State private var text = ""
    @FocusState private var focused: Bool

    var body: some View {
        HStack(spacing: 10) {
            RoundButton(systemImage: "xmark", label: "Close keyboard", size: 44) {
                isOpen = false
            }
            TextField("Ask anything", text: $text, axis: .vertical)
                .font(.system(size: 16))
                .foregroundStyle(Theme.text)
                .lineLimit(1...5)
                .focused($focused)
                .padding(.horizontal, 14).padding(.vertical, 11)
                .background(RoundedRectangle(cornerRadius: 22).fill(Theme.surface))
                .overlay(RoundedRectangle(cornerRadius: 22).stroke(Theme.border, lineWidth: 0.5))
                .submitLabel(.send)
                .onSubmit(send)
            RoundButton(systemImage: "arrow.up", label: "Send", size: 44, filled: true, action: send)
                .disabled(text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        }
        .onAppear { focused = true }
    }

    private func send() {
        let question = text
        text = ""
        isOpen = false
        conversation.submit(question)
    }
}

struct AttachmentChip: View {
    let attachment: Attachment
    let remove: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            if let image = attachment.image {
                Image(uiImage: image).resizable().scaledToFill()
                    .frame(width: 40, height: 56).clipShape(RoundedRectangle(cornerRadius: 8))
            }
            VStack(alignment: .leading, spacing: 2) {
                Text("\(attachment.label) attached").font(.system(size: 15, weight: .medium)).foregroundStyle(Theme.text)
                Text(attachment.date, style: .relative).font(.system(size: 13)).foregroundStyle(Theme.muted)
            }
            Spacer()
            Button("Remove", action: remove)
                .font(.system(size: 13)).foregroundStyle(Theme.secondary)
                .padding(.horizontal, 12).frame(height: 36)
                .background(Capsule().fill(Theme.raised))
                .buttonStyle(.plain)
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 18).fill(Theme.surface))
        .overlay(RoundedRectangle(cornerRadius: 18).stroke(Theme.hairline, lineWidth: 0.5))
    }
}
