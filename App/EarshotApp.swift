import SwiftUI
import UIKit

@MainActor
final class AppState: ObservableObject {
    let settings: AppSettings
    let store: ChatStore
    let spend: SpendMeter
    let conversation: Conversation

    init() {
        settings = AppSettings()
        store = ChatStore()
        spend = SpendMeter()
        conversation = Conversation(settings: settings, store: store, spend: spend)
    }
}

@main
struct EarshotApp: App {
    @StateObject private var state = AppState()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(state.settings)
                .environmentObject(state.store)
                .environmentObject(state.spend)
                .environmentObject(state.conversation)
                .preferredColorScheme(.dark)
                .tint(Theme.clay)
        }
    }
}

/// A picture that arrived on the clipboard since Earshot last looked: what the Action Button
/// shortcut copies.
enum ClipboardImage {
    private static let key = "clipboard.changeCount"

    static func takeNew() -> UIImage? {
        let board = UIPasteboard.general
        let seen = UserDefaults.standard.object(forKey: key) as? Int
        UserDefaults.standard.set(board.changeCount, forKey: key)
        // The first launch only takes note: an old picture on the clipboard is not a question.
        guard let seen, seen != board.changeCount, board.hasImages else { return nil }
        return board.image
    }
}

struct RootView: View {
    @EnvironmentObject var conversation: Conversation
    @EnvironmentObject var settings: AppSettings
    @EnvironmentObject var spend: SpendMeter
    @Environment(\.scenePhase) private var scenePhase

    @State private var showChats = false
    @State private var showSettings = false
    @State private var showModels = false
    @State private var typing = false
    @State private var leftAt: Date? = .distantPast

    var body: some View {
        ZStack {
            Theme.ground.ignoresSafeArea()
            if conversation.chat == nil {
                TalkView(openChats: { showChats = true }, openSettings: { showSettings = true }, openModels: { showModels = true })
            } else {
                ChatView(openChats: { showChats = true }, openModels: { showModels = true }, typing: $typing)
            }
            if conversation.phase == .listening || conversation.phase == .transcribing {
                ListeningView(typeInstead: { typing = true })
                    .transition(.opacity)
            }
        }
        .animation(.easeOut(duration: 0.2), value: conversation.phase)
        .sheet(isPresented: $showModels) { ModelsSheet() }
        .fullScreenCover(isPresented: $showChats) { ChatsView() }
        .fullScreenCover(isPresented: $showSettings) { SettingsView() }
        .onChange(of: scenePhase) { _, phase in
            switch phase {
            case .active: arrived()
            case .background: leftAt = Date()
            default: break
            }
        }
        .onOpenURL { url in
            // earshot://ask?q=… asks a typed question.
            if let query = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems?.first(where: { $0.name == "q" })?.value,
               !query.isEmpty {
                conversation.newChat()
                conversation.submit(query)
            }
        }
    }

    private var sheetIsOpen: Bool { showChats || showSettings || showModels }

    private func arrived() {
        spend.refresh()
        let away = leftAt.map { Date().timeIntervalSince($0) } ?? 0
        leftAt = nil
        if let image = ClipboardImage.takeNew() {
            if conversation.phase == .idle {
                startFresh()
                conversation.attach(image, label: "Screenshot")
                closeSheets()
                conversation.startListening()
            } else {
                conversation.attach(image, label: "Screenshot")
            }
            return
        }
        guard settings.data.listenOnOpen, conversation.phase == .idle, !sheetIsOpen, !typing, away > 20 else { return }
        startFresh()
        conversation.startListening()
    }

    /// A chat left a while ago is finished; a question now starts a new one.
    private func startFresh() {
        if let chat = conversation.chat, Date().timeIntervalSince(chat.updated) > 300 {
            conversation.newChat()
        }
    }

    private func closeSheets() {
        showChats = false
        showSettings = false
        showModels = false
        typing = false
    }
}
