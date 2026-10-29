import LocalVoices
import Network
import SwiftUI
import UIKit

@MainActor
final class AppState: ObservableObject {
    let settings: AppSettings
    let store: ChatStore
    let spend: SpendMeter
    /// The Talk tab: the assistant, with the preset's model.
    let conversation: Conversation
    /// The Agent tab: Hermes on the server.
    let agent: Conversation

    init() {
        settings = AppSettings()
        store = ChatStore()
        spend = SpendMeter()
        conversation = Conversation(settings: settings, store: store, spend: spend)
        agent = Conversation(settings: settings, store: store, spend: spend, isAgent: true)
        conversation.otherIsBusy = { [unowned agent] in agent.isBusy }
        agent.otherIsBusy = { [unowned conversation] in conversation.isBusy }
        // Back online: collect a Hermes answer that was still on its way.
        network.pathUpdateHandler = { [weak self] path in
            guard path.status == .satisfied else { return }
            Task { @MainActor in self?.agent.resumePendingRun() }
        }
        network.start(queue: .main)
        agent.openWaitingChat()
    }

    private let network = NWPathMonitor()
}

@main
struct EarshotApp: App {
    @StateObject private var state = AppState()

    var body: some Scene {
        WindowGroup {
            RootView(agent: state.agent)
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

enum AppTab: Hashable {
    case talk, agent
}

/// Two tabs, two separate workflows: Talk asks the assistant, Agent has Hermes do things.
struct RootView: View {
    /// The Talk tab's conversation is the environment's; the Agent tab gets its own.
    @EnvironmentObject var conversation: Conversation
    @EnvironmentObject var spend: SpendMeter
    @EnvironmentObject var settings: AppSettings
    let agent: Conversation
    @Environment(\.scenePhase) private var scenePhase

    @State private var tab: AppTab = .talk
    @State private var leftAt: Date? = .distantPast
    /// Changed to close the Talk tab's sheets when the Action Button brings a screenshot.
    @State private var closeSheets = 0

    var body: some View {
        TabView(selection: $tab) {
            Tab("Talk", systemImage: "waveform", value: AppTab.talk) {
                ConversationTab(closeSheets: closeSheets)
            }
            Tab("Agent", systemImage: "terminal", value: AppTab.agent) {
                ConversationTab(closeSheets: 0)
                    .environmentObject(agent)
            }
        }
        .onChange(of: tab) { old, _ in
            // Only the tab in front listens.
            let left = old == .talk ? conversation : agent
            if left.phase == .listening || left.phase == .transcribing { left.cancelListening(reason: "tab changed") }
        }
        .onChange(of: scenePhase) { _, phase in
            switch phase {
            case .active: arrived()
            case .background:
                leftAt = Date()
                if conversation.phase == .idle && agent.phase == .idle {
                    AudioIO.shared.suspend()
                    // A voice model can hold gigabytes; iOS ends background apps that keep that much.
                    Task { await LocalVoiceEngine.shared.unload() }
                }
            default: break
            }
        }
        .onOpenURL { url in
            // earshot://ask?q=… asks a typed question.
            if let query = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems?.first(where: { $0.name == "q" })?.value,
               !query.isEmpty {
                tab = .talk
                conversation.newChat()
                conversation.submit(query)
            }
        }
    }

    private func arrived() {
        spend.refresh()
        let away = leftAt.map { Date().timeIntervalSince($0) } ?? 0
        leftAt = nil
        // A Hermes answer that arrived while Earshot was away.
        agent.resumePendingRun()
        // Only the Action Button's new screenshot starts listening; opening Earshot yourself waits for a tap.
        guard let image = ClipboardImage.takeNew() else {
            if away > 20 { startFresh(conversation); startFresh(agent) }
            return
        }
        tab = .talk
        if conversation.phase == .idle {
            startFresh(conversation)
            conversation.attach(image, label: "Screenshot")
            closeSheets += 1
            if settings.data.listenOnActionButton { listenSoon() }
        } else {
            conversation.attach(image, label: "Screenshot")
        }
    }

    /// iOS hands silence to recording that starts the moment an app comes to the front; a short
    /// wait avoids it.
    private func listenSoon() {
        Task {
            try? await Task.sleep(nanoseconds: 500_000_000)
            if conversation.phase == .idle { conversation.startListening() }
        }
    }

    /// A chat left a while ago is finished; a question now starts a new one.
    private func startFresh(_ conversation: Conversation) {
        guard conversation.phase == .idle else { return }
        if let chat = conversation.chat, Date().timeIntervalSince(chat.updated) > 300 {
            conversation.newChat()
        }
    }
}

/// One tab's screens: home or the chat, the listening overlay, and its sheets.
struct ConversationTab: View {
    @EnvironmentObject var conversation: Conversation
    let closeSheets: Int

    @State private var showChats = false
    @State private var showSettings = false
    @State private var showModels = false
    @State private var typing = false

    private var listening: Bool { conversation.phase == .listening || conversation.phase == .transcribing }

    var body: some View {
        ZStack {
            Theme.ground.ignoresSafeArea()
            if conversation.chat == nil {
                if conversation.isAgent {
                    AgentHome(openChats: { showChats = true }, openSettings: { showModels = true })
                } else {
                    TalkView(openChats: { showChats = true }, openSettings: { showSettings = true }, openModels: { showModels = true })
                }
            } else {
                ChatView(openChats: { showChats = true }, openModels: { showModels = true }, typing: $typing)
            }
            if listening {
                ListeningView(typeInstead: { typing = true })
                    .transition(.opacity)
            }
        }
        .animation(.easeOut(duration: 0.2), value: conversation.phase)
        .toolbar(listening ? .hidden : .visible, for: .tabBar)
        .sheet(isPresented: $showModels) {
            if conversation.isAgent { AgentSettingsSheet() } else { ModelsSheet() }
        }
        .fullScreenCover(isPresented: $showChats) { ChatsView() }
        .fullScreenCover(isPresented: $showSettings) { SettingsView() }
        .onChange(of: closeSheets) { _, _ in
            showChats = false
            showSettings = false
            showModels = false
            typing = false
        }
    }
}
