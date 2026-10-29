import SwiftUI

/// Past chats, newest first, grouped by day.
struct ChatsView: View {
    @EnvironmentObject var store: ChatStore
    @EnvironmentObject var conversation: Conversation
    @EnvironmentObject var spend: SpendMeter
    @EnvironmentObject var settings: AppSettings
    @Environment(\.dismiss) private var dismiss
    @State private var search = ""

    var body: some View {
        VStack(spacing: 16) {
            PageHeader(title: conversation.isAgent ? "Agent chats" : "Chats", back: { dismiss() }) {
                Button {
                    conversation.newChat()
                    dismiss()
                } label: {
                    Image(systemName: "plus").font(.system(size: 18, weight: .semibold)).foregroundStyle(Theme.onClay)
                        .frame(width: 44, height: 44).background(Circle().fill(Theme.clay))
                }
                .buttonStyle(.plain)
                .accessibilityLabel("New chat")
            }

            HStack(spacing: 10) {
                Image(systemName: "magnifyingglass").font(.system(size: 15)).foregroundStyle(Theme.muted)
                TextField("", text: $search, prompt: Text("Search chats").foregroundStyle(Theme.muted))
                    .font(.system(size: 16)).foregroundStyle(Theme.text)
            }
            .padding(.horizontal, 14).frame(height: 44)
            .background(RoundedRectangle(cornerRadius: 12).fill(Theme.surface))

            if filtered.isEmpty {
                Spacer()
                Text(search.isEmpty ? "Your chats will show up here." : "No chats match.")
                    .font(.system(size: 15)).foregroundStyle(Theme.muted)
                Spacer()
            } else {
                List {
                    ForEach(sections) { section in
                        Section {
                            ForEach(section.chats) { chat in
                                ChatRow(chat: chat)
                                    .contentShape(Rectangle())
                                    .onTapGesture {
                                        conversation.open(chat)
                                        dismiss()
                                    }
                                    .listRowBackground(Theme.ground)
                                    .listRowInsets(EdgeInsets(top: 0, leading: 0, bottom: 0, trailing: 0))
                                    .listRowSeparatorTint(Theme.hairline)
                                    .swipeActions {
                                        Button(role: .destructive) {
                                            if conversation.chat?.id == chat.id { conversation.newChat() }
                                            store.delete(chat)
                                        } label: {
                                            Label("Delete", systemImage: "trash")
                                        }
                                    }
                            }
                        } header: {
                            Text(section.label).font(.system(size: 13)).foregroundStyle(Theme.muted).textCase(nil)
                                .padding(.leading, -16)
                        }
                    }
                }
                .listStyle(.plain)
                .scrollContentBackground(.hidden)
            }

            SpendCard(title: "\(Date().formatted(.dateTime.month(.wide))) API spend", spent: spend.thisMonth,
                      budget: settings.data.monthlyBudget)
        }
        .padding(.horizontal, 20)
        .padding(.top, 8)
        .padding(.bottom, 12)
        .background(Theme.ground.ignoresSafeArea())
        .onAppear { spend.refresh() }
    }

    private var filtered: [Chat] {
        let query = search.trimmingCharacters(in: .whitespaces)
        // Each tab lists its own chats: Hermes in Agent, everything else in Talk.
        let mine = store.chats.filter { ($0.kind == .agent) == conversation.isAgent }
        guard !query.isEmpty else { return mine }
        return mine.filter { chat in
            chat.title.localizedCaseInsensitiveContains(query)
                || chat.turns.contains { $0.text.localizedCaseInsensitiveContains(query) }
        }
    }

    private var sections: [ChatSection] {
        let calendar = Calendar.current
        var today: [Chat] = [], yesterday: [Chat] = [], earlier: [Chat] = []
        for chat in filtered {
            if calendar.isDateInToday(chat.updated) {
                today.append(chat)
            } else if calendar.isDateInYesterday(chat.updated) {
                yesterday.append(chat)
            } else {
                earlier.append(chat)
            }
        }
        return [ChatSection(label: "Today", chats: today), ChatSection(label: "Yesterday", chats: yesterday),
                ChatSection(label: "Earlier", chats: earlier)].filter { !$0.chats.isEmpty }
    }
}

struct ChatSection: Identifiable {
    let label: String
    let chats: [Chat]
    var id: String { label }
}

struct ChatRow: View {
    let chat: Chat

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: icon).font(.system(size: 16)).foregroundStyle(Theme.secondary)
                .frame(width: 36, height: 36).background(RoundedRectangle(cornerRadius: 10).fill(Theme.surface))
            VStack(alignment: .leading, spacing: 3) {
                Text(chat.title).font(.system(size: 16)).foregroundStyle(Theme.text).lineLimit(1)
                Text(meta).font(.system(size: 13)).foregroundStyle(Theme.muted).lineLimit(1)
            }
            Spacer(minLength: 8)
            Text(when).font(.system(size: 12)).foregroundStyle(Theme.muted)
        }
        .padding(.vertical, 10)
    }

    private var icon: String {
        switch chat.kind {
        case .shot: return "photo"
        case .page: return "globe"
        case .voice: return "mic"
        case .agent: return "terminal"
        }
    }

    private var meta: String {
        var parts: [String] = []
        if chat.kind == .shot { parts.append("Screenshot") }
        if chat.kind == .page { parts.append("Web page") }
        if let model = chat.lastModel { parts.append(model) }
        if chat.cost > 0 { parts.append(Money.text(chat.cost)) }
        return parts.isEmpty ? "\(chat.turns.count) messages" : parts.joined(separator: " · ")
    }

    private var when: String {
        let calendar = Calendar.current
        if calendar.isDateInToday(chat.updated) || calendar.isDateInYesterday(chat.updated) {
            return chat.updated.formatted(date: .omitted, time: .shortened)
        }
        return chat.updated.formatted(.dateTime.month(.abbreviated).day())
    }
}
