import Foundation
import UIKit

enum ChatKind: String, Codable {
    case voice, shot, page
}

struct Turn: Codable, Identifiable, Equatable {
    var id = UUID()
    var isUser: Bool
    var text: String
    var date = Date()
    /// File name of an attached image in the store's image folder.
    var imageFile: String?
    /// "Screenshot", "Photo", "Web page".
    var attachmentLabel: String?
    var modelLabel: String?
    var cost: Double?
    var firstWordSeconds: Double?
    var failed = false
    /// Web pages the answer drew on, when it searched.
    var sources: [Source]?
}

struct Source: Codable, Equatable, Hashable {
    var title: String
    var url: String
}

struct Chat: Codable, Identifiable, Equatable {
    var id = UUID()
    var title: String
    var created = Date()
    var updated = Date()
    var kind: ChatKind = .voice
    var turns: [Turn] = []

    var cost: Double { turns.compactMap(\.cost).reduce(0, +) }
    var lastModel: String? { turns.last(where: { !$0.isUser })?.modelLabel }
}

/// Chats as JSON files, images as JPEG files, both in Application Support.
@MainActor
final class ChatStore: ObservableObject {
    @Published private(set) var chats: [Chat] = []

    private let folder: URL
    private let images: URL

    init() {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        folder = base.appendingPathComponent("Chats", isDirectory: true)
        images = base.appendingPathComponent("Images", isDirectory: true)
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try? FileManager.default.createDirectory(at: images, withIntermediateDirectories: true)
        load()
    }

    func save(_ chat: Chat) {
        var chat = chat
        chat.updated = Date()
        if let index = chats.firstIndex(where: { $0.id == chat.id }) {
            chats[index] = chat
        } else {
            chats.append(chat)
        }
        chats.sort { $0.updated > $1.updated }
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        if let data = try? encoder.encode(chat) {
            try? data.write(to: folder.appendingPathComponent(chat.id.uuidString + ".json"), options: .atomic)
        }
    }

    func delete(_ chat: Chat) {
        chats.removeAll { $0.id == chat.id }
        try? FileManager.default.removeItem(at: folder.appendingPathComponent(chat.id.uuidString + ".json"))
        for file in chat.turns.compactMap(\.imageFile) {
            try? FileManager.default.removeItem(at: images.appendingPathComponent(file))
        }
    }

    func saveImage(_ jpeg: Data) -> String {
        let name = UUID().uuidString + ".jpg"
        try? jpeg.write(to: images.appendingPathComponent(name), options: .atomic)
        return name
    }

    func imageData(_ name: String) -> Data? {
        try? Data(contentsOf: images.appendingPathComponent(name))
    }

    func image(_ name: String) -> UIImage? {
        imageData(name).flatMap(UIImage.init(data:))
    }

    private func load() {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let files = (try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? []
        chats = files.compactMap { url in
            guard url.pathExtension == "json", let data = try? Data(contentsOf: url) else { return nil }
            return try? decoder.decode(Chat.self, from: data)
        }
        .sorted { $0.updated > $1.updated }
    }
}

/// Money spent on replies this calendar month.
@MainActor
final class SpendMeter: ObservableObject {
    @Published private(set) var thisMonth: Double = 0

    private var key: String {
        let parts = Calendar.current.dateComponents([.year, .month], from: Date())
        return "spend.\(parts.year ?? 0).\(parts.month ?? 0)"
    }

    init() {
        thisMonth = UserDefaults.standard.double(forKey: key)
    }

    func add(_ dollars: Double) {
        thisMonth = UserDefaults.standard.double(forKey: key) + dollars
        UserDefaults.standard.set(thisMonth, forKey: key)
    }

    func refresh() {
        thisMonth = UserDefaults.standard.double(forKey: key)
    }
}

/// A picture waiting to go with the next question.
struct Attachment: Equatable {
    var jpeg: Data
    var label: String
    var date = Date()

    var image: UIImage? { UIImage(data: jpeg) }

    /// Scaled so the long side is at most 1568 points, the most a model reads at full detail.
    static func make(_ image: UIImage, label: String) -> Attachment? {
        let longest: CGFloat = 1568
        let size = image.size
        let scale = min(1, longest / max(size.width, size.height))
        let target = CGSize(width: (size.width * scale).rounded(), height: (size.height * scale).rounded())
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        let resized = UIGraphicsImageRenderer(size: target, format: format).image { _ in
            image.draw(in: CGRect(origin: .zero, size: target))
        }
        guard let jpeg = resized.jpegData(compressionQuality: 0.8) else { return nil }
        return Attachment(jpeg: jpeg, label: label)
    }
}
