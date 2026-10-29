import Foundation

/// Something the store can download: a folder of its own and the Hugging Face files that fill it.
public protocol DownloadableModel: Sendable {
    var folderName: String { get }
    var displayName: String { get }
    var sources: [LocalModelSource] { get }
}

extension LocalVoiceModel: DownloadableModel {
    public var folderName: String { rawValue }
    public var displayName: String { info.name }
    public var sources: [LocalModelSource] { info.sources }
}

/// Downloads models from Hugging Face into one folder each and remembers which are complete.
public final class LocalModelStore: @unchecked Sendable {
    public static let shared = LocalModelStore()

    /// Where the models live. On the phone this is Application Support, which iOS keeps and does
    /// not back up to iCloud.
    public let root: URL

    public init(root: URL? = nil) {
        if let root {
            self.root = root
        } else {
            let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            self.root = support.appendingPathComponent("LocalVoices", isDirectory: true)
        }
    }

    public func folder<Model: DownloadableModel>(_ model: Model) -> URL {
        root.appendingPathComponent(model.folderName, isDirectory: true)
    }

    private func marker<Model: DownloadableModel>(_ model: Model) -> URL {
        folder(model).appendingPathComponent(".complete")
    }

    public func isDownloaded<Model: DownloadableModel>(_ model: Model) -> Bool {
        FileManager.default.fileExists(atPath: marker(model).path)
    }

    /// Bytes on disk, for the storage line in Settings.
    public func bytesOnDisk<Model: DownloadableModel>(_ model: Model) -> Int64 {
        guard let walker = FileManager.default.enumerator(at: folder(model), includingPropertiesForKeys: [.fileSizeKey]) else { return 0 }
        var total: Int64 = 0
        for case let url as URL in walker {
            total += Int64((try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        }
        return total
    }

    public func delete<Model: DownloadableModel>(_ model: Model) throws {
        if FileManager.default.fileExists(atPath: folder(model).path) {
            try FileManager.default.removeItem(at: folder(model))
        }
    }

    public struct RemoteFile: Sendable {
        let source: LocalModelSource
        let path: String
        let size: Int64
    }

    /// The files a model needs, with their sizes.
    public func remoteFiles<Model: DownloadableModel>(_ model: Model) async throws -> [RemoteFile] {
        var result: [RemoteFile] = []
        for source in model.sources {
            let url = URL(string: "https://huggingface.co/api/models/\(source.repo)/tree/\(source.revision)?recursive=true")!
            let (data, response) = try await URLSession.shared.data(from: url)
            guard (response as? HTTPURLResponse)?.statusCode == 200 else {
                throw LocalVoiceError("Hugging Face did not list \(source.repo).")
            }
            let entries = (try JSONSerialization.jsonObject(with: data) as? [[String: Any]]) ?? []
            for entry in entries where entry["type"] as? String == "file" {
                guard let path = entry["path"] as? String else { continue }
                if path.hasPrefix(".") || path.hasSuffix(".md") || path.hasSuffix(".png") { continue }
                if !source.files.isEmpty && !source.files.contains(path) { continue }
                let lfs = entry["lfs"] as? [String: Any]
                let size = (lfs?["size"] as? NSNumber)?.int64Value ?? (entry["size"] as? NSNumber)?.int64Value ?? 0
                result.append(RemoteFile(source: source, path: path, size: size))
            }
        }
        guard !result.isEmpty else { throw LocalVoiceError("No files were found for \(model.displayName).") }
        return result
    }

    /// Fetches every missing file. `progress` gets bytes done and bytes in all.
    public func download<Model: DownloadableModel>(_ model: Model, progress: @escaping @Sendable (Int64, Int64) -> Void) async throws {
        if isDownloaded(model) { return }
        let files = try await remoteFiles(model)
        let total = files.reduce(0) { $0 + $1.size }
        var done: Int64 = 0
        let base = folder(model)
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        for file in files {
            try Task.checkCancellation()
            // A source's subfolder is dropped from the path, so the files sit where the loaders expect.
            let relative = localPath(file)
            let target = base.appendingPathComponent(relative)
            if let size = try? target.resourceValues(forKeys: [.fileSizeKey]).fileSize, Int64(size) == file.size {
                done += file.size
                progress(done, total)
                continue
            }
            try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            let remote = URL(string: "https://huggingface.co/\(file.source.repo)/resolve/\(file.source.revision)/\(file.path)")!
            let start = done
            let temporary = try await Downloader.fetch(remote) { written in
                progress(start + written, total)
            }
            if FileManager.default.fileExists(atPath: target.path) {
                try FileManager.default.removeItem(at: target)
            }
            try FileManager.default.moveItem(at: temporary, to: target)
            done += file.size
            progress(done, total)
        }
        try Data().write(to: marker(model))
    }

    private func localPath(_ file: RemoteFile) -> String {
        var path = file.path
        // Raon's checkpoint folder is flattened so that config.yaml sits at the top.
        if file.source.repo == "drbaph/Raon-OpenTTS-comfyui", let slash = path.firstIndex(of: "/") {
            path = String(path[path.index(after: slash)...])
        }
        return file.source.folder.isEmpty ? path : file.source.folder + "/" + path
    }
}

public struct LocalVoiceError: LocalizedError, Sendable {
    public let message: String
    public init(_ message: String) { self.message = message }
    public var errorDescription: String? { message }
}

/// One file at a time through a URLSession download task, with progress.
private final class Downloader: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let onProgress: (Int64) -> Void
    private var continuation: CheckedContinuation<URL, Error>?

    private init(onProgress: @escaping (Int64) -> Void) {
        self.onProgress = onProgress
    }

    static func fetch(_ url: URL, progress: @escaping (Int64) -> Void) async throws -> URL {
        let downloader = Downloader(onProgress: progress)
        let session = URLSession(configuration: .default, delegate: downloader, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }
        let task = session.downloadTask(with: url)
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                downloader.continuation = continuation
                task.resume()
            }
        } onCancel: {
            task.cancel()
        }
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64,
                    totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        onProgress(totalBytesWritten)
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        let status = (downloadTask.response as? HTTPURLResponse)?.statusCode ?? 0
        guard status == 200 else {
            continuation?.resume(throwing: LocalVoiceError("Download failed with HTTP \(status)."))
            continuation = nil
            return
        }
        // The system deletes `location` when this returns, so it is moved first.
        let kept = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        do {
            try FileManager.default.moveItem(at: location, to: kept)
            continuation?.resume(returning: kept)
        } catch {
            continuation?.resume(throwing: error)
        }
        continuation = nil
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error {
            continuation?.resume(throwing: error)
            continuation = nil
        }
    }
}
