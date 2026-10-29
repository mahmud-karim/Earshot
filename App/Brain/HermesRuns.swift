import Foundation

/// Hermes stopped being reachable before its run finished. The run carries on on the server; its answer
/// is collected later, so this is not shown as a failed reply.
struct HermesOffline: LocalizedError {
    var errorDescription: String? { "Lost touch with Hermes. It's still working; the answer will appear when Earshot can reach it again." }
}

/// Hermes through its Runs API. The job runs on the server whether or not the phone is connected, so an
/// answer survives the app being suspended, closed or offline. The phone follows the run's events; when
/// it loses them it reattaches from the last one it saw, and once the server no longer keeps the events
/// it reads the run's status, then the session's history.
struct HermesRunBrain: Brain {
    let baseURL: URL
    let apiKey: String
    /// Hermes keeps the conversation, tools and all, under this id: one per Earshot chat.
    let sessionID: String
    /// Follow a run started earlier instead of starting one.
    var resumeRunID: String?
    /// Makes starting the run safe to retry: the same question never starts two runs.
    var idempotencyKey: String
    /// When the question was asked, to find its answer in the session's history.
    var askedAt: Date

    func stream(_ request: BrainRequest) -> AsyncThrowingStream<BrainEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await run(request, continuation)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func run(_ request: BrainRequest, _ continuation: AsyncThrowingStream<BrainEvent, Error>.Continuation) async throws {
        let runID: String
        if let resumeRunID {
            runID = resumeRunID
        } else {
            runID = try await start(request)
        }
        continuation.yield(.runStarted(runID))
        var follower = Follower(runID: runID)
        var misses = 0
        while true {
            try Task.checkCancellation()
            do {
                if try await follower.follow(self, continuation) { return }
                misses = 0
            } catch let error as BrainError {
                throw error
            } catch {
                try Task.checkCancellation()
                misses += 1
            }
            // The events stream closed early or couldn't be reached: ask how the run is doing.
            switch await status(runID) {
            case .running:
                break
            case .finished(let output, let usage):
                follower.finish(with: output, usage: usage, continuation)
                return
            case .failed(let message):
                throw BrainError(message: message)
            case .gone:
                if let answer = await sessionAnswer() {
                    follower.finish(with: answer, usage: nil, continuation)
                    return
                }
                throw BrainError(message: "Hermes no longer has this run, and its answer wasn't saved in the session.")
            case .unreachable:
                misses += 1
            }
            // Offline for about two minutes: give up for now. The chat keeps the run, to collect later.
            if misses >= 24 { throw HermesOffline() }
            try await Task.sleep(nanoseconds: UInt64(min(misses, 5)) * 1_000_000_000)
        }
    }

    /// Where the run's events have got to, so a reconnection carries on rather than repeating.
    private struct Follower {
        let runID: String
        var lastSeq: Int?
        var spoken = ""

        /// Reads the run's events until it ends (true) or the stream drops (false, or a thrown error).
        mutating func follow(_ brain: HermesRunBrain, _ continuation: AsyncThrowingStream<BrainEvent, Error>.Continuation) async throws -> Bool {
            var request = URLRequest(url: brain.baseURL.appendingPathComponent("runs/\(runID)/events"))
            request.timeoutInterval = 60
            brain.authorize(&request)
            if let lastSeq { request.setValue(String(lastSeq), forHTTPHeaderField: "Last-Event-ID") }
            let (bytes, response) = try await URLSession.shared.bytes(for: request)
            guard (response as? HTTPURLResponse)?.statusCode == 200 else { return false }
            for try await line in bytes.lines {
                guard let event = SSE.json(line), let name = event["event"] as? String else { continue }
                if let seq = event["seq"] as? Int {
                    if let lastSeq, seq <= lastSeq { continue }
                    lastSeq = seq
                }
                switch name {
                case "message.delta":
                    if let delta = event["delta"] as? String, !delta.isEmpty {
                        spoken += delta
                        continuation.yield(.text(delta))
                    }
                case "tool.started":
                    if let tool = event["tool"] as? String {
                        continuation.yield(.working(tool: tool, label: event["preview"] as? String ?? tool))
                    }
                case "approval.request":
                    if let approval = HermesApproval(event) { continuation.yield(.approval(approval)) }
                case "run.completed":
                    finish(with: event["output"] as? String ?? "", usage: event["usage"] as? [String: Any], continuation)
                    return true
                case "run.failed":
                    throw BrainError(message: "Hermes failed: " + (event["error"] as? String ?? "no reason given"))
                case "run.cancelled", "run.interrupted":
                    throw BrainError(message: name == "run.cancelled" ? "Hermes's run was stopped before it finished."
                                                                      : "Hermes's run was interrupted before it finished.")
                default:
                    break
                }
            }
            return false
        }

        /// The final answer, less whatever of it already came as deltas.
        mutating func finish(with output: String, usage: [String: Any]?, _ continuation: AsyncThrowingStream<BrainEvent, Error>.Continuation) {
            if !output.isEmpty {
                if spoken.isEmpty {
                    continuation.yield(.text(output))
                } else if output.hasPrefix(spoken), output.count > spoken.count {
                    continuation.yield(.text(String(output.dropFirst(spoken.count))))
                }
                spoken = output
            }
            if let usage {
                continuation.yield(.usage(input: usage["input_tokens"] as? Int ?? 0, output: usage["output_tokens"] as? Int ?? 0))
            }
        }
    }

    fileprivate func authorize(_ request: inout URLRequest) {
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "authorization")
    }

    private func start(_ request: BrainRequest) async throws -> String {
        var urlRequest = URLRequest(url: baseURL.appendingPathComponent("runs"))
        urlRequest.httpMethod = "POST"
        urlRequest.timeoutInterval = 30
        authorize(&urlRequest)
        urlRequest.setValue("application/json", forHTTPHeaderField: "content-type")
        urlRequest.setValue(idempotencyKey, forHTTPHeaderField: "Idempotency-Key")
        let question = request.messages.last(where: { $0.role == .user })?.text ?? ""
        urlRequest.httpBody = try JSONSerialization.data(withJSONObject: [
            "input": question, "session_id": sessionID, "instructions": request.system, "model": request.model
        ])
        let (data, response) = try await URLSession.shared.data(for: urlRequest)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard status == 200 || status == 202,
              let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              let runID = object["run_id"] as? String else {
            throw BrainError(message: SSE.readableError(status: status, body: String(decoding: data, as: UTF8.self)))
        }
        return runID
    }

    enum RunStatus {
        case running
        case finished(String, [String: Any]?)
        case failed(String)
        /// The server no longer keeps it: about an hour after it ended.
        case gone
        case unreachable
    }

    func status(_ runID: String) async -> RunStatus {
        var request = URLRequest(url: baseURL.appendingPathComponent("runs/\(runID)"))
        request.timeoutInterval = 15
        authorize(&request)
        guard let (data, response) = try? await URLSession.shared.data(for: request) else { return .unreachable }
        let code = (response as? HTTPURLResponse)?.statusCode ?? 0
        if code == 404 { return .gone }
        guard code == 200, let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else { return .unreachable }
        switch object["status"] as? String ?? "" {
        case "completed":
            return .finished(object["output"] as? String ?? "", object["usage"] as? [String: Any])
        case "failed":
            return .failed("Hermes failed: " + (object["error"] as? String ?? "no reason given"))
        case "cancelled":
            return .failed("Hermes's run was stopped before it finished.")
        case "interrupted":
            return .failed("Hermes's run was interrupted (the server restarted) before it finished.")
        default:
            return .running
        }
    }

    /// The last thing Hermes said in the session after the question was asked.
    private func sessionAnswer() async -> String? {
        let url = baseURL.deletingLastPathComponent().appendingPathComponent("api/sessions/\(sessionID)/messages")
        var request = URLRequest(url: url)
        request.timeoutInterval = 15
        authorize(&request)
        guard let (data, _) = try? await URLSession.shared.data(for: request),
              let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              let messages = object["data"] as? [[String: Any]] else { return nil }
        let since = askedAt.timeIntervalSince1970 - 5
        return messages.last { message in
            message["role"] as? String == "assistant"
                && !((message["content"] as? String ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                && (message["timestamp"] as? Double ?? 0) >= since
        }?["content"] as? String
    }
}

extension Hermes {
    /// Stops a run on the server: the user pressed stop.
    static func stop(runID: String, baseURL: URL, key: String) async {
        var request = URLRequest(url: baseURL.appendingPathComponent("runs/\(runID)/stop"))
        request.httpMethod = "POST"
        request.timeoutInterval = 15
        request.setValue("Bearer \(key)", forHTTPHeaderField: "authorization")
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data("{}".utf8)
        _ = try? await URLSession.shared.data(for: request)
    }
}
