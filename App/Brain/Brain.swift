import Foundation

// The part that answers. Two wire formats cover every provider: Anthropic's Messages API, and the
// OpenAI-compatible chat completions API (OpenAI, OpenRouter, Ollama and LM Studio on the PC).

struct BrainMessage {
    enum Role: String { case user, assistant }
    var role: Role
    var text: String
    /// JPEG bytes of an attached image, user turns only.
    var image: Data?
}

struct BrainRequest {
    var model: String
    var system: String
    var messages: [BrainMessage]
    var effort: Effort
    /// Lets the model search the web and read pages. Claude only; other brains ignore it.
    var webAccess = false
}

enum BrainEvent {
    case text(String)
    case usage(input: Int, output: Int)
    /// The model started a web search, or opened a page when `webSearch` is false.
    case searching(webSearch: Bool)
    /// A page the answer is drawn from.
    case source(title: String, url: String)
}

protocol Brain {
    func stream(_ request: BrainRequest) -> AsyncThrowingStream<BrainEvent, Error>
}

struct BrainError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

// MARK: - Anthropic

struct AnthropicBrain: Brain {
    let apiKey: String

    func stream(_ request: BrainRequest) -> AsyncThrowingStream<BrainEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    do {
                        try await run(request, fallbacks: true, continuation)
                    } catch is FallbackRejected {
                        try await run(request, fallbacks: false, continuation)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private struct FallbackRejected: Error {}

    /// Sonnet 5.5 and Opus 5.5 can hand a refused request to another model inside the same call.
    private func supportsFallback(_ model: String) -> Bool {
        model == "claude-sonnet-5-5" || model == "claude-opus-5-5"
    }

    private func run(_ request: BrainRequest, fallbacks: Bool, _ continuation: AsyncThrowingStream<BrainEvent, Error>.Continuation) async throws {
        var urlRequest = URLRequest(url: URL(string: "https://api.anthropic.com/v1/messages")!)
        urlRequest.httpMethod = "POST"
        urlRequest.timeoutInterval = 300
        urlRequest.setValue(apiKey, forHTTPHeaderField: "x-api-key")
        urlRequest.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")
        urlRequest.setValue("application/json", forHTTPHeaderField: "content-type")
        let useFallback = fallbacks && supportsFallback(request.model)
        if useFallback {
            urlRequest.setValue("server-side-fallback-2026-07-01", forHTTPHeaderField: "anthropic-beta")
        }

        var messages: [[String: Any]] = request.messages.map { message in
            var content: [[String: Any]] = []
            if let image = message.image {
                content.append([
                    "type": "image",
                    "source": ["type": "base64", "media_type": "image/jpeg", "data": image.base64EncodedString()]
                ])
            }
            content.append(["type": "text", "text": message.text.isEmpty ? "(no text)" : message.text])
            return ["role": message.role.rawValue, "content": content]
        }
        var body: [String: Any] = [
            "model": request.model,
            "max_tokens": 32000,
            "stream": true,
            "system": request.system,
            "output_config": ["effort": request.effort.rawValue]
        ]
        if request.webAccess {
            // The basic versions: the newer ones filter results by running code first, which is
            // slower to the first spoken word.
            body["tools"] = [
                ["type": "web_search_20250305", "name": "web_search", "max_uses": 5],
                ["type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 3, "max_content_tokens": 20000]
            ]
        }
        if useFallback {
            body["fallbacks"] = "default"
        }

        var input = 0
        var output = 0
        var sources = Set<String>()
        // A turn with many searches can pause; sending the paused answer back resumes it.
        for _ in 0..<4 {
            body["messages"] = messages
            urlRequest.httpBody = try JSONSerialization.data(withJSONObject: body)
            let (bytes, response) = try await URLSession.shared.bytes(for: urlRequest)
            let status = (response as? HTTPURLResponse)?.statusCode ?? 0
            if status != 200 {
                let text = try await SSE.collect(bytes)
                if status == 400 && useFallback && text.contains("fallback") {
                    throw FallbackRejected()
                }
                throw BrainError(message: SSE.readableError(status: status, body: text))
            }

            var blocks = ContentBlocks()
            var stopReason: String?
            for try await line in bytes.lines {
                guard let event = SSE.json(line) else { continue }
                blocks.apply(event)
                switch event["type"] as? String {
                case "message_start":
                    if let usage = (event["message"] as? [String: Any])?["usage"] as? [String: Any] {
                        input += (usage["input_tokens"] as? Int ?? 0)
                            + (usage["cache_read_input_tokens"] as? Int ?? 0)
                            + (usage["cache_creation_input_tokens"] as? Int ?? 0)
                    }
                case "content_block_start":
                    guard let block = event["content_block"] as? [String: Any] else { break }
                    switch block["type"] as? String {
                    case "server_tool_use":
                        continuation.yield(.searching(webSearch: block["name"] as? String == "web_search"))
                    case "web_fetch_tool_result":
                        // A page it opened. Search results only count once the answer cites them.
                        if let result = block["content"] as? [String: Any], let url = result["url"] as? String,
                           sources.insert(url).inserted {
                            let title = ((result["content"] as? [String: Any])?["title"] as? String) ?? ""
                            continuation.yield(.source(title: title, url: url))
                        }
                    default:
                        break
                    }
                case "content_block_delta":
                    guard let delta = event["delta"] as? [String: Any] else { break }
                    switch delta["type"] as? String {
                    case "text_delta":
                        if let text = delta["text"] as? String { continuation.yield(.text(text)) }
                    case "citations_delta":
                        if let citation = delta["citation"] as? [String: Any], let url = citation["url"] as? String,
                           sources.insert(url).inserted {
                            continuation.yield(.source(title: citation["title"] as? String ?? "", url: url))
                        }
                    default:
                        break
                    }
                case "message_delta":
                    if let usage = event["usage"] as? [String: Any], let tokens = usage["output_tokens"] as? Int {
                        output += tokens
                    }
                    if let delta = event["delta"] as? [String: Any], let reason = delta["stop_reason"] as? String {
                        stopReason = reason
                        if reason == "refusal" {
                            continuation.yield(.text("\n\nThe model declined to answer this one."))
                        }
                    }
                case "error":
                    let message = ((event["error"] as? [String: Any])?["message"] as? String) ?? "The stream failed."
                    throw BrainError(message: message)
                default:
                    break
                }
            }
            guard stopReason == "pause_turn" else { break }
            messages.append(["role": "assistant", "content": blocks.content])
        }
        continuation.yield(.usage(input: input, output: output))
    }
}

/// Rebuilds the answer's content blocks from the stream, so a paused turn can be sent back as it was.
private struct ContentBlocks {
    private var blocks: [Int: [String: Any]] = [:]
    private var partialJSON: [Int: String] = [:]

    var content: [[String: Any]] {
        blocks.keys.sorted().compactMap { blocks[$0] }.filter { $0["type"] as? String != "fallback" }
    }

    mutating func apply(_ event: [String: Any]) {
        guard let index = event["index"] as? Int else { return }
        switch event["type"] as? String {
        case "content_block_start":
            blocks[index] = event["content_block"] as? [String: Any]
        case "content_block_delta":
            guard var block = blocks[index], let delta = event["delta"] as? [String: Any] else { return }
            switch delta["type"] as? String {
            case "text_delta":
                block["text"] = (block["text"] as? String ?? "") + (delta["text"] as? String ?? "")
            case "thinking_delta":
                block["thinking"] = (block["thinking"] as? String ?? "") + (delta["thinking"] as? String ?? "")
            case "signature_delta":
                block["signature"] = delta["signature"]
            case "citations_delta":
                if let citation = delta["citation"] {
                    block["citations"] = (block["citations"] as? [Any] ?? []) + [citation]
                }
            case "input_json_delta":
                partialJSON[index, default: ""] += delta["partial_json"] as? String ?? ""
            default:
                break
            }
            blocks[index] = block
        case "content_block_stop":
            if let json = partialJSON.removeValue(forKey: index), var block = blocks[index] {
                let data = json.data(using: .utf8) ?? Data()
                block["input"] = (try? JSONSerialization.jsonObject(with: data)) ?? [String: Any]()
                blocks[index] = block
            }
        default:
            break
        }
    }
}

// MARK: - OpenAI-compatible

struct OpenAICompatibleBrain: Brain {
    let baseURL: URL
    let apiKey: String?
    var extraHeaders: [String: String] = [:]

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
        var urlRequest = URLRequest(url: baseURL.appendingPathComponent("chat/completions"))
        urlRequest.httpMethod = "POST"
        urlRequest.timeoutInterval = 300
        urlRequest.setValue("application/json", forHTTPHeaderField: "content-type")
        if let apiKey {
            urlRequest.setValue("Bearer \(apiKey)", forHTTPHeaderField: "authorization")
        }
        for (name, value) in extraHeaders {
            urlRequest.setValue(value, forHTTPHeaderField: name)
        }

        var messages: [[String: Any]] = [["role": "system", "content": request.system]]
        for message in request.messages {
            if let image = message.image {
                messages.append(["role": message.role.rawValue, "content": [
                    ["type": "text", "text": message.text],
                    ["type": "image_url", "image_url": ["url": "data:image/jpeg;base64," + image.base64EncodedString()]]
                ]])
            } else {
                messages.append(["role": message.role.rawValue, "content": message.text])
            }
        }
        let body: [String: Any] = [
            "model": request.model,
            "stream": true,
            "stream_options": ["include_usage": true],
            "messages": messages
        ]
        urlRequest.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (bytes, response) = try await URLSession.shared.bytes(for: urlRequest)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        if status != 200 {
            let text = try await SSE.collect(bytes)
            throw BrainError(message: SSE.readableError(status: status, body: text))
        }
        for try await line in bytes.lines {
            if line.hasPrefix("data: [DONE]") { break }
            guard let event = SSE.json(line) else { continue }
            if let error = event["error"] as? [String: Any] {
                throw BrainError(message: error["message"] as? String ?? "The stream failed.")
            }
            if let choices = event["choices"] as? [[String: Any]],
               let delta = choices.first?["delta"] as? [String: Any],
               let text = delta["content"] as? String, !text.isEmpty {
                continuation.yield(.text(text))
            }
            if let usage = event["usage"] as? [String: Any] {
                continuation.yield(.usage(input: usage["prompt_tokens"] as? Int ?? 0,
                                          output: usage["completion_tokens"] as? Int ?? 0))
            }
        }
    }
}

// MARK: - Server-sent events

enum SSE {
    /// The JSON object of a `data:` line.
    static func json(_ line: String) -> [String: Any]? {
        guard line.hasPrefix("data:") else { return nil }
        let payload = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
        guard let data = payload.data(using: .utf8) else { return nil }
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }

    static func collect(_ bytes: URLSession.AsyncBytes) async throws -> String {
        var data = Data()
        for try await byte in bytes {
            data.append(byte)
            if data.count > 64_000 { break }
        }
        return String(decoding: data, as: UTF8.self)
    }

    static func readableError(status: Int, body: String) -> String {
        var detail = body
        if let data = body.data(using: .utf8),
           let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] {
            if let error = object["error"] as? [String: Any], let message = error["message"] as? String {
                detail = message
            } else if let message = object["error"] as? String {
                detail = message
            } else if let message = object["detail"] as? String {
                detail = message
            }
        }
        switch status {
        case 401: return "The API key was not accepted. Check it in Settings. (\(detail))"
        case 429: return "Rate limited or out of credit. (\(detail))"
        case 529, 503: return "The model is overloaded right now. Try again in a moment."
        default: return "Error \(status): \(detail.prefix(300))"
        }
    }
}
