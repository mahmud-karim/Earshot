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
}

enum BrainEvent {
    case text(String)
    case usage(input: Int, output: Int)
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

        let messages: [[String: Any]] = request.messages.map { message in
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
            "messages": messages,
            "output_config": ["effort": request.effort.rawValue]
        ]
        if useFallback {
            body["fallbacks"] = "default"
        }
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

        var input = 0
        var output = 0
        for try await line in bytes.lines {
            guard let event = SSE.json(line) else { continue }
            switch event["type"] as? String {
            case "message_start":
                if let usage = (event["message"] as? [String: Any])?["usage"] as? [String: Any] {
                    input = (usage["input_tokens"] as? Int ?? 0)
                        + (usage["cache_read_input_tokens"] as? Int ?? 0)
                        + (usage["cache_creation_input_tokens"] as? Int ?? 0)
                }
            case "content_block_delta":
                if let delta = event["delta"] as? [String: Any],
                   delta["type"] as? String == "text_delta",
                   let text = delta["text"] as? String {
                    continuation.yield(.text(text))
                }
            case "message_delta":
                if let usage = event["usage"] as? [String: Any], let tokens = usage["output_tokens"] as? Int {
                    output = tokens
                }
                if let delta = event["delta"] as? [String: Any], delta["stop_reason"] as? String == "refusal" {
                    continuation.yield(.text("\n\nThe model declined to answer this one."))
                }
            case "error":
                let message = ((event["error"] as? [String: Any])?["message"] as? String) ?? "The stream failed."
                throw BrainError(message: message)
            default:
                break
            }
        }
        continuation.yield(.usage(input: input, output: output))
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
