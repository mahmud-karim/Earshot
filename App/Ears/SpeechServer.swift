import SwiftUI

/// The speech-to-text server on the PC (F:\AI\STT-Server): Qwen3-ASR on vLLM behind an API-key gateway
/// that speaks OpenAI's transcription API. Its health check sits next to /v1 and needs no key.
enum SpeechServer {
    /// What a failed answer from the server means, in a sentence.
    static func message(status: Int, body: String) -> String {
        let detail = SSE.readableError(status: status, body: body)
        switch status {
        case 401: return "The speech server didn't accept the key. Check it in Models › Ears, or make a new one on your PC."
        case 429: return "The speech server allows 60 requests a minute per key. Wait a moment and try again."
        case 503: return "The speech server is up but its model isn't running. On your PC, run start.ps1 in F:\\AI\\STT-Server."
        case 400: return "The speech server couldn't read the audio. (\(detail))"
        case 0: return "The speech server didn't answer. Check that the PC is on and the address is right."
        default: return "The speech server failed. \(detail)"
        }
    }

    /// Checks the server step by step: it answers, its model runs, and the key is accepted.
    static func test(baseURL: URL, key: String?) async -> (ok: Bool, text: String) {
        let health = baseURL.deletingLastPathComponent().appendingPathComponent("health")
        do {
            var request = URLRequest(url: health)
            request.timeoutInterval = 10
            let (data, response) = try await URLSession.shared.data(for: request)
            let status = (response as? HTTPURLResponse)?.statusCode ?? 0
            guard status == 200 else {
                return (false, status == 503 ? message(status: 503, body: "")
                        : "The server answered the health check with \(status). Check the address.")
            }
            let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
            guard object?["status"] as? String == "ok" else {
                return (false, message(status: 503, body: ""))
            }
        } catch {
            return (false, "The speech server didn't answer: \(error.localizedDescription) Check that the PC is on and the address is right.")
        }
        guard let key else {
            return (false, "The server is running. Add its key to finish setting it up.")
        }
        do {
            var request = URLRequest(url: baseURL.appendingPathComponent("models"))
            request.timeoutInterval = 10
            request.setValue("Bearer \(key)", forHTTPHeaderField: "authorization")
            let (data, response) = try await URLSession.shared.data(for: request)
            let status = (response as? HTTPURLResponse)?.statusCode ?? 0
            guard status == 200 else {
                return (false, message(status: status, body: String(decoding: data, as: UTF8.self)))
            }
            let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
            let models = (object?["data"] as? [[String: Any]])?.compactMap { $0["id"] as? String } ?? []
            return (true, "Connected. The key works" + (models.isEmpty ? "." : " and the server runs \(models.joined(separator: ", "))."))
        } catch {
            return (false, "The speech server didn't answer: \(error.localizedDescription)")
        }
    }
}

/// Address, key and a connection test for the speech server.
struct SpeechServerPage: View {
    @EnvironmentObject var settings: AppSettings
    @Environment(\.dismiss) private var dismiss
    @State private var result: (ok: Bool, text: String)?
    @State private var testing = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                PageHeader(title: "Speech server", back: { dismiss() }) { EmptyView() }
                Text("Qwen3-ASR on your PC's graphics card, reachable from anywhere through Tailscale. It needs the PC on and the server started, and an API key made on the PC.")
                    .font(.system(size: 14)).foregroundStyle(Theme.muted)
                VStack(alignment: .leading, spacing: 8) {
                    GroupLabel(text: "Address")
                    TextField("", text: $settings.data.speechServerURL, prompt: Text("https://…/v1").foregroundStyle(Theme.muted))
                        .font(.system(size: 15)).foregroundStyle(Theme.text)
                        .textInputAutocapitalization(.never).autocorrectionDisabled().keyboardType(.URL)
                        .padding(.horizontal, 14).frame(height: 48)
                        .background(RoundedRectangle(cornerRadius: 12).fill(Theme.surface))
                }
                RowGroup {
                    NavigationLink(value: ModelsRoute.speechServerKey) {
                        RowLabel(title: "API key", subtitle: settings.key(.speechServer) == nil ? "Not set" : nil) {
                            if let hint = settings.keyHint(.speechServer) {
                                Text(hint).font(.system(size: 14)).foregroundStyle(Theme.secondary)
                            }
                            Chevron()
                        }
                    }
                    .buttonStyle(.plain)
                }
                PrimaryPill(title: testing ? "Testing…" : "Test connection", systemImage: "antenna.radiowaves.left.and.right") {
                    Task { await test() }
                }
                .disabled(testing)
                if let result {
                    Label(result.text, systemImage: result.ok ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                        .font(.system(size: 15)).foregroundStyle(result.ok ? Theme.text : Theme.danger)
                        .fixedSize(horizontal: false, vertical: true)
                }
                VStack(alignment: .leading, spacing: 6) {
                    Text("On your PC, in PowerShell in F:\\AI\\STT-Server:").font(.system(size: 13)).foregroundStyle(Theme.muted)
                    Text(".\\keys.ps1 create \"Earshot iPhone\"   makes a key; copy it here\n.\\start.ps1   starts the server, about 2 minutes\n.\\status.ps1   checks it")
                        .font(.system(size: 13, design: .monospaced)).foregroundStyle(Theme.textBody)
                        .textSelection(.enabled)
                }
            }
            .padding(20)
        }
        .background(Theme.ground.ignoresSafeArea())
        .toolbar(.hidden, for: .navigationBar)
    }

    private func test() async {
        testing = true
        defer { testing = false }
        guard let url = URL(string: settings.data.speechServerURL.trimmingCharacters(in: .whitespacesAndNewlines)), url.scheme != nil else {
            result = (false, "That address isn't a full URL. It should start with https://.")
            return
        }
        result = await SpeechServer.test(baseURL: url, key: settings.key(.speechServer))
    }
}
