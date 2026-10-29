import SwiftUI

/// A risky step Hermes wants to take, waiting for the user. It comes as an `approval.request` event
/// in the reply's stream, and is answered with POST /v1/runs/{run_id}/approval.
struct HermesApproval: Equatable {
    let runID: String
    let command: String
    let description: String
    /// What the server accepts, e.g. once, session, deny.
    let choices: [String]
    let requestID: String?
    let generation: Int?

    init?(_ event: [String: Any]) {
        guard let runID = event["run_id"] as? String else { return nil }
        self.runID = runID
        command = event["command"] as? String ?? ""
        description = event["description"] as? String ?? ""
        choices = event["choices"] as? [String] ?? ["once", "deny"]
        requestID = (event["request_id"] as? String) ?? (event["prompt_id"] as? String)
        generation = event["execution_generation"] as? Int
    }

    /// What is read aloud: the reason when Hermes gives one, else the command itself.
    var spoken: String {
        let what = description.isEmpty ? command : description
        return "Hermes wants to " + (what.isEmpty ? "take a step that needs your OK" : "run something: \(what.prefix(160))") + ". Allow it?"
    }
}

/// The Hermes agent's API server: OpenAI-style chat plus its own approval and health routes.
enum Hermes {
    /// Answers a pending approval. `choice` is once or deny.
    static func answer(_ approval: HermesApproval, choice: String, baseURL: URL, key: String) async throws {
        var request = URLRequest(url: baseURL.appendingPathComponent("runs/\(approval.runID)/approval"))
        request.httpMethod = "POST"
        request.timeoutInterval = 15
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.setValue("Bearer \(key)", forHTTPHeaderField: "authorization")
        var body: [String: Any] = ["choice": choice]
        var requestID = approval.requestID
        var generation = approval.generation
        if requestID == nil || generation == nil, let control = await pendingControl(approval.runID, baseURL: baseURL, key: key) {
            // A run's approval must name the exact prompt; its status has it when the event didn't.
            requestID = requestID ?? (control["prompt_id"] as? String) ?? (control["request_id"] as? String)
            generation = generation ?? (control["execution_generation"] as? Int)
        }
        if let requestID { body["request_id"] = requestID }
        if let generation { body["execution_generation"] = generation }
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, response) = try await URLSession.shared.data(for: request)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard status == 200 else {
            throw BrainError(message: status == 409
                ? "Hermes had already moved on: the request timed out or was answered."
                : "Hermes didn't take the answer. " + SSE.readableError(status: status, body: String(decoding: data, as: UTF8.self)))
        }
    }

    /// The first control a run is waiting on, from its status.
    private static func pendingControl(_ runID: String, baseURL: URL, key: String) async -> [String: Any]? {
        var request = URLRequest(url: baseURL.appendingPathComponent("runs/\(runID)"))
        request.timeoutInterval = 15
        request.setValue("Bearer \(key)", forHTTPHeaderField: "authorization")
        guard let (data, _) = try? await URLSession.shared.data(for: request),
              let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else { return nil }
        var control = (object["pending_controls"] as? [[String: Any]])?.first
        if control != nil, control?["execution_generation"] == nil {
            control?["execution_generation"] = object["execution_generation"]
        }
        return control
    }

    /// Checks the server answers (no key needed), then that the key is accepted.
    static func test(baseURL: URL, key: String?) async -> (ok: Bool, text: String) {
        let health = baseURL.deletingLastPathComponent().appendingPathComponent("health")
        do {
            var request = URLRequest(url: health)
            request.timeoutInterval = 10
            let (data, response) = try await URLSession.shared.data(for: request)
            let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
            guard (response as? HTTPURLResponse)?.statusCode == 200, object?["status"] as? String == "ok" else {
                return (false, "Something answered at that address, but it isn't Hermes. Check the address.")
            }
        } catch {
            return (false, "Hermes didn't answer: \(error.localizedDescription) Check that the server is up and the address is right.")
        }
        guard let key else {
            return (false, "Hermes is running. Add its key to finish setting it up.")
        }
        do {
            var request = URLRequest(url: baseURL.appendingPathComponent("models"))
            request.timeoutInterval = 10
            request.setValue("Bearer \(key)", forHTTPHeaderField: "authorization")
            let (data, response) = try await URLSession.shared.data(for: request)
            let status = (response as? HTTPURLResponse)?.statusCode ?? 0
            guard status == 200 else {
                return (false, status == 401 ? "Hermes didn't accept the key. Copy it again from ~/.hermes/.env on the server."
                                             : SSE.readableError(status: status, body: String(decoding: data, as: UTF8.self)))
            }
            return (true, "Connected. The key works.")
        } catch {
            return (false, "Hermes didn't answer: \(error.localizedDescription)")
        }
    }
}

/// Address, key and a connection test for Hermes.
struct HermesPage: View {
    /// False in the Agent tab's sheet, which is closed by dragging down.
    var showsBack = true
    @EnvironmentObject var settings: AppSettings
    @Environment(\.dismiss) private var dismiss
    @State private var result: (ok: Bool, text: String)?
    @State private var testing = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                PageHeader(title: "Hermes", back: showsBack ? { dismiss() } : nil) { EmptyView() }
                Text("Your Hermes agent on shopto-cloud. It can run commands, use files and its other tools to do what you ask, so replies can take longer. When it wants to do something risky, Earshot asks you first.")
                    .font(.system(size: 14)).foregroundStyle(Theme.muted)
                VStack(alignment: .leading, spacing: 8) {
                    GroupLabel(text: "Address")
                    TextField("", text: $settings.data.hermesURL, prompt: Text("https://…/v1").foregroundStyle(Theme.muted))
                        .font(.system(size: 15)).foregroundStyle(Theme.text)
                        .textInputAutocapitalization(.never).autocorrectionDisabled().keyboardType(.URL)
                        .padding(.horizontal, 14).frame(height: 48)
                        .background(RoundedRectangle(cornerRadius: 12).fill(Theme.surface))
                }
                RowGroup {
                    NavigationLink(value: ModelsRoute.hermesKey) {
                        RowLabel(title: "API key", subtitle: settings.key(.hermes) == nil ? "Not set" : nil) {
                            if let hint = settings.keyHint(.hermes) {
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
                    Text("To see the key, on the server run:").font(.system(size: 13)).foregroundStyle(Theme.muted)
                    Text("grep API_SERVER_KEY ~/.hermes/.env")
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
        guard let url = URL(string: settings.data.hermesURL.trimmingCharacters(in: .whitespacesAndNewlines)), url.scheme != nil else {
            result = (false, "That address isn't a full URL. It should start with https://.")
            return
        }
        result = await Hermes.test(baseURL: url, key: settings.key(.hermes))
    }
}
