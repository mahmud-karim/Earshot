import AVFoundation
import Foundation
import os
import SwiftUI
import UIKit

let talkLog = Logger(subsystem: "com.mahmudkarim.Earshot", category: "talk")

enum TalkPhase: Equatable {
    case idle, listening, transcribing, thinking, speaking
}

/// Hands microphone samples to whoever is listening. Called on the audio thread.
final class AudioRouter {
    var onLevel: ((Float, Double) -> Void)?

    private let lock = NSLock()
    private var transcriber: Transcriber?
    private var muted = false
    /// The last moment of audio, so that speech which interrupts a reply keeps its first word.
    private var preroll: [Float] = []
    private let prerollLimit = 12_800

    func handle(_ samples: [Float], _ level: Float) {
        lock.lock()
        if let transcriber, !muted {
            lock.unlock()
            transcriber.feed(samples)
        } else {
            preroll.append(contentsOf: samples)
            if preroll.count > prerollLimit {
                preroll.removeFirst(preroll.count - prerollLimit)
            }
            lock.unlock()
        }
        let seconds = Double(samples.count) / 16_000
        DispatchQueue.main.async { self.onLevel?(level, seconds) }
    }

    func attach(_ transcriber: Transcriber, withPreroll: Bool) {
        lock.lock()
        let earlier = withPreroll ? preroll : []
        preroll = []
        self.transcriber = transcriber
        muted = false
        lock.unlock()
        if !earlier.isEmpty { transcriber.feed(earlier) }
    }

    func detach() {
        lock.lock()
        transcriber = nil
        preroll = []
        lock.unlock()
    }

    func setMuted(_ value: Bool) {
        lock.lock()
        muted = value
        lock.unlock()
    }
}

/// Recent microphone levels for the bars, published about 20 times a second.
@MainActor
final class LevelMeter: ObservableObject {
    @Published private(set) var levels: [Float] = Array(repeating: 0, count: 28)
    private var peak: Float = 0
    private var gathered = 0.0

    func add(_ level: Float, seconds: Double) {
        peak = max(peak, level)
        gathered += seconds
        guard gathered >= 0.05 else { return }
        var next = levels
        next.removeFirst()
        next.append(peak)
        levels = next
        peak = 0
        gathered = 0
    }
}

/// One conversation at a time: listen, think, speak, and back.
@MainActor
final class Conversation: ObservableObject {
    @Published private(set) var phase: TalkPhase = .idle
    @Published private(set) var liveText = ""
    /// The microphone bars live in their own object: they change many times a second, and only the
    /// bars should redraw for that, not every screen that watches the conversation.
    let meter = LevelMeter()
    @Published var chat: Chat?
    @Published var attachment: Attachment?
    /// The reply while it is being written.
    @Published private(set) var reply = ""
    @Published private(set) var replyModel = ""
    @Published var error: String?
    @Published private(set) var status: String?
    @Published private(set) var paused = false
    @Published private(set) var transcriberGivesLiveText = true

    let settings: AppSettings
    let store: ChatStore
    let spend: SpendMeter

    private let router = AudioRouter()
    private var transcriber: Transcriber?
    private var transcriberReady = false
    private var speech: SpeechQueue?
    private var replyTask: Task<Void, Never>?

    private var noiseFloor: Float = -60
    private var lastVoice = Date()
    private var listenStart = Date()
    private var voicedSeconds = 0.0
    private var heard = false
    private var interruptSeconds = 0.0

    /// Listens while a reply is spoken, to notice the user talking over it.
    private var interruptListener: Transcriber?
    /// Everything the current reply has said, for telling its own echo from the user.
    private var spokenText = ""
    /// Loud moments while speaking; the user is close to the phone, the echo and a TV are not.
    private var lastLoud = Date.distantPast
    private var peakWhileSpeaking: Float = -100
    /// Words of the reply the user interrupted, dropped from the start of what they say next.
    private var echoWords: Set<String> = []

    init(settings: AppSettings, store: ChatStore, spend: SpendMeter) {
        self.settings = settings
        self.store = store
        self.spend = spend
        let router = self.router
        AudioIO.shared.onSamples = { samples, level in router.handle(samples, level) }
        router.onLevel = { [weak self] level, seconds in self?.onLevel(level, seconds) }
    }

    var isBusy: Bool { phase != .idle }

    // MARK: - Listening

    func startListening(interrupting: Bool = false) {
        guard phase == .idle || phase == .speaking || phase == .thinking else { return }
        if phase != .idle {
            interruptReply()
        }
        error = nil
        Task { await beginListening(withPreroll: interrupting) }
    }

    private func beginListening(withPreroll: Bool) async {
        guard await AVAudioApplication.requestRecordPermission() else {
            error = "Earshot needs the microphone. Allow it for LiveContainer in Settings › Privacy › Microphone."
            return
        }
        do {
            try AudioIO.shared.start()
        } catch {
            self.error = "The microphone could not start: \(error.localizedDescription)"
            return
        }
        let made: Transcriber
        do {
            made = try makeTranscriber()
        } catch {
            self.error = error.localizedDescription
            stopAudioIfIdle()
            return
        }
        transcriber = made
        transcriberReady = false
        transcriberGivesLiveText = made.givesLiveText
        made.onText = { [weak self] text in
            guard let self, self.transcriber === made else { return }
            self.liveText = text
            self.lastVoice = Date()
            if !text.isEmpty { self.heard = true }
        }
        made.onStatus = { [weak self] text in self?.status = text }
        liveText = ""
        paused = false
        heard = false
        voicedSeconds = 0
        listenStart = Date()
        lastVoice = Date()
        phase = .listening
        Haptics.tap()
        do {
            try await made.start()
            guard transcriber === made else { return }
            router.attach(made, withPreroll: withPreroll)
            transcriberReady = true
            listenStart = Date()
        } catch {
            guard transcriber === made else { return }
            self.error = error.localizedDescription
            cancelListening(reason: "start failed")
        }
    }

    /// Stops listening and sends what was heard.
    func sendNow(reason: String = "button") {
        guard phase == .listening, let current = transcriber else { return }
        talkLog.notice("send (\(reason, privacy: .public))")
        phase = .transcribing
        router.detach()
        transcriberReady = false
        Haptics.tap()
        Task {
            do {
                let text = try await current.finish()
                guard transcriber === current else { return }
                transcriber = nil
                let words = Conversation.dropEcho(text.isEmpty ? liveText : text, echoWords)
                echoWords = []
                guard !words.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                    phase = .idle
                    liveText = ""
                    stopAudioIfIdle()
                    return
                }
                submit(words)
            } catch {
                transcriber = nil
                self.error = error.localizedDescription
                phase = .idle
                stopAudioIfIdle()
            }
        }
    }

    func cancelListening(reason: String = "button") {
        talkLog.notice("cancel listening (\(reason, privacy: .public)) in phase \(String(describing: self.phase), privacy: .public)")
        transcriber?.cancel()
        transcriber = nil
        transcriberReady = false
        router.detach()
        liveText = ""
        status = nil
        phase = .idle
        stopAudioIfIdle()
    }

    func togglePause() {
        paused.toggle()
        router.setMuted(paused)
        lastVoice = Date()
    }

    private func onLevel(_ level: Float, _ seconds: Double) {
        let shown = max(0, min(1, (level + 55) / 45))
        meter.add(phase == .listening && !paused ? shown : 0, seconds: seconds)

        if level < noiseFloor {
            noiseFloor = level
        } else {
            noiseFloor += (level - noiseFloor) * Float(seconds * 0.15)
        }

        let now = Date()
        switch phase {
        case .listening where transcriberReady && !paused:
            if level > max(noiseFloor + 12, -52) {
                voicedSeconds += seconds
                lastVoice = now
                if voicedSeconds > 0.2 { heard = true }
            }
            let silence = now.timeIntervalSince(lastVoice)
            let hasWords = !transcriberGivesLiveText || !liveText.isEmpty
            if heard && hasWords && silence > settings.data.pauseSeconds {
                sendNow(reason: "pause")
            } else if !heard && now.timeIntervalSince(listenStart) > 10 {
                cancelListening(reason: "nothing heard")
            }
        case .speaking where settings.data.interruptByTalking:
            let loud = level > max(noiseFloor + 15, -40)
            if loud { lastLoud = now }
            if level > peakWhileSpeaking { peakWhileSpeaking = level }
            // Without live words to check, only a long, loud stretch counts as the user.
            guard interruptListener == nil else { break }
            if level > max(noiseFloor + 25, -28) {
                interruptSeconds += seconds
            } else {
                interruptSeconds = max(0, interruptSeconds - seconds)
            }
            if interruptSeconds > 0.6 {
                interruptSeconds = 0
                startListening(interrupting: true)
            }
        default:
            interruptSeconds = 0
        }
    }

    // MARK: - Asking

    /// Sends a question, typed or heard.
    func submit(_ text: String) {
        let question = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !question.isEmpty else { return }
        if phase == .thinking || phase == .speaking {
            interruptReply()
        }
        var current = chat ?? Chat(title: Conversation.title(from: question), kind: attachment == nil ? .voice : .shot)
        var turn = Turn(isUser: true, text: question)
        if let attachment {
            turn.imageFile = store.saveImage(attachment.jpeg)
            turn.attachmentLabel = attachment.label
        }
        attachment = nil
        current.turns.append(turn)
        chat = current
        store.save(current)
        liveText = ""
        startReply()
    }

    func retry() {
        guard var current = chat, let last = current.turns.last, !last.isUser, last.failed else { return }
        current.turns.removeLast()
        chat = current
        store.save(current)
        startReply()
    }

    private func startReply() {
        guard let current = chat else { return }
        let preset = settings.preset
        let brain: Brain
        do {
            brain = try makeBrain(preset.brain)
        } catch {
            appendReply(Turn(isUser: false, text: error.localizedDescription, failed: true))
            phase = .idle
            stopAudioIfIdle()
            return
        }

        try? AudioIO.shared.start()
        let queue = SpeechQueue(synthesizer: makeSynthesizer(preset.voice))
        queue.onStart = { [weak self] in
            guard let self, self.speech === queue, self.phase == .thinking else { return }
            self.phase = .speaking
            self.armInterrupt()
        }
        queue.onDone = { [weak self] in
            guard let self, self.speech === queue else { return }
            self.speechFinished()
        }
        queue.onError = { [weak self] message in self?.error = message }
        speech = queue

        let request = BrainRequest(model: preset.brain.model, system: Conversation.systemPrompt(),
                                   messages: history(current), effort: preset.effort)
        let label = preset.brain.label
        let model = preset.brain.model
        let chatID = current.id
        reply = ""
        spokenText = ""
        replyModel = label
        phase = .thinking

        replyTask = Task { [weak self] in
            let started = Date()
            var firstWord: Double?
            var input = 0
            var output = 0
            var splitter = SentenceSplitter()
            do {
                for try await event in brain.stream(request) {
                    guard let self else { return }
                    switch event {
                    case .text(let text):
                        if firstWord == nil { firstWord = Date().timeIntervalSince(started) }
                        self.reply += text
                        self.spokenText += text
                        for sentence in splitter.push(text) { self.speech?.enqueue(sentence) }
                    case .usage(let i, let o):
                        input = i
                        output = o
                    }
                }
                guard let self else { return }
                for sentence in splitter.flush() { self.speech?.enqueue(sentence) }
                let cost = Catalog.cost(model: model, input: input, output: output)
                if let cost { self.spend.add(cost) }
                self.appendReply(Turn(isUser: false, text: self.reply, modelLabel: label, cost: cost, firstWordSeconds: firstWord), to: chatID)
                self.reply = ""
                if let speech = self.speech {
                    speech.finishInput()
                } else if self.phase == .thinking || self.phase == .speaking {
                    self.phase = .idle
                    self.stopAudioIfIdle()
                }
            } catch {
                guard let self else { return }
                let partial = self.reply
                self.reply = ""
                if Task.isCancelled {
                    if !partial.isEmpty {
                        self.appendReply(Turn(isUser: false, text: partial, modelLabel: label, firstWordSeconds: firstWord), to: chatID)
                    }
                    return
                }
                self.speech?.stop()
                self.speech = nil
                self.appendReply(Turn(isUser: false, text: partial.isEmpty ? error.localizedDescription : partial + "\n\n" + error.localizedDescription,
                                      modelLabel: label, failed: true), to: chatID)
                if self.phase == .thinking || self.phase == .speaking {
                    self.phase = .idle
                    self.stopAudioIfIdle()
                }
            }
        }
    }

    /// Stops the reply and its voice; what was written so far is kept.
    func stopSpeaking() {
        interruptSeconds = 0
        interruptReply()
        stopAudioIfIdle()
    }

    /// Stops the reply and its voice, keeping what was written so far.
    private func interruptReply() {
        disarmInterrupt()
        speech?.stop()
        speech = nil
        replyTask?.cancel()
        replyTask = nil
        phase = .idle
    }

    private func speechFinished() {
        disarmInterrupt()
        speech = nil
        if settings.data.handsFree && phase == .speaking {
            phase = .idle
            startListening()
        } else if phase == .speaking || phase == .thinking {
            phase = .idle
            stopAudioIfIdle()
        }
    }

    // MARK: - Talking over a reply

    /// Starts listening under the reply. The user has interrupted when it hears words the reply
    /// did not say, while the room is loud; its own echo repeats the reply's words.
    private func armInterrupt() {
        guard settings.data.interruptByTalking, settings.data.ears == .apple, interruptListener == nil else { return }
        let listener = AppleTranscriber()
        interruptListener = listener
        listener.onText = { [weak self] text in
            guard let self, self.interruptListener === listener, self.phase == .speaking else { return }
            let reply = Conversation.wordSet(self.spokenText)
            let fresh = Conversation.words(text).filter { !reply.contains($0) }
            let sinceLoud = Date().timeIntervalSince(self.lastLoud)
            talkLog.notice("under reply heard \(text, privacy: .public) fresh=\(fresh.count) sinceLoud=\(sinceLoud, format: .fixed(precision: 2)) floor=\(self.noiseFloor, format: .fixed(precision: 1))")
            if fresh.count >= 2 && sinceLoud < 1.0 {
                self.takeOver(listener, text: text, echo: reply)
            }
        }
        Task {
            do {
                try await listener.start()
                guard interruptListener === listener, phase == .speaking else {
                    listener.cancel()
                    return
                }
                router.attach(listener, withPreroll: false)
            } catch {
                if interruptListener === listener { interruptListener = nil }
            }
        }
    }

    private func disarmInterrupt() {
        if peakWhileSpeaking > -100 {
            talkLog.notice("reply over: peak level \(self.peakWhileSpeaking, format: .fixed(precision: 1)) floor \(self.noiseFloor, format: .fixed(precision: 1))")
            peakWhileSpeaking = -100
        }
        guard let listener = interruptListener else { return }
        interruptListener = nil
        listener.cancel()
        if transcriber == nil { router.detach() }
    }

    /// The user is talking: stop the reply and keep listening with what was already heard.
    private func takeOver(_ listener: Transcriber, text: String, echo: Set<String>) {
        interruptListener = nil
        speech?.stop()
        speech = nil
        replyTask?.cancel()
        replyTask = nil
        echoWords = echo
        transcriber = listener
        transcriberReady = true
        transcriberGivesLiveText = true
        listener.onText = { [weak self] text in
            guard let self, self.transcriber === listener else { return }
            self.liveText = Conversation.dropEcho(text, self.echoWords)
            self.lastVoice = Date()
            if !self.liveText.isEmpty { self.heard = true }
        }
        liveText = Conversation.dropEcho(text, echo)
        paused = false
        heard = true
        voicedSeconds = 0
        listenStart = Date()
        lastVoice = Date()
        phase = .listening
        Haptics.tap()
    }

    static func words(_ text: String) -> [String] {
        text.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { !$0.isEmpty }
    }

    static func wordSet(_ text: String) -> Set<String> {
        Set(words(text))
    }

    /// Drops the reply's own words from the start of what was heard.
    static func dropEcho(_ text: String, _ echo: Set<String>) -> String {
        guard !echo.isEmpty else { return text }
        let parts = text.split(separator: " ", omittingEmptySubsequences: true)
        var start = 0
        while start < parts.count, let word = words(String(parts[start])).first, echo.contains(word) {
            start += 1
        }
        return parts[start...].joined(separator: " ")
    }

    private var previewQueue: SpeechQueue?

    /// Says a line in a voice, from the Models sheet.
    func preview(_ voice: VoiceChoice) {
        guard phase == .idle else { return }
        previewQueue?.stop()
        do {
            try AudioIO.shared.start()
        } catch {
            self.error = error.localizedDescription
            return
        }
        let queue = SpeechQueue(synthesizer: makeSynthesizer(voice))
        queue.onDone = { [weak self] in
            guard let self, self.previewQueue === queue else { return }
            self.previewQueue = nil
            self.stopAudioIfIdle()
        }
        queue.onError = { [weak self] message in self?.error = message }
        previewQueue = queue
        queue.enqueue("Hi, this is how I sound. Ask me anything.")
        queue.finishInput()
    }

    // MARK: - Chats

    func newChat() {
        cancelEverything()
        chat = nil
    }

    func open(_ chat: Chat) {
        cancelEverything()
        self.chat = chat
    }

    func cancelEverything() {
        if phase == .listening || phase == .transcribing { cancelListening(reason: "chat changed") }
        interruptReply()
        stopAudioIfIdle()
    }

    func attach(_ image: UIImage, label: String) {
        attachment = Attachment.make(image, label: label)
    }

    /// Adds a turn to its chat, which may no longer be the open one.
    private func appendReply(_ turn: Turn, to chatID: UUID? = nil) {
        let id = chatID ?? chat?.id
        if var current = chat, current.id == id {
            current.turns.append(turn)
            chat = current
            store.save(current)
        } else if var other = store.chats.first(where: { $0.id == id }) {
            other.turns.append(turn)
            store.save(other)
        }
    }

    private func stopAudioIfIdle() {
        if phase == .idle {
            AudioIO.shared.stop()
        }
    }

    // MARK: - Building requests

    private func history(_ chat: Chat) -> [BrainMessage] {
        let turns = chat.turns.filter { !$0.failed && !$0.text.isEmpty }
        // Images are sent for the two most recent pictures only.
        let recentImages = Set(turns.filter { $0.imageFile != nil }.suffix(2).map(\.id))
        var messages: [BrainMessage] = []
        for turn in turns {
            let role: BrainMessage.Role = turn.isUser ? .user : .assistant
            var text = turn.text
            if turn.isUser, let label = turn.attachmentLabel, turn.imageFile != nil {
                text = "[\(label) attached]\n" + text
            }
            let image = recentImages.contains(turn.id) ? turn.imageFile.flatMap(store.imageData) : nil
            // The API wants turns to alternate; merge two in a row from the same side.
            if let last = messages.last, last.role == role {
                messages[messages.count - 1].text += "\n\n" + text
                if messages[messages.count - 1].image == nil { messages[messages.count - 1].image = image }
            } else {
                messages.append(BrainMessage(role: role, text: text, image: image))
            }
        }
        while let first = messages.first, first.role == .assistant {
            messages.removeFirst()
        }
        return messages
    }

    private func makeBrain(_ choice: BrainChoice) throws -> Brain {
        switch choice.kind {
        case .anthropic:
            guard let key = settings.key(.anthropic) else { throw BrainError(message: "Add your Anthropic API key in Settings to use Claude.") }
            return AnthropicBrain(apiKey: key)
        case .openRouter:
            guard let key = settings.key(.openRouter) else { throw BrainError(message: "Add your OpenRouter key in Settings.") }
            return OpenAICompatibleBrain(baseURL: URL(string: "https://openrouter.ai/api/v1")!, apiKey: key,
                                         extraHeaders: ["HTTP-Referer": "https://github.com/mahmud-karim/Earshot", "X-Title": "Earshot"])
        case .openAI:
            guard let key = settings.key(.openAI) else { throw BrainError(message: "Add your OpenAI key in Settings.") }
            return OpenAICompatibleBrain(baseURL: URL(string: "https://api.openai.com/v1")!, apiKey: key)
        case .pc:
            guard let url = settings.pcURL(port: settings.data.pcChatPort) else { throw BrainError(message: "Set your PC's address in Settings.") }
            return OpenAICompatibleBrain(baseURL: url, apiKey: nil)
        }
    }

    private func makeTranscriber() throws -> Transcriber {
        switch settings.data.ears {
        case .apple:
            return AppleTranscriber()
        case .openAI:
            guard let key = settings.key(.openAI) else { throw BrainError(message: "Add your OpenAI key in Settings, or pick other Ears in Models.") }
            return UploadTranscriber(baseURL: URL(string: "https://api.openai.com/v1")!, apiKey: key, model: settings.data.openAITranscribeModel)
        case .pc:
            guard let url = settings.pcURL(port: settings.data.pcSpeechPort) else { throw BrainError(message: "Set your PC's address in Settings.") }
            return UploadTranscriber(baseURL: url, apiKey: nil, model: settings.data.pcTranscribeModel)
        }
    }

    private func makeSynthesizer(_ choice: VoiceChoice) -> Synthesizer {
        switch choice.kind {
        case .apple:
            return AppleSynthesizer(identifier: choice.voice)
        case .openAI:
            if let key = settings.key(.openAI) {
                return OpenAISpeechSynthesizer(baseURL: URL(string: "https://api.openai.com/v1")!, apiKey: key,
                                               model: settings.data.openAIVoiceModel, voice: choice.voice.isEmpty ? "coral" : choice.voice)
            }
        case .elevenLabs:
            if let key = settings.key(.elevenLabs) {
                return ElevenLabsSynthesizer(apiKey: key, voiceID: choice.voice, model: settings.data.elevenLabsModel)
            }
        case .pc:
            if let url = settings.pcURL(port: settings.data.pcSpeechPort) {
                return OpenAISpeechSynthesizer(baseURL: url, apiKey: nil, model: settings.data.pcVoiceModel,
                                               voice: choice.voice.isEmpty ? "af_heart" : choice.voice)
            }
        }
        error = "That voice isn't set up yet, so Apple's voice is speaking instead."
        return AppleSynthesizer(identifier: "")
    }

    static func title(from text: String) -> String {
        let clean = text.replacingOccurrences(of: "\n", with: " ")
        return clean.count > 48 ? String(clean.prefix(47)) + "…" : clean
    }

    static func systemPrompt() -> String {
        let date = Date().formatted(date: .complete, time: .shortened)
        return """
        You are Earshot, a voice assistant on the user's iPhone. Your replies are read aloud, so:
        - Talk the way a person would. Lead with the answer and keep it short: one to three sentences for a simple question, more only when the user asks for detail or the task needs it.
        - Write plain sentences. No headings, tables, bullet lists or bold text.
        - When the user asks you to draft something they will copy or send (a message, an email, a note, code), put only the draft inside a fenced block (```), and say one short sentence before it. The fenced block is shown on screen and is not read aloud.
        - An attached image is usually a screenshot of what was on the user's screen when they asked. Talk about what is in it directly.
        - Speech recognition sometimes mishears words. If a question looks garbled, answer your best guess and say briefly what you assumed.
        It is \(date). The user's locale is \(Locale.current.identifier).
        """
    }
}

enum Haptics {
    @MainActor static func tap() {
        UIImpactFeedbackGenerator(style: .light).impactOccurred()
    }
}
