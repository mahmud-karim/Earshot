import AVFoundation
import Foundation
import LocalVoices
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

    /// What the transcriber was given, with each piece's level, for the voiceprint check. Up to a minute.
    private var heard: [(samples: [Float], level: Float)] = []
    private var heardCount = 0

    func handle(_ samples: [Float], _ level: Float) {
        lock.lock()
        if let transcriber, !muted {
            heard.append((samples, level))
            heardCount += samples.count
            while heardCount > 16_000 * 60, !heard.isEmpty {
                heardCount -= heard.removeFirst().samples.count
            }
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
        heard = []
        heardCount = 0
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

    /// The pieces of what was heard louder than `minLevel`: the talking, without the pauses.
    func voiced(above minLevel: Float) -> [Float] {
        lock.withLock { heard.filter { $0.level > minLevel }.flatMap(\.samples) }
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
    /// The earlier answer being read out again, if any.
    @Published private(set) var replaying: UUID?
    /// The reply is waiting on a web search or a page.
    @Published private(set) var searching = false
    /// Pages the reply in progress draws on.
    @Published private(set) var replySources: [Source] = []
    /// The on-device voice model being loaded, by name. Nothing is heard until it is ready.
    @Published private(set) var loadingVoice: String?
    /// Sound is actually coming out. A reply stays in the speaking phase between sentences, while the
    /// next one is still being made; the speaking animation follows this instead.
    @Published private(set) var sounding = false
    /// The step Hermes is on, e.g. "date" for a terminal command.
    @Published private(set) var working: String?
    /// A risky step Hermes is waiting on the user to allow.
    @Published private(set) var approval: HermesApproval?
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
    private var speech: SpeechQueue? {
        didSet { if speech !== oldValue { sounding = false } }
    }
    private var replyTask: Task<Void, Never>?

    private var noiseFloor: Float = -60
    private var lastVoice = Date()
    private var listenStart = Date()
    private var voicedSeconds = 0.0
    private var heard = false
    private var interruptSeconds = 0.0
    /// How long the microphone has given pure digital silence while listening.
    private var deadSeconds = 0.0
    private var restartsThisTurn = 0

    /// Listens while a reply is spoken, to notice the user talking over it.
    private var interruptListener: Transcriber?
    /// Everything the current reply has said, for telling its own echo from the user.
    private var spokenText = ""
    /// Loud moments while speaking; the user is close to the phone, the echo and a TV are not.
    private var lastLoud = Date.distantPast
    private var peakWhileSpeaking: Float = -100
    /// Words of the reply the user interrupted, dropped from the start of what they say next.
    private var echoWords: Set<String> = []

    /// The Agent tab's conversation: always Hermes, with its own chats.
    let isAgent: Bool
    /// Whether the other tab's conversation is using the audio, so this one leaves it running.
    var otherIsBusy: () -> Bool = { false }

    init(settings: AppSettings, store: ChatStore, spend: SpendMeter, isAgent: Bool = false) {
        self.settings = settings
        self.store = store
        self.spend = spend
        self.isAgent = isAgent
        router.onLevel = { [weak self] level, seconds in self?.onLevel(level, seconds) }
        claimMicrophone()
    }

    /// Both tabs share one microphone; whichever starts listening or recording takes it.
    private func claimMicrophone() {
        let router = self.router
        AudioIO.shared.onSamples = { samples, level in router.handle(samples, level) }
    }

    /// What answers in this tab.
    private var brainChoice: BrainChoice {
        isAgent ? BrainChoice(kind: .hermes, model: "hermes-agent") : settings.preset.brain
    }

    var isBusy: Bool { phase != .idle }

    // MARK: - Listening

    func startListening(interrupting: Bool = false) {
        guard phase == .idle || phase == .speaking || phase == .thinking else { return }
        if approval != nil {
            // Hermes is waiting for a yes or no: hush the question but keep its turn open.
            disarmInterrupt()
            freshSpeech()
        } else if phase != .idle {
            interruptReply()
        }
        error = nil
        // No preroll: the moment before a tap over a reply is the reply's own voice.
        Task { await beginListening(withPreroll: false) }
    }

    private func beginListening(withPreroll: Bool) async {
        claimMicrophone()
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
        warmUpVoice()
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
        deadSeconds = 0
        restartsThisTurn = 0
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
        // Only a send that happened on its own is checked: a tap on send is the user's own say-so.
        let voiced = reason == "pause" && VoiceFilter.shared.isOn(settings) ? router.voiced(above: max(noiseFloor + 12, -52)) : nil
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
                if let voiced, let match = await VoiceFilter.shared.match(voiced, settings: settings), !match.isYou {
                    ignoreSomeoneElse(words, score: match.score)
                    return
                }
                ignoredInRow = 0
                submit(words)
            } catch {
                transcriber = nil
                self.error = error.localizedDescription
                phase = .idle
                stopAudioIfIdle()
            }
        }
    }

    private var ignoredInRow = 0

    /// Speech that didn't match the user's voiceprint (the TV, someone else) is dropped. Earshot goes
    /// back to listening for the user, but not forever: after two in a row it stops.
    private func ignoreSomeoneElse(_ words: String, score: Float) {
        talkLog.notice("voice filter ignored \(words, privacy: .public) score=\(score, format: .fixed(precision: 2))")
        ignoredInRow += 1
        liveText = ""
        phase = .idle
        let shown = words.count > 60 ? String(words.prefix(57)) + "…" : words
        error = "Ignored “\(shown)”: it didn't sound like you (match \(Int(score * 100))%)."
        if ignoredInRow < 2 {
            Task { await beginListening(withPreroll: false) }
        } else {
            ignoredInRow = 0
            stopAudioIfIdle()
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
        if (phase == .listening && !paused) || phase == .thinking {
            // With voice processing on, the input of a freshly started engine can stay at exact
            // zeros until the engine is restarted once (seen on iOS 27). A live microphone, even in
            // a quiet room, is never exactly zero, so a third of a second of zeros means restart.
            deadSeconds = level <= -99.9 ? deadSeconds + seconds : 0
            if deadSeconds > 0.3 && restartsThisTurn < 2 {
                restartsThisTurn += 1
                deadSeconds = 0
                talkLog.notice("microphone gives only zeros; restarting the audio engine")
                AudioIO.shared.restart()
                if phase == .listening { listenStart = now }
            }
        }
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
            // Loudness alone never interrupts: the reply's own voice through the microphone is as
            // loud as the user, and was being taken as a question. The listener in armInterrupt
            // decides by the words it hears.
        default:
            interruptSeconds = 0
        }
    }

    // MARK: - Asking

    /// Sends a question, typed or heard.
    func submit(_ text: String) {
        let question = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !question.isEmpty else { return }
        if approval != nil, let choice = Conversation.approvalChoice(question) {
            liveText = ""
            answerApproval(choice)
            return
        }
        if phase == .thinking || phase == .speaking || approval != nil {
            interruptReply()
        }
        var current = chat ?? Chat(title: Conversation.title(from: question), kind: isAgent ? .agent : attachment == nil ? .voice : .shot)
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

    /// Answers the chat's last question. `resuming` follows a Hermes run already working on it.
    private func startReply(resuming runID: String? = nil) {
        guard let current = chat else { return }
        let preset = settings.preset
        let brain: Brain
        do {
            if isAgent, let question = current.turns.last(where: \.isUser), question.imageFile == nil {
                // A run carries on on the server without the phone; its answer is collected later if need be.
                let (url, key) = try hermesAccess()
                brain = HermesRunBrain(baseURL: url, apiKey: key, sessionID: "earshot-" + current.id.uuidString.lowercased(),
                                       resumeRunID: runID, idempotencyKey: "earshot-" + question.id.uuidString.lowercased(),
                                       askedAt: question.date)
            } else {
                brain = try makeBrain(brainChoice)
            }
        } catch {
            appendReply(Turn(isUser: false, text: error.localizedDescription, failed: true))
            phase = .idle
            stopAudioIfIdle()
            return
        }

        try? AudioIO.shared.start()
        freshSpeech()

        let choice = brainChoice
        let webAccess = choice.kind == .anthropic && settings.data.webSearch
        let request = BrainRequest(model: choice.model,
                                   system: Conversation.systemPrompt(webAccess: webAccess, agent: choice.kind == .hermes),
                                   messages: history(current), effort: preset.effort, webAccess: webAccess)
        let label = choice.label
        let model = choice.model
        let chatID = current.id
        reply = ""
        spokenText = ""
        replySources = []
        searching = false
        working = nil
        approval = nil
        replyModel = label
        deadSeconds = 0
        restartsThisTurn = 0
        phase = .thinking

        // A reply keeps going for a while if the user switches apps; iOS gives about 30 seconds.
        let background = UIApplication.shared.beginBackgroundTask(withName: "reply") { [weak self] in
            // Called on the main thread when the time runs out; it must end the task before returning.
            MainActor.assumeIsolated { self?.endBackground() }
        }
        endBackground()
        backgroundTask = background
        replyTask = Task { [weak self] in
            defer { self?.endBackground(background) }
            let started = Date()
            var firstWord: Double?
            var input = 0
            var output = 0
            var splitter = SentenceSplitter()
            var searches = 0
            do {
                for try await event in brain.stream(request) {
                    guard let self else { return }
                    switch event {
                    case .text(let text):
                        if firstWord == nil { firstWord = Date().timeIntervalSince(started) }
                        self.searching = false
                        self.working = nil
                        self.reply += text
                        self.spokenText += text
                        for sentence in splitter.push(text) { self.speech?.enqueue(sentence) }
                    case .usage(let i, let o):
                        input = i
                        output = o
                    case .searching(let webSearch):
                        if webSearch { searches += 1 }
                        // Searching takes a few seconds. Say so, unless the answer already has.
                        if !self.searching && self.spokenText.isEmpty {
                            let cue = "Let me look that up."
                            self.spokenText += cue + " "
                            self.speech?.enqueue(cue)
                        }
                        self.searching = true
                    case .source(let title, let url):
                        self.replySources.append(Source(title: title, url: url))
                    case .working(let tool, let label):
                        self.working = label.isEmpty ? tool : label
                        // Agent steps can take a while. Say so once, unless the answer already has.
                        if self.spokenText.isEmpty {
                            let cue = "On it."
                            self.spokenText += cue + " "
                            self.speech?.enqueue(cue)
                        }
                    case .approval(let approval):
                        self.approval = approval
                        self.spokenText += approval.spoken + " "
                        self.speech?.enqueue(approval.spoken)
                    case .runStarted(let runID):
                        self.currentRun = (chatID, runID)
                        self.setPendingRun(runID, chatID: chatID)
                    }
                }
                guard let self else { return }
                self.searching = false
                self.working = nil
                self.approval = nil
                // A stopped reply ends its stream quietly rather than with an error.
                if Task.isCancelled {
                    let partial = self.reply
                    self.reply = ""
                    self.replySources = []
                    if self.pendingRun(chatID) == nil {
                        self.saveStopped(partial, label: label, firstWord: firstWord, chatID: chatID)
                    }
                    return
                }
                self.currentRun = nil
                self.setPendingRun(nil, chatID: chatID)
                for sentence in splitter.flush() { self.speech?.enqueue(sentence) }
                if self.reply.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    self.reply = choice.kind == .hermes ? "Hermes finished without saying anything." : "The model didn't answer."
                }
                // Web searches are billed per search on top of tokens.
                let cost = Catalog.cost(model: model, input: input, output: output).map { $0 + Double(searches) * Catalog.searchPrice }
                if let cost { self.spend.add(cost) }
                self.appendReply(Turn(isUser: false, text: self.reply.trimmingCharacters(in: .whitespacesAndNewlines), modelLabel: label,
                                      cost: cost, firstWordSeconds: firstWord,
                                      sources: self.replySources.isEmpty ? nil : self.replySources), to: chatID)
                self.reply = ""
                self.replySources = []
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
                self.searching = false
                self.working = nil
                self.approval = nil
                self.replySources = []
                if Task.isCancelled {
                    // Left for another chat or a new question: a Hermes run carries on and is collected later.
                    if self.pendingRun(chatID) == nil {
                        self.saveStopped(partial, label: label, firstWord: firstWord, chatID: chatID)
                    }
                    return
                }
                self.currentRun = nil
                self.speech?.stop()
                self.speech = nil
                if error is HermesOffline {
                    // Still running on the server: the chat keeps the run and shows that it's waiting.
                    if self.phase == .thinking || self.phase == .speaking {
                        self.phase = .idle
                        self.stopAudioIfIdle()
                    }
                    return
                }
                self.setPendingRun(nil, chatID: chatID)
                self.appendReply(Turn(isUser: false, text: partial.isEmpty ? error.localizedDescription : partial + "\n\n" + error.localizedDescription,
                                      modelLabel: label, failed: true), to: chatID)
                if self.phase == .thinking || self.phase == .speaking {
                    self.phase = .idle
                    self.stopAudioIfIdle()
                }
            }
        }
    }

    private var backgroundTask: UIBackgroundTaskIdentifier = .invalid

    /// The Hermes run this reply follows, for the stop button.
    private var currentRun: (chatID: UUID, runID: String)?

    private func pendingRun(_ chatID: UUID) -> String? {
        chat?.id == chatID ? chat?.pendingRun : store.chats.first(where: { $0.id == chatID })?.pendingRun
    }

    private func setPendingRun(_ runID: String?, chatID: UUID) {
        if var current = chat, current.id == chatID {
            guard current.pendingRun != runID else { return }
            current.pendingRun = runID
            chat = current
            store.save(current)
        } else if var other = store.chats.first(where: { $0.id == chatID }), other.pendingRun != runID {
            other.pendingRun = runID
            store.save(other)
        }
    }

    /// Picks up a Hermes run that was still working when Earshot lost it: closed, suspended or offline.
    /// Its answer is shown and spoken as if it had just arrived.
    func resumePendingRun() {
        guard isAgent, phase == .idle, let current = chat, let runID = current.pendingRun else { return }
        talkLog.notice("resuming Hermes run \(runID, privacy: .public)")
        startReply(resuming: runID)
    }

    /// At launch: the newest agent chat whose answer hasn't arrived yet, so it's collected straight away.
    func openWaitingChat() {
        guard isAgent, chat == nil, phase == .idle,
              let waiting = store.chats.filter({ $0.kind == .agent && $0.pendingRun != nil }).max(by: { $0.updated < $1.updated }) else { return }
        chat = waiting
        resumePendingRun()
    }

    /// Ends the reply's background time: `only` ends it only if it is still that one.
    private func endBackground(_ only: UIBackgroundTaskIdentifier? = nil) {
        guard backgroundTask != .invalid, only == nil || only == backgroundTask else { return }
        UIApplication.shared.endBackgroundTask(backgroundTask)
        backgroundTask = .invalid
    }

    /// What a stopped reply leaves in the chat: what it had said, or a note it was stopped before an
    /// answer, with Retry. An empty turn would show as the waiting dots forever.
    private func saveStopped(_ partial: String, label: String, firstWord: Double?, chatID: UUID) {
        let said = partial.trimmingCharacters(in: .whitespacesAndNewlines)
        appendReply(Turn(isUser: false, text: said.isEmpty ? "Stopped before an answer." : said, modelLabel: label,
                         firstWordSeconds: firstWord, failed: said.isEmpty), to: chatID)
    }

    /// Stops the reply and its voice; what was written so far is kept. A Hermes run is stopped on the
    /// server too.
    func stopSpeaking() {
        if let run = currentRun, let (url, key) = try? hermesAccess() {
            setPendingRun(nil, chatID: run.chatID)
            currentRun = nil
            Task { await Hermes.stop(runID: run.runID, baseURL: url, key: key) }
        } else if let current = chat, current.pendingRun != nil, let (url, key) = try? hermesAccess(), let runID = current.pendingRun {
            setPendingRun(nil, chatID: current.id)
            Task { await Hermes.stop(runID: runID, baseURL: url, key: key) }
        }
        interruptSeconds = 0
        interruptReply()
        stopAudioIfIdle()
    }

    /// Starts the voice for the reply, or a new one for the rest of it when the old one was hushed.
    private func freshSpeech() {
        speech?.stop()
        warmUpVoice()
        let queue = SpeechQueue(synthesizer: makeSynthesizer(settings.preset.voice))
        queue.onStart = { [weak self] in
            guard let self, self.speech === queue, self.phase == .thinking else { return }
            self.phase = .speaking
            self.armInterrupt()
        }
        queue.onDone = { [weak self] in
            guard let self, self.speech === queue else { return }
            self.speechFinished()
        }
        queue.onSounding = { [weak self] on in
            guard let self, self.speech === queue else { return }
            self.sounding = on
        }
        queue.onError = { [weak self] message in self?.error = message }
        speech = queue
    }

    /// Answers Hermes's pending question, by button or by saying yes or no.
    func answerApproval(_ choice: String) {
        guard let pending = approval else { return }
        approval = nil
        if phase == .idle || phase == .listening || phase == .transcribing {
            cancelListeningQuietly()
            phase = .thinking
        }
        Haptics.tap()
        Task {
            do {
                let (url, key) = try hermesAccess()
                try await Hermes.answer(pending, choice: choice, baseURL: url, key: key)
            } catch {
                self.error = error.localizedDescription
            }
        }
    }

    /// "Yes, go ahead" allows once, "no" denies; anything else is a new question.
    static func approvalChoice(_ text: String) -> String? {
        let said = words(text)
        guard !said.isEmpty, said.count <= 6 else { return nil }
        let no: Set<String> = ["no", "nope", "deny", "don", "dont", "stop", "cancel", "never"]
        let yes: Set<String> = ["yes", "yeah", "yep", "sure", "allow", "approve", "ok", "okay", "go", "do"]
        if said.contains(where: no.contains) { return "deny" }
        if said.contains(where: yes.contains) { return "once" }
        return nil
    }

    private func cancelListeningQuietly() {
        transcriber?.cancel()
        transcriber = nil
        transcriberReady = false
        router.detach()
        liveText = ""
        status = nil
    }

    /// Stops the reply and its voice, keeping what was written so far.
    private func interruptReply() {
        disarmInterrupt()
        speech?.stop()
        speech = nil
        replaying = nil
        searching = false
        working = nil
        approval = nil
        // The run itself goes on; the chat keeps it as pending unless the user stopped it.
        currentRun = nil
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

    /// Reads an earlier answer out loud again, in the voice the preset uses now. It plays like a
    /// reply, so the stop button and talking over it work the same; it never starts listening after.
    func replay(_ turn: Turn) {
        guard phase == .idle, !turn.text.isEmpty else { return }
        previewQueue?.stop()
        previewQueue = nil
        do {
            try AudioIO.shared.start()
        } catch {
            self.error = error.localizedDescription
            return
        }
        warmUpVoice()
        let queue = SpeechQueue(synthesizer: makeSynthesizer(settings.preset.voice))
        // Like a reply, it shows as speaking once the first sound plays, not while the voice is made.
        queue.onStart = { [weak self] in
            guard let self, self.speech === queue, self.phase == .thinking else { return }
            self.phase = .speaking
        }
        queue.onDone = { [weak self] in
            guard let self, self.speech === queue else { return }
            self.speech = nil
            self.replaying = nil
            if self.phase == .speaking || self.phase == .thinking { self.phase = .idle }
            self.stopAudioIfIdle()
        }
        queue.onSounding = { [weak self] on in
            guard let self, self.speech === queue else { return }
            self.sounding = on
        }
        queue.onError = { [weak self] message in self?.error = message }
        speech = queue
        replaying = turn.id
        spokenText = turn.text
        phase = .thinking
        var splitter = SentenceSplitter()
        for sentence in splitter.push(turn.text) + splitter.flush() { queue.enqueue(sentence) }
        queue.finishInput()
    }

    // MARK: - Talking over a reply

    /// Starts listening under the reply with Apple's recognizer, whatever ears are picked: only
    /// words tell the user from the reply's own voice coming back through the microphone, which is
    /// as loud as a person. The user has interrupted when it hears words the reply did not say.
    private func armInterrupt() {
        // Not in the Agent tab: talking over a reply ends it, and there that would cut off Hermes mid-job
        // whenever the room is noisy. The Stop button still does.
        guard settings.data.interruptByTalking, !isAgent, interruptListener == nil else { return }
        claimMicrophone()
        let listener = AppleTranscriber()
        interruptListener = listener
        listener.onText = { [weak self] text in
            guard let self, self.interruptListener === listener, self.phase == .speaking else { return }
            let reply = Conversation.wordSet(self.spokenText)
            let heard = Conversation.words(text)
            let fresh = heard.filter { !reply.contains($0) }
            let sinceLoud = Date().timeIntervalSince(self.lastLoud)
            talkLog.notice("under reply heard \(text, privacy: .public) fresh=\(fresh.count) sinceLoud=\(sinceLoud, format: .fixed(precision: 2)) floor=\(self.noiseFloor, format: .fixed(precision: 1))")
            guard sinceLoud < 1.0 else { return }
            if fresh.contains(where: Conversation.stopWords.contains) {
                // "Stop" or "wait": hush, without taking what was heard as a question.
                self.stopSpeaking()
            } else if fresh.count >= 3 && fresh.count * 2 >= heard.count {
                // Echo the recognizer got slightly wrong gives a word or two that aren't in the reply;
                // the user talking gives mostly new words.
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
        if approval != nil {
            // Likely a yes or no to Hermes: its turn stays open for the answer.
            freshSpeech()
        } else {
            speech?.stop()
            speech = nil
            replyTask?.cancel()
            replyTask = nil
        }
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

    static let stopWords: Set<String> = ["stop", "wait", "quiet", "enough", "shush", "pause"]

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
        // The stored copy: one whose reply was just detached may have changed since the list was read.
        self.chat = store.chats.first(where: { $0.id == chat.id }) ?? chat
        resumePendingRun()
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

    // MARK: - Recording a voice to copy

    /// Starts collecting the microphone for "My voice". Nil when the microphone is not available.
    func startSampleRecording() async -> SampleRecorder? {
        guard phase == .idle else { return nil }
        claimMicrophone()
        guard await AVAudioApplication.requestRecordPermission() else {
            error = "Earshot needs the microphone. Allow it for LiveContainer in Settings › Privacy › Microphone."
            return nil
        }
        do {
            try AudioIO.shared.start()
        } catch {
            self.error = "The microphone could not start: \(error.localizedDescription)"
            return nil
        }
        let recorder = SampleRecorder()
        router.attach(recorder, withPreroll: false)
        return recorder
    }

    func stopSampleRecording(_ recorder: SampleRecorder, save: Bool) async throws {
        router.detach()
        stopAudioIfIdle()
        if save { _ = try await recorder.finish() }
    }

    /// Plays the saved "My voice" recording as it is, so the user can hear whether it came out clear.
    /// `done` runs when it ends or is stopped.
    func playMyRecording(done: @escaping () -> Void) {
        guard phase == .idle else { return done() }
        previewQueue?.stop()
        do {
            let file = try AVAudioFile(forReading: VoiceSamples.myFile)
            guard let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)) else {
                return done()
            }
            try file.read(into: buffer)
            try AudioIO.shared.start()
            AudioIO.shared.play(buffer) {
                Task { @MainActor [weak self] in
                    self?.stopAudioIfIdle()
                    done()
                }
            }
        } catch {
            self.error = error.localizedDescription
            done()
        }
    }

    func stopMyRecording() {
        AudioIO.shared.stopPlayback()
    }

    /// Loads an on-device voice while the user is still talking, so the reply does not wait for it.
    /// Also called before a reply is spoken: the engine runs requests in order, so the first sentence
    /// waits behind the load, and `loadingVoice` says so until it is done.
    private func warmUpVoice() {
        guard settings.preset.voice.kind == .onDevice, let voice = OnDeviceVoice(settings.preset.voice.voice),
              LocalModelStore.shared.isDownloaded(voice.model),
              LocalVoiceEngine.shared.loadedModel != voice.model, voiceLoad == nil else { return }
        let name = voice.model.info.name
        loadingVoice = name
        voiceLoad = Task { [weak self] in
            try? await Task.detached(priority: .userInitiated) {
                try await LocalVoiceEngine.shared.prepare(voice.model)
            }.value
            self?.voiceLoad = nil
            self?.loadingVoice = nil
        }
    }

    private var voiceLoad: Task<Void, Never>?

    private func stopAudioIfIdle() {
        if phase == .idle && !otherIsBusy() {
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
        case .hermes:
            let (url, key) = try hermesAccess()
            return OpenAICompatibleBrain(baseURL: url, apiKey: key)
        }
    }

    private func hermesAccess() throws -> (URL, String) {
        guard let url = URL(string: settings.data.hermesURL.trimmingCharacters(in: .whitespacesAndNewlines)), url.scheme != nil else {
            throw BrainError(message: "Set Hermes's address in the Agent tab's settings.")
        }
        guard let key = settings.key(.hermes) else {
            throw BrainError(message: "Add Hermes's key in the Agent tab's settings (the gear at the top).")
        }
        return (url, key)
    }

    private func makeTranscriber() throws -> Transcriber {
        switch settings.data.ears {
        case .apple:
            return AppleTranscriber()
        case .fluid:
            guard LocalModelStore.shared.isDownloaded(FluidSpeechMini.model) else {
                throw BrainError(message: "Fluid Speech Mini isn't downloaded yet. Get it in Models › Ears.")
            }
            return FluidTranscriber()
        case .server:
            guard let url = URL(string: settings.data.speechServerURL.trimmingCharacters(in: .whitespacesAndNewlines)),
                  url.scheme != nil else {
                throw BrainError(message: "Set the speech server's address in Models › Ears.")
            }
            guard let key = settings.key(.speechServer) else {
                throw BrainError(message: "Add the speech server's key in Models › Ears.")
            }
            return UploadTranscriber(baseURL: url, apiKey: key, model: settings.data.speechServerModel,
                                     language: "en", describe: SpeechServer.message)
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
        case .onDevice:
            if let voice = OnDeviceVoice(choice.voice) {
                guard LocalModelStore.shared.isDownloaded(voice.model) else {
                    error = "\(voice.model.info.name) isn't downloaded yet, so Apple's voice is speaking instead."
                    return AppleSynthesizer(identifier: settings.data.appleVoice)
                }
                return OnDeviceSynthesizer(choice: voice, fallback: AppleSynthesizer(identifier: settings.data.appleVoice),
                                           pace: settings.data.mingPace)
            }
        }
        error = "That voice isn't set up yet, so Apple's voice is speaking instead."
        return AppleSynthesizer(identifier: "")
    }

    static func title(from text: String) -> String {
        let clean = text.replacingOccurrences(of: "\n", with: " ")
        return clean.count > 48 ? String(clean.prefix(47)) + "…" : clean
    }

    static func systemPrompt(webAccess: Bool = false, agent: Bool = false) -> String {
        let date = Date().formatted(date: .complete, time: .shortened)
        let web = webAccess ? "\n- You can search the web and open pages. Do it when the answer depends on recent or specific facts you can't be sure of, not for things you already know. Never read web addresses aloud; the app lists your sources on screen." : ""
        // Hermes keeps its own identity and tools; this only adds how to talk through the phone.
        let opening = agent
            ? "The user is talking to you by voice through Earshot, an app on their iPhone. When they ask you to do something, do it with your tools, then say in a sentence or two what you did and what came of it. Don't read out command output, paths or code. Your replies are read aloud, so:"
            : "You are Earshot, a voice assistant on the user's iPhone. Your replies are read aloud, so:"
        return """
        \(opening)
        - Talk the way a person would. Lead with the answer and keep it short: one to three sentences for a simple question, more only when the user asks for detail or the task needs it.
        - Write plain sentences. No headings, tables, bullet lists or bold text.
        - When the user asks you to draft something they will copy or send (a message, an email, a note, code), put only the draft inside a fenced block (```), and say one short sentence before it. The fenced block is shown on screen and is not read aloud.
        - An attached image is usually a screenshot of what was on the user's screen when they asked. Talk about what is in it directly.
        - Speech recognition sometimes mishears words. If a question looks garbled, answer your best guess and say briefly what you assumed.\(web)
        It is \(date). The user's locale is \(Locale.current.identifier).
        """
    }
}

enum Haptics {
    @MainActor static func tap() {
        UIImpactFeedbackGenerator(style: .light).impactOccurred()
    }
}
