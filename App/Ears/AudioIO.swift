import AVFoundation

/// One audio engine for both directions. The microphone runs with voice processing, which cancels the
/// app's own speech from what it hears, so the user can talk over a reply. Every voice plays through
/// the same engine for that reason: echo cancellation only knows about audio it plays itself.
final class AudioIO {
    static let shared = AudioIO()

    /// 16 kHz mono samples and their level in dBFS. Called on the audio thread.
    var onSamples: (([Float], Float) -> Void)?

    let engine = AVAudioEngine()
    let player = AVAudioPlayerNode()
    /// What the player takes; every voice is converted to it.
    let playFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 24_000, channels: 1, interleaved: false)!
    let speechFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false)!

    private(set) var isRunning = false
    /// The session stays active between turns while the app is in front: an engine started right
    /// after the session is activated again hears only silence.
    private var sessionActive = false
    private var isWired = false
    private var converter: AVAudioConverter?
    private var monoFormat: AVAudioFormat?

    func start() throws {
        guard !isRunning else { return }
        if !sessionActive {
            let session = AVAudioSession.sharedInstance()
            // Default mode, not voiceChat: voiceChat makes iOS treat the app as a call in progress.
            // Echo cancellation comes from voice processing on the engine below, in either mode.
            try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker, .allowBluetoothHFP])
            try session.setActive(true)
            sessionActive = true
        }

        if !isWired {
            // Voice processing changes the formats of both I/O nodes, so it comes before any wiring.
            try? engine.inputNode.setVoiceProcessingEnabled(true)
            engine.attach(player)
            engine.connect(player, to: engine.mainMixerNode, format: playFormat)
            isWired = true
            // Turning voice processing on, a route change or an interruption reconfigures the I/O
            // and stops the engine without saying so to the tap. Start it again.
            NotificationCenter.default.addObserver(forName: .AVAudioEngineConfigurationChange, object: engine, queue: .main) { [weak self] _ in
                self?.restartAfterChange()
            }
        }

        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else {
            throw BrainError(message: "No microphone is available.")
        }
        let mono = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: format.sampleRate, channels: 1, interleaved: false)!
        monoFormat = mono
        converter = AVAudioConverter(from: mono, to: speechFormat)

        input.removeTap(onBus: 0)
        input.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, _ in
            self?.handle(buffer)
        }
        engine.prepare()
        try engine.start()
        isRunning = true
    }

    private func restartAfterChange() {
        guard isRunning else { return }
        restart()
    }

    /// Stops and starts the engine again, keeping the session. For a microphone that delivers only
    /// silence, which iOS does to recording that began before the app was fully in front.
    func restart() {
        guard isRunning else { return }
        isRunning = false
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        try? start()
    }

    /// Stops the microphone and playback. The session stays active for the next turn.
    func stop() {
        guard isRunning else { return }
        player.stop()
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        isRunning = false
    }

    /// For the background: stops everything and lets other apps' audio come back.
    func suspend() {
        stop()
        guard sessionActive else { return }
        sessionActive = false
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }

    // MARK: - Playback

    /// Plays a buffer after whatever is already scheduled. `done` runs once it has been heard, or
    /// when playback is stopped.
    func play(_ buffer: AVAudioPCMBuffer, done: @escaping () -> Void) {
        guard isRunning, let converted = AudioIO.convert(buffer, to: playFormat) else {
            done()
            return
        }
        player.scheduleBuffer(converted, completionCallbackType: .dataPlayedBack) { _ in done() }
        if !player.isPlaying {
            player.play()
        }
    }

    func stopPlayback() {
        player.stop()
    }

    // MARK: - Capture

    private func handle(_ buffer: AVAudioPCMBuffer) {
        guard let channels = buffer.floatChannelData, let mono = monoFormat, let converter else { return }
        let frames = Int(buffer.frameLength)
        guard frames > 0, let single = AVAudioPCMBuffer(pcmFormat: mono, frameCapacity: AVAudioFrameCount(frames)) else { return }
        single.frameLength = AVAudioFrameCount(frames)
        single.floatChannelData![0].update(from: channels[0], count: frames)

        var sum: Float = 0
        for index in 0..<frames {
            let value = channels[0][index]
            sum += value * value
        }
        let rms = sqrt(sum / Float(frames))
        let level = 20 * log10(max(rms, 0.000_01))

        let ratio = speechFormat.sampleRate / mono.sampleRate
        let capacity = AVAudioFrameCount(Double(frames) * ratio) + 32
        guard let output = AVAudioPCMBuffer(pcmFormat: speechFormat, frameCapacity: capacity) else { return }
        var given = false
        var error: NSError?
        converter.convert(to: output, error: &error) { _, status in
            if given {
                status.pointee = .noDataNow
                return nil
            }
            given = true
            status.pointee = .haveData
            return single
        }
        guard error == nil, let data = output.floatChannelData else { return }
        let samples = Array(UnsafeBufferPointer(start: data[0], count: Int(output.frameLength)))
        onSamples?(samples, level)
    }

    /// A whole buffer in another format: resampled, and mixed down to mono when needed.
    static func convert(_ buffer: AVAudioPCMBuffer, to format: AVAudioFormat) -> AVAudioPCMBuffer? {
        if buffer.format == format { return buffer }
        guard buffer.frameLength > 0, let converter = AVAudioConverter(from: buffer.format, to: format) else { return nil }
        converter.downmix = true
        let ratio = format.sampleRate / buffer.format.sampleRate
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 1024
        guard let output = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: capacity) else { return nil }
        var given = false
        var error: NSError?
        let status = converter.convert(to: output, error: &error) { _, inputStatus in
            if given {
                inputStatus.pointee = .endOfStream
                return nil
            }
            given = true
            inputStatus.pointee = .haveData
            return buffer
        }
        guard status != .error, error == nil, output.frameLength > 0 else { return nil }
        return output
    }
}
