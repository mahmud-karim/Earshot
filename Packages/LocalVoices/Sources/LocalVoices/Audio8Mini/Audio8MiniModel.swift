// Adapted from mlx-audio-swift (MIT, Prince Canuma and contributors), Sources/MLXAudioTTS/Models/Audio8/
// Audio8Model.swift and Audio8Transformer.swift at fe2c182, following every `uses_falcon_slow` branch of
// mlx-audio's arktts.py (MIT) for the 0.1B checkpoint.

import Foundation
@preconcurrency import MLX
@preconcurrency import MLXAudioCore
import MLXAudioTTS
import MLXNN
import MLXRandom
import Tokenizers

/// The DualAR network: the Falcon-H1 slow stack predicts one semantic token per audio frame, and the
/// attention-only fast stack fills in that frame's nine residual codebooks.
final class Audio8MiniBackbone: Module {
    let config: Audio8MiniConfig
    @ModuleInfo(key: "codebook_embeddings") var codebookEmbeddings: Embedding
    let slow: Audio8MiniFalconStack
    /// The 0.1B's own semantic head. Its logits are COMPACT: column i is semantic code i, and column
    /// `codebookSize` is end of speech. (The 0.6B instead reuses its full text-vocabulary embedding.)
    @ModuleInfo(key: "semantic_output") var semanticOutput: Linear
    @ModuleInfo(key: "fast_embeddings") var fastEmbeddings: Embedding
    @ModuleInfo(key: "fast_layers") var fastLayers: [Audio8MiniBlock]
    @ModuleInfo(key: "fast_norm") var fastNorm: Audio8MiniNorm
    @ModuleInfo(key: "fast_output") var fastOutput: Linear
    let fastRoPE: Audio8MiniRoPE

    init(_ c: Audio8MiniConfig) {
        config = c
        _codebookEmbeddings.wrappedValue = Embedding(embeddingCount: c.codebookSize * c.numCodebooks, dimensions: c.dim)
        slow = Audio8MiniFalconStack(c)
        _semanticOutput.wrappedValue = Linear(c.dim, c.codebookSize + 1, bias: false)
        _fastEmbeddings.wrappedValue = Embedding(embeddingCount: c.codebookSize, dimensions: c.fastDim)
        _fastLayers.wrappedValue = (0..<c.nFastLayer).map { _ in
            Audio8MiniBlock(dim: c.fastDim, intermediate: c.fastIntermediateSize, heads: c.fastNHead,
                            kvHeads: c.fastNLocalHeads, headDim: c.fastHeadDim, qkvBias: c.fastAttentionQKVBias,
                            outputBias: c.fastAttentionOBias, eps: c.normEps)
        }
        _fastNorm.wrappedValue = Audio8MiniNorm(c.fastDim, eps: c.normEps)
        _fastOutput.wrappedValue = Linear(c.fastDim, c.codebookSize, bias: false)
        fastRoPE = Audio8MiniRoPE(length: c.numCodebooks, dim: c.fastHeadDim, base: c.ropeBase)
    }

    /// Text embedding plus, on semantic positions, the sum of the ten codebook embeddings. Unscaled:
    /// the slow stack applies `embedding_multiplier` to the whole sum.
    func embed(_ ids: MLXArray) -> MLXArray {
        let c = config, tokens = ids[0..., 0, 0...]
        let books = (0..<c.numCodebooks).map {
            codebookEmbeddings(ids[0..., $0 + 1, 0...] + Int32($0 * c.codebookSize))
        }
        let total = stacked(books, axis: 1).sum(axis: 1)
        let semantic = (tokens .>= c.semanticBeginID) .&& (tokens .<= c.semanticEndID)
        return slow.embedTokens(tokens) + which(semantic.expandedDimensions(axis: -1), total, 0)
    }

    /// Feeds `ids` (1, 11, T) through the slow stack. Returns the compact semantic logits of the last
    /// position and its normalized hidden state, which on this variant is also the fast stack's input.
    func slowStep(_ ids: MLXArray, cache: [Audio8MiniFalconCache]) -> (logits: MLXArray, hidden: MLXArray) {
        let normalized = slow(embed(ids), cache: cache)
        let last = normalized[0..., (normalized.dim(1) - 1)..., 0...]
        return (semanticOutput(last)[0..., -1, 0...], last)
    }

    func fastStep(_ h: MLXArray, position: Int, cache: [Audio8MiniKVCache]) -> MLXArray {
        var x = h
        for (layer, layerCache) in zip(fastLayers, cache) {
            x = layer(x, rope: fastRoPE, offset: position, mask: .none, cache: layerCache)
        }
        return fastOutput(fastNorm(x))[0..., -1, 0...]
    }
}

/// Legacy Audio8 top-k/top-p: filter before temperature, and nucleus filtering drops the token that
/// crosses `topP` while always keeping the best one.
private func audio8MiniFilteredScores(_ logits: MLXArray, topK: Int, topP: Float, temperature: Float) -> MLXArray {
    let count = logits.shape.last!
    let indices = argSort(-logits, axis: -1)
    let sorted = takeAlong(logits, indices, axis: -1)
    let ranks = MLX.arange(count).reshaped([1, count])
    let cumulative = cumsum(softmax(sorted, axis: -1), axis: -1)
    let remove = ((cumulative .> topP) .|| (ranks .>= max(1, min(topK, count)))) .&& (ranks .> 0)
    let mask = putAlong(MLXArray.zeros(remove.shape, dtype: .bool), indices, values: remove, axis: -1)
    return which(mask, MLXArray(-Float.infinity).asType(logits.dtype), logits) / max(temperature, 1e-5)
}

/// Exponential-race sampling, the reference's way of drawing from softmax(scores).
private func audio8MiniRaceSample(_ scores: MLXArray, key: MLXArray) -> MLXArray {
    let uniform = MLXRandom.uniform(low: 0.0, high: 1.0, scores.shape, key: key)
    return argMax(softmax(scores, axis: -1) / -log(uniform), axis: -1).asType(.int32)
}

/// Carries a non-Sendable value into a task that is its only user.
private final class Audio8MiniHandOff<Value>: @unchecked Sendable {
    private var value: Value?
    init(_ value: Value) { self.value = value }
    func take() -> Value { defer { value = nil }; return value! }
}

final class Audio8MiniModel: Module, SpeechGenerationModel, @unchecked Sendable {
    let config: Audio8MiniConfig
    let model: Audio8MiniBackbone
    let codec: Audio8MiniCodec
    private var tokenizer: (any Tokenizer)?

    var sampleRate: Int { config.codecSampleRate }
    var defaultGenerationParameters: GenerateParameters {
        .init(maxTokens: 512, temperature: 0.7, topP: 0.9, topK: 50)
    }

    init(config: Audio8MiniConfig) {
        self.config = config
        model = Audio8MiniBackbone(config)
        codec = Audio8MiniCodec(config)
    }

    // MARK: Prompt

    static func cleanText(_ text: String) -> String { text.split(whereSeparator: \.isWhitespace).joined(separator: " ") }

    static func referenceText(_ text: String) -> String {
        let cleaned = cleanText(text)
        return cleaned.range(of: #"<\|speaker:\d+\|>"#, options: .regularExpression) != nil
            ? cleaned : "<|speaker:0|>" + cleaned
    }

    /// The chat-style prompt around the reference codes: (before the codes, after the codes).
    func promptSegments(text: String, referenceText: String?, hasReference: Bool) throws -> ([Int], [Int]) {
        guard let tokenizer else { throw LocalVoiceError("The Audio8 tokenizer is not loaded.") }
        let target = Self.cleanText(text)
        guard !target.isEmpty else { throw LocalVoiceError("There is no text to speak.") }
        func encode(_ parts: [String]) -> [Int] { parts.flatMap { tokenizer.encode(text: $0, addSpecialTokens: false) } }
        let suffix = ["<|im_end|>\n", "<|im_start|>user\n", target, "<|im_end|>\n", "<|im_start|>assistant\n<|voice|>"]
        if !hasReference {
            return (encode(["<|im_start|>system\n", "convert the provided text to speech"] + suffix), [])
        }
        guard let referenceText, !Self.cleanText(referenceText).isEmpty else {
            throw LocalVoiceError("Audio8 needs the words of the reference recording.")
        }
        return (encode(["<|im_start|>system\n",
                        "convert the provided text to speech reference to the following:\n\nText:\n",
                        Self.referenceText(referenceText), "\n\nSpeech:\n"]), encode(suffix))
    }

    /// Rows: 0 holds text ids or full-vocabulary semantic ids, 1...10 the codebooks of reference frames.
    func preparePrompt(prefix: [Int], suffix: [Int], referenceCodes: MLXArray?) -> MLXArray {
        let frames = referenceCodes?.shape[2] ?? 0
        let length = prefix.count + frames + suffix.count
        var rows = Array(repeating: Int32(0), count: (config.numCodebooks + 1) * length)
        for (i, token) in prefix.enumerated() { rows[i] = Int32(token) }
        for (i, token) in suffix.enumerated() { rows[prefix.count + frames + i] = Int32(token) }
        if let referenceCodes {
            let codes = referenceCodes.asArray(Int32.self)
            for i in 0..<frames {
                rows[prefix.count + i] = codes[i] + Int32(config.semanticBeginID)
                for q in 0..<config.numCodebooks { rows[(q + 1) * length + prefix.count + i] = codes[q * frames + i] }
            }
        }
        return MLXArray(rows).reshaped([1, config.numCodebooks + 1, length])
    }

    // MARK: Sampling

    /// The semantic draw with repetition-aware sampling: when the normal draw repeats one of the last
    /// `ras_window_size` semantic codes, a second draw at the RAS temperature and top-p replaces it.
    /// The compact head needs no semantic-range filter; every column but EOS is a semantic code.
    private func sampleSemantic(_ logits: MLXArray, p: GenerateParameters, keys: (MLXArray, MLXArray),
                                previous: MLXArray?) -> MLXArray {
        let scores = audio8MiniFilteredScores(logits, topK: p.topK, topP: p.topP, temperature: p.temperature)
        let normal = audio8MiniRaceSample(scores, key: keys.0)
        guard let previous else { return normal }
        let high = audio8MiniFilteredScores(logits, topK: p.topK, topP: config.rasTopP, temperature: config.rasTemperature)
        let rescue = audio8MiniRaceSample(high, key: keys.1)
        let repeated = any(previous .== normal.expandedDimensions(axis: 1), axis: 1)
        return which(repeated .&& (normal .< config.codebookSize), rescue, normal)
    }

    /// The frame's codebooks: codebook 0 is the semantic code itself, the other nine come from the fast stack.
    private func generateCodebooks(hidden: MLXArray, semantic: MLXArray, p: GenerateParameters, key: MLXArray) -> MLXArray {
        let cache = model.fastLayers.map { _ in Audio8MiniKVCache() }
        _ = model.fastStep(hidden, position: 0, cache: cache)
        var current = clip(semantic, min: 0, max: config.codebookSize - 1)
        var codes = [current]
        var key = key
        for position in 1..<config.numCodebooks {
            let logits = model.fastStep(model.fastEmbeddings(current).expandedDimensions(axis: 1), position: position, cache: cache)
            let scores = audio8MiniFilteredScores(logits, topK: p.topK, topP: p.topP, temperature: p.temperature)
            let (next, sub) = MLXRandom.split(key: key)
            key = next
            current = audio8MiniRaceSample(scores, key: sub)
            codes.append(current)
        }
        return stacked(codes, axis: 1)
    }

    /// Generates codec codes (1, 10, frames) for one prompt.
    func generateCodes(prompt: MLXArray, parameters p: GenerateParameters) throws -> MLXArray {
        let width = prompt.shape[2]
        guard width < config.maxSeqLen else { throw LocalVoiceError("The Audio8 prompt is too long; use a shorter reference.") }
        guard p.topK > 0, p.topP > 0, p.topP <= 1, p.temperature.isFinite, p.temperature >= 0 else {
            throw LocalVoiceError("Audio8 needs a positive top-k and a top-p in (0, 1].")
        }
        let budget = min(p.maxTokens ?? 512, config.maxSeqLen - width)
        let cache = model.slow.makeCache()
        var key = MLXRandom.key(p.seed ?? UInt64.random(in: 0...UInt64.max))
        let traceStart = Date()
        var (logits, hidden) = model.slowStep(prompt, cache: cache)
        eval(logits, hidden)
        print(String(format: "TRACE prefill %ld tokens %.3f s, peak %ld MB", width, Date().timeIntervalSince(traceStart), Memory.peakMemory / 1_000_000))
        let loopStart = Date()
        defer { print(String(format: "TRACE loop %.3f s, peak %ld MB", Date().timeIntervalSince(loopStart), Memory.peakMemory / 1_000_000)) }
        var previous: MLXArray?
        var frames: [MLXArray] = []
        func drawSemantic() -> (semantic: MLXArray, fastKey: MLXArray) {
            let keys = MLXRandom.split(key: key, into: 4)
            key = keys[3]
            return (sampleSemantic(logits, p: p, keys: (keys[0], keys[1]), previous: previous), keys[2])
        }
        var pending = drawSemantic()
        asyncEval(pending.semantic)
        for step in 0..<budget {
            try Task.checkCancellation()
            let (semantic, fastKey) = pending
            let frame = generateCodebooks(hidden: hidden, semantic: semantic, p: p, key: fastKey)
            // Like the reference, the history starts as zeros after the first frame and only then
            // takes in semantic codes.
            if let window = previous {
                previous = concatenated([window[0..., 1...], semantic.expandedDimensions(axis: 1)], axis: 1)
            } else {
                previous = MLXArray.zeros([1, config.rasWindowSize], dtype: .int32)
            }
            let more = step + 1 < budget
            if more {
                // Queue the next frame before reading this one's token, so the GPU never waits for
                // the CPU to build a graph. If this token turns out to be EOS, that work is dropped.
                // The prompt rows are full-vocabulary ids, so the slow stack gets the semantic code
                // back in that space, and EOS as the tokenizer's eos_token_id (as arktts.py does);
                // left as is, EOS would index one past the end of the text embedding.
                let fullID = which(semantic .== config.semanticEOS, MLXArray(Int32(config.eosTokenID)),
                                   semantic + Int32(config.semanticBeginID))
                let column = concatenated([fullID.reshaped([1, 1]), frame], axis: 1)
                (logits, hidden) = model.slowStep(column.expandedDimensions(axis: 2), cache: cache)
                pending = drawSemantic()
                asyncEval(pending.semantic)
            }
            // In the compact space EOS is the head's last column, not the tokenizer's eos_token_id.
            if semantic.item(Int32.self) == Int32(config.semanticEOS) { break }
            frames.append(frame)
            if !more { break }
        }
        guard !frames.isEmpty else { throw LocalVoiceError("Audio8 made no audio for this text.") }
        return stacked(frames, axis: 2)
    }

    // MARK: Generation

    private func run(text: String, reference: MLXArray?, transcript: String?,
                     parameters: GenerateParameters) throws -> (audio: MLXArray, promptLength: Int, frames: Int) {
        let (prefix, suffix) = try promptSegments(text: text, referenceText: transcript, hasReference: reference != nil)
        var referenceCodes: MLXArray?
        if let reference {
            guard reference.size > 0, all(isFinite(reference)).item(Bool.self) else {
                throw LocalVoiceError("The reference recording is empty or damaged.")
            }
            referenceCodes = codec.encode(reference.asType(.float32))
            eval(referenceCodes!)
        }
        let prompt = preparePrompt(prefix: prefix, suffix: suffix, referenceCodes: referenceCodes)
        let codes = try generateCodes(prompt: prompt, parameters: parameters)
        eval(codes)
        let decodeStart = Date()
        let audio = codec.decode(codes)
        eval(audio)
        print(String(format: "TRACE decode %ld frames %.3f s, peak %ld MB, active %ld MB", codes.shape[2], Date().timeIntervalSince(decodeStart), Memory.peakMemory / 1_000_000, Memory.activeMemory / 1_000_000))
        return (audio, prompt.shape[2], codes.shape[2])
    }

    func generate(text: String, voice: String?, refAudio: sending MLXArray?, refText: String?, language: String?,
                  generationParameters: GenerateParameters) async throws -> sending MLXArray {
        let (audio, _, _) = try run(text: text, reference: refAudio, transcript: refText, parameters: generationParameters)
        nonisolated(unsafe) let result = audio
        return result
    }

    /// Generation makes the whole waveform at once, so the stream yields one chunk.
    func generateStream(text: String, voice: String?, refAudio: sending MLXArray?, refText: String?, language: String?,
                        generationParameters: GenerateParameters) -> sending AsyncThrowingStream<AudioGeneration, Error> {
        let reference = Audio8MiniHandOff(refAudio)
        return AsyncThrowingStream { continuation in
            let task = Task { @Sendable [self] in
                do {
                    let start = Date()
                    let (audio, prompt, frames) = try run(text: text, reference: reference.take(), transcript: refText,
                                                          parameters: generationParameters)
                    let elapsed = Date().timeIntervalSince(start)
                    let info = AudioGenerationInfo(
                        promptTokenCount: prompt, generationTokenCount: frames, prefillTime: 0, generateTime: elapsed,
                        tokensPerSecond: Double(frames) / max(elapsed, 1e-6), peakMemoryUsage: Double(Memory.peakMemory) / 1e9)
                    nonisolated(unsafe) let output = audio
                    if case .terminated = continuation.yield(.audio(output)) { throw CancellationError() }
                    continuation.yield(.info(info))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    // MARK: Loading

    /// The codec's quantizer keeps its resample stages one level deeper than the checkpoint names them.
    static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        Dictionary(uniqueKeysWithValues: weights.map { key, value in
            var parts = key.split(separator: ".").map(String.init)
            if parts.count > 4, parts[0] == "codec", parts[1] == "quantizer",
               ["downsample", "upsample"].contains(parts[2]) { parts.insert("layers", at: 4) }
            return (parts.joined(separator: "."), value)
        })
    }

    static func fromModelDirectory(_ directory: URL) async throws -> Audio8MiniModel {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(Audio8MiniConfig.self, from: data)
        try config.validate()
        let model = Audio8MiniModel(config: config)
        var weights = try loadArrays(url: directory.appendingPathComponent("model.safetensors"))
        for (key, value) in try loadArrays(url: directory.appendingPathComponent(config.codecFilename)) {
            guard weights[key] == nil else { throw LocalVoiceError("Audio8 has the weight \(key) twice.") }
            weights[key] = value
        }
        try model.update(parameters: ModuleParameters.unflattened(sanitize(weights)), verify: .all)
        eval(model)
        print(String(format: "TRACE loaded, active %ld MB, peak %ld MB", Memory.activeMemory / 1_000_000, Memory.peakMemory / 1_000_000))
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        for (token, expected) in [("<|semantic:0|>", config.semanticBeginID),
                                  ("<|semantic:\(config.codebookSize - 1)|>", config.semanticEndID)] {
            guard tokenizer.convertTokenToId(token) == expected else {
                throw LocalVoiceError("The Audio8 tokenizer does not match the model (\(token)).")
            }
        }
        model.tokenizer = tokenizer
        return model
    }
}
