// Adapted from mlx-audio-swift (MIT, Prince Canuma and contributors), Sources/MLXAudioTTS/Models/Audio8/
// Audio8Config.swift at fe2c182, with the Falcon-H1 fields of mlx-audio's arktts.py (MIT).

import Foundation

/// config.json of Audio8 TTS Preview 0.1B. Only the fields this runtime reads are decoded; the
/// Falcon-H1 names mirror the checkpoint so they read the same as arktts.py.
struct Audio8MiniConfig: Decodable {
    var modelType = "arktts"
    var slowBackbone = "arktts"
    var vocabSize = 69633
    var dim = 512
    var nLayer = 24
    var nHead = 8
    var nLocalHeads = 2
    var headDim = 64
    var intermediateSize = 768
    var maxSeqLen = 2048
    var ropeBase: Float = 1e11
    var normEps: Float = 1e-5
    var codebookSize = 4096
    var numCodebooks = 10
    var semanticBeginID = 65537
    var semanticEndID = 69632
    var nFastLayer = 4
    var fastDim = 512
    var fastNHead = 8
    var fastNLocalHeads = 2
    var fastHeadDim = 64
    var fastIntermediateSize = 4864
    var fastAttentionQKVBias = false
    var fastAttentionOBias = false
    var codecFilename = "codec.safetensors"
    var codecSampleRate = 44100
    var codecFrameSize = 2048
    var codecPostNLayer = 8
    var codecPostNHead = 16
    var codecPostNLocalHeads = 8
    var codecPostIntermediateSize = 1216
    var rasWindowSize = 10
    var rasTemperature: Float = 1
    var rasTopP: Float = 0.9
    var eosTokenID = 228

    // Falcon-H1 slow stack.
    var attentionBias = false
    var attentionInMultiplier: Float = 1
    var attentionOutMultiplier: Float = 1
    var keyMultiplier: Float = 1
    var embeddingMultiplier: Float = 1
    var mlpBias = false
    var mlpMultipliers: [Float] = [1, 1]
    var projectorsBias = false
    var ssmInMultiplier: Float = 1
    var ssmMultipliers: [Float] = [1, 1, 1, 1, 1]
    var ssmOutMultiplier: Float = 1
    var mambaChunkSize = 128
    var mambaConvBias = true
    var mambaDConv = 4
    var mambaDHead = 32
    var mambaDSSM = 768
    var mambaDState = 64
    var mambaNGroups = 1
    var mambaNHeads = 24
    var mambaProjBias = false
    var mambaRMSNorm = false

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type", slowBackbone = "slow_backbone", vocabSize = "vocab_size", dim
        case nLayer = "n_layer", nHead = "n_head", nLocalHeads = "n_local_heads", headDim = "head_dim"
        case intermediateSize = "intermediate_size", maxSeqLen = "max_seq_len", ropeBase = "rope_base"
        case normEps = "norm_eps", codebookSize = "codebook_size", numCodebooks = "num_codebooks"
        case semanticBeginID = "semantic_begin_id", semanticEndID = "semantic_end_id"
        case nFastLayer = "n_fast_layer", fastDim = "fast_dim", fastNHead = "fast_n_head"
        case fastNLocalHeads = "fast_n_local_heads", fastHeadDim = "fast_head_dim"
        case fastIntermediateSize = "fast_intermediate_size"
        case fastAttentionQKVBias = "fast_attention_qkv_bias", fastAttentionOBias = "fast_attention_o_bias"
        case codecFilename = "codec_filename", codecSampleRate = "codec_sample_rate"
        case codecFrameSize = "codec_frame_size", codecPostNLayer = "codec_post_n_layer"
        case codecPostNHead = "codec_post_n_head", codecPostNLocalHeads = "codec_post_n_local_heads"
        case codecPostIntermediateSize = "codec_post_intermediate_size"
        case rasWindowSize = "ras_window_size", rasTemperature = "ras_temperature", rasTopP = "ras_top_p"
        case eosTokenID = "eos_token_id"
        case attentionBias = "attention_bias", attentionInMultiplier = "attention_in_multiplier"
        case attentionOutMultiplier = "attention_out_multiplier", keyMultiplier = "key_multiplier"
        case embeddingMultiplier = "embedding_multiplier", mlpBias = "mlp_bias"
        case mlpMultipliers = "mlp_multipliers", projectorsBias = "projectors_bias"
        case ssmInMultiplier = "ssm_in_multiplier", ssmMultipliers = "ssm_multipliers"
        case ssmOutMultiplier = "ssm_out_multiplier", mambaChunkSize = "mamba_chunk_size"
        case mambaConvBias = "mamba_conv_bias", mambaDConv = "mamba_d_conv", mambaDHead = "mamba_d_head"
        case mambaDSSM = "mamba_d_ssm", mambaDState = "mamba_d_state", mambaNGroups = "mamba_n_groups"
        case mambaNHeads = "mamba_n_heads", mambaProjBias = "mamba_proj_bias", mambaRMSNorm = "mamba_rms_norm"
    }

    init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func read<T: Decodable>(_ key: CodingKeys, _ value: inout T) throws {
            if let decoded = try c.decodeIfPresent(T.self, forKey: key) { value = decoded }
        }
        try read(.modelType, &modelType); try read(.slowBackbone, &slowBackbone)
        try read(.vocabSize, &vocabSize); try read(.dim, &dim); try read(.nLayer, &nLayer)
        try read(.nHead, &nHead); try read(.nLocalHeads, &nLocalHeads); try read(.headDim, &headDim)
        try read(.intermediateSize, &intermediateSize); try read(.maxSeqLen, &maxSeqLen)
        try read(.ropeBase, &ropeBase); try read(.normEps, &normEps)
        try read(.codebookSize, &codebookSize); try read(.numCodebooks, &numCodebooks)
        try read(.semanticBeginID, &semanticBeginID); try read(.semanticEndID, &semanticEndID)
        try read(.nFastLayer, &nFastLayer); try read(.fastDim, &fastDim); try read(.fastNHead, &fastNHead)
        try read(.fastNLocalHeads, &fastNLocalHeads); try read(.fastHeadDim, &fastHeadDim)
        try read(.fastIntermediateSize, &fastIntermediateSize)
        try read(.fastAttentionQKVBias, &fastAttentionQKVBias); try read(.fastAttentionOBias, &fastAttentionOBias)
        try read(.codecFilename, &codecFilename); try read(.codecSampleRate, &codecSampleRate)
        try read(.codecFrameSize, &codecFrameSize); try read(.codecPostNLayer, &codecPostNLayer)
        try read(.codecPostNHead, &codecPostNHead); try read(.codecPostNLocalHeads, &codecPostNLocalHeads)
        try read(.codecPostIntermediateSize, &codecPostIntermediateSize)
        try read(.rasWindowSize, &rasWindowSize); try read(.rasTemperature, &rasTemperature)
        try read(.rasTopP, &rasTopP); try read(.eosTokenID, &eosTokenID)
        try read(.attentionBias, &attentionBias); try read(.attentionInMultiplier, &attentionInMultiplier)
        try read(.attentionOutMultiplier, &attentionOutMultiplier); try read(.keyMultiplier, &keyMultiplier)
        try read(.embeddingMultiplier, &embeddingMultiplier); try read(.mlpBias, &mlpBias)
        try read(.mlpMultipliers, &mlpMultipliers); try read(.projectorsBias, &projectorsBias)
        try read(.ssmInMultiplier, &ssmInMultiplier); try read(.ssmMultipliers, &ssmMultipliers)
        try read(.ssmOutMultiplier, &ssmOutMultiplier); try read(.mambaChunkSize, &mambaChunkSize)
        try read(.mambaConvBias, &mambaConvBias); try read(.mambaDConv, &mambaDConv)
        try read(.mambaDHead, &mambaDHead); try read(.mambaDSSM, &mambaDSSM); try read(.mambaDState, &mambaDState)
        try read(.mambaNGroups, &mambaNGroups); try read(.mambaNHeads, &mambaNHeads)
        try read(.mambaProjBias, &mambaProjBias); try read(.mambaRMSNorm, &mambaRMSNorm)
    }

    /// The width of the compact semantic head: one column per semantic code, then EOS.
    var semanticEOS: Int { codebookSize }

    func validate() throws {
        guard modelType == "arktts", slowBackbone == "falcon_h1" else {
            throw LocalVoiceError("This is not the Audio8 0.1B checkpoint (model_type \(modelType), slow_backbone \(slowBackbone)).")
        }
        // Upstream Falcon-H1 folds these muP multipliers into the weights. arktts.py does not, and only
        // accepts checkpoints that carry them at 1, so this port does the same rather than run unscaled.
        let scalars = [attentionInMultiplier, attentionOutMultiplier, keyMultiplier, ssmInMultiplier, ssmOutMultiplier]
        guard (scalars + mlpMultipliers + ssmMultipliers).allSatisfy({ $0 == 1 }) else {
            throw LocalVoiceError("Audio8 0.1B with muP multipliers other than 1 is not supported.")
        }
        guard !mambaRMSNorm else { throw LocalVoiceError("Audio8 0.1B with a gated Mamba norm is not supported.") }
        guard dim > 0, nLayer > 0, nHead % nLocalHeads == 0, headDim % 2 == 0,
              fastNHead % fastNLocalHeads == 0, fastHeadDim % 2 == 0, fastDim == dim,
              mambaNHeads * mambaDHead == mambaDSSM, mambaNHeads % mambaNGroups == 0,
              codebookSize == 4096, numCodebooks == 10, semanticEndID - semanticBeginID + 1 == codebookSize,
              semanticEndID < vocabSize, codecSampleRate == 44100, codecFrameSize == 2048, rasWindowSize > 0,
              codecFilename.hasSuffix(".safetensors"),
              codecFilename == URL(fileURLWithPath: codecFilename).lastPathComponent else {
            throw LocalVoiceError("Audio8 0.1B config.json has values this runtime does not support.")
        }
    }
}
