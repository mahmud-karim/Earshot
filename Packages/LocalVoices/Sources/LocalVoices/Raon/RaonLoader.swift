import Foundation
@preconcurrency import MLX
import MLXAudioTTS

/// Raon-OpenTTS 1B: an F5-TTS style flow-matching DiT with a 16 kHz HiFi-GAN vocoder.
///
/// The folder holds the ComfyUI repack's `config.yaml`, `vocab.txt` and bf16 safetensors at the top,
/// and speechbrain's `generator.ckpt` under `vocoder/`.
enum RaonLoader {
    static func load(_ folder: URL) async throws -> SpeechGenerationModel {
        let files = try FileManager.default.contentsOfDirectory(atPath: folder.path)
        guard let weightsName = files.filter({ $0.hasSuffix(".safetensors") }).sorted().first else {
            throw LocalVoiceError("Raon's weights are missing.")
        }
        let config = try RaonDiTConfig(yaml: String(contentsOf: folder.appendingPathComponent("config.yaml"), encoding: .utf8))

        // Computed in float16, as the official CUDA inference does: every Apple GPU runs it at full speed,
        // while bfloat16 is emulated on older ones.
        let dit = try RaonDiT(weights: loadArrays(url: folder.appendingPathComponent(weightsName)),
                              config: config, dtype: .float16)
        // A few tensors at a time, so each bf16 original is freed as soon as its float16 copy exists.
        let arrays = dit.arrays
        for start in stride(from: 0, to: arrays.count, by: 8) {
            eval(Array(arrays[start ..< min(start + 8, arrays.count)]))
        }

        let vocoderWeights = try TorchStateDict.load(folder.appendingPathComponent("vocoder/generator.ckpt"))
        let vocoder = try RaonHiFiGAN(weights: vocoderWeights, dtype: .float32)
        eval(vocoder.arrays)
        let tokenizer = try RaonTokenizer(vocabURL: folder.appendingPathComponent("vocab.txt"),
                                          usableCount: dit.textRows - 1)
        let model = RaonTTS(dit: dit, vocoder: vocoder, tokenizer: tokenizer)
        model.warmUp()
        Memory.clearCache()
        return model
    }
}
