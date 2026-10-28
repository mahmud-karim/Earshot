import Foundation
import MLXAudioCore

/// Raon-OpenTTS 1B: an F5-TTS style flow-matching DiT with a 16 kHz HiFi-GAN vocoder.
enum RaonLoader {
    static func load(_ folder: URL) async throws -> SpeechGenerationModel {
        throw LocalVoiceError("Raon-OpenTTS is not ported yet.")
    }
}
