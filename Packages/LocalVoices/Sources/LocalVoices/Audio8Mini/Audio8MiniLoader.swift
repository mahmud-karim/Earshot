import Foundation
import MLXAudioCore

/// Audio8 TTS Preview 0.1B. Its slow stack is Falcon-H1 (Mamba-2, attention and an MLP in every
/// layer), which the upstream Swift Audio8 port does not run.
enum Audio8MiniLoader {
    static func load(_ folder: URL) async throws -> SpeechGenerationModel {
        throw LocalVoiceError("Audio8 0.1B is not ported yet.")
    }
}
