import Foundation
import MLXAudioCore
import MLXAudioTTS

/// Audio8 TTS Preview 0.1B. Its slow stack is Falcon-H1 (Mamba-2, attention and an MLP in every
/// layer), which the upstream Swift Audio8 port does not run, so this folder carries its own runtime:
/// a port of mlx-audio's arktts.py on top of copies of the upstream Swift Audio8 and Falcon-H1 code.
enum Audio8MiniLoader {
    static func load(_ folder: URL) async throws -> SpeechGenerationModel {
        try await Audio8MiniModel.fromModelDirectory(folder)
    }
}
