// swift-tools-version:6.2
import PackageDescription

// Voices that run on the iPhone itself. The app links the LocalVoices library; voice-check is a small
// macOS tool that CI uses to make each voice speak and to time it, before anything reaches the phone.
let package = Package(
    name: "LocalVoices",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "LocalVoices", targets: ["LocalVoices"]),
        .executable(name: "voice-check", targets: ["VoiceCheck"])
    ],
    dependencies: [
        // Pinned: the model code moves quickly upstream.
        .package(url: "https://github.com/Blaizzy/mlx-audio-swift.git", revision: "fe2c182f4da8115fe2b6b99c9a4d2048fbf9b1e6"),
        .package(url: "https://github.com/ml-explore/mlx-swift.git", .upToNextMajor(from: "0.32.3")),
        // Already in the graph through mlx-audio-swift; named here because Audio8 0.1B loads its tokenizer.
        .package(url: "https://github.com/huggingface/swift-transformers.git", .upToNextMajor(from: "1.1.6"))
    ],
    targets: [
        .target(
            name: "LocalVoices",
            dependencies: [
                .product(name: "MLXAudioTTS", package: "mlx-audio-swift"),
                .product(name: "MLXAudioCore", package: "mlx-audio-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXFFT", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers")
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "VoiceCheck",
            dependencies: ["LocalVoices"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        )
    ]
)
