# Earshot

A voice assistant for iPhone that you point at any model. Pick the brain (Claude, anything on
OpenRouter or OpenAI, or a model on your own PC), the ears (speech to text) and the voice (text to
speech) separately, and switch between presets in one tap.

Press the Action Button in any app and Earshot takes a screenshot, opens, and listens for your
question about what's on screen.

Earshot is built on Windows with GitHub Actions and runs in [LiveContainer](https://github.com/LiveContainer/LiveContainer).
Your API keys stay in the iPhone's Keychain and are sent only to the provider they belong to.

## Parts

| Part | Choices |
| --- | --- |
| Brain | Claude Haiku 5.5 / Sonnet 5.5 / Opus 5.5 with a thinking level, OpenRouter, OpenAI, Ollama or LM Studio on your PC |
| Ears | Apple on-device speech, OpenAI transcription, Whisper on your PC |
| Voice | Apple voices, OpenAI voices, ElevenLabs, Kokoro on your PC |

Replies stream in and are spoken sentence by sentence while they are still being written. You can
talk over a reply to interrupt it.

## Build

Every push to `main` builds an unsigned IPA (`earshot-ipa` artifact) on a macOS runner. To build on
a Mac: `brew install xcodegen && xcodegen generate`, then open `Earshot.xcodeproj`.

## Action Button

Settings › Shortcuts › Action Button in the app has the steps: a Shortcut that runs Take Screenshot,
Copy to Clipboard, and Open URLs with
`livecontainer://livecontainer-launch?bundle-name=com.mahmudkarim.Earshot.app`.

## License

MIT
