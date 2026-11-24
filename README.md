<p align="center">
  <img src="App/Resources/Assets.xcassets/AppIcon.appiconset/AppIcon.png" width="120" alt="Earshot icon">
</p>

<h1 align="center">Earshot</h1>

<p align="center">
  <b>An open-source replacement for Siri that you build out of the models you choose.</b><br>
  Any brain. Private on-device ears and voice. A real agent behind it when you need one.
</p>

<p align="center">
  <img alt="Platform" src="https://img.shields.io/badge/platform-iOS%2026-black">
  <img alt="Swift" src="https://img.shields.io/badge/Swift-SwiftUI-orange">
  <img alt="On-device" src="https://img.shields.io/badge/speech-on--device%20(MLX%20%2B%20Core%20ML)-blue">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>

<!-- SCREENSHOTS -->

---

## Why Earshot

Siri decides which model you talk to, how it hears you and how it sounds. Earshot leaves every one
of those choices to you:

- **Brain.** Claude (Haiku, Sonnet or Opus with a thinking level), any model on OpenRouter, OpenAI,
  or a model running on your own PC through Ollama or LM Studio. Use your own API keys and pay only
  for what you use.
- **Ears (speech to text).** Run fully on the iPhone, or send audio to a cloud API when you want speed
  or a newer model.
- **Voice (text to speech).** Five open TTS models that run on the iPhone, plus Apple, OpenAI,
  ElevenLabs or Kokoro on your PC.
- **Agent.** Connect a [Hermes](#hermes-agent) agent on your own server for multi-step work the phone
  can't do itself, with spoken approvals before it runs anything risky.

Save any combination as a preset, like *Quick question* (Haiku), *Balanced* (Sonnet) or
*Think hard* (Opus), and switch between them with one tap.

## Ask about anything on screen

Press the **Action Button** in any app. Earshot takes a screenshot, opens with the screenshot already
attached, and starts listening. Just ask: *"What does this error mean?"*, *"Summarize this
article,"* *"Reply to this for me."* You can also open it and talk with no screenshot at all.

Replies stream in and are spoken sentence by sentence while they're still being written, and you can
talk over a reply to interrupt it.

## Runs on your iPhone

The on-device models download once from Hugging Face, are pinned to tested revisions and stay in the
app's storage. Audio never has to leave the phone.

**Speech to text**

| Model | Maker | Size | Runs on |
| --- | --- | --- | --- |
| Fluid Speech Mini (Parakeet TDT 0.6B v3 format) | FluidVoice Labs | ~253 MB | Neural Engine via [FluidAudio](https://github.com/FluidInference/FluidAudio) |
| Apple on-device speech | Apple | built in | iOS |

**Text to speech** (via [MLX](https://github.com/ml-explore/mlx-swift) and [mlx-audio-swift](https://github.com/Blaizzy/mlx-audio-swift))

| Model | Maker | Params | Voice cloning | Notes |
| --- | --- | --- | --- | --- |
| Pocket TTS | Kyutai | 100M | Yes | Smallest and fastest, English |
| Audio8 TTS 0.1B | Audio8 | 170M | Yes | English and Chinese |
| Ming-3.0 TTS | Ant Group | 0.5B | Yes | Best quality for its size |
| Qwen3-TTS | Alibaba Qwen | 0.6B | No | Named voices in 10 languages |
| Raon-OpenTTS | KRAFTON | 1B | Yes | Largest, English, non-commercial license |

**Only-my-voice listening.** Record a short sample and Earshot uses a CAM++ voiceprint model
(Core ML, ~16 MB) to ignore speech that isn't yours: TV, people nearby, or its own voice.

## Or use the cloud

| Part | Cloud and self-hosted options |
| --- | --- |
| Brain | Anthropic, OpenRouter, OpenAI, Ollama / LM Studio on your PC, Hermes |
| Ears | OpenAI transcription, Whisper on your PC, any OpenAI-compatible speech server (e.g. Qwen3-ASR on vLLM) |
| Voice | OpenAI voices, ElevenLabs, Kokoro on your PC |

API keys stay in the iPhone's Keychain and are sent only to the provider they belong to.

## Hermes agent

Pick **Hermes** as the brain and Earshot becomes a voice front end for a Hermes agent running on a
VPS or home server. Hermes uses its own tools to work through multi-step tasks the phone can't do
alone. When it wants to take a risky step, Earshot reads the request aloud and you answer
*allow* or *deny* by voice. The Agent tab shows its runs as they happen.

## Install

Earshot doesn't need a Mac or a paid developer account.

1. Every push to `main` builds an unsigned IPA on a GitHub Actions macOS runner. Download the
   `earshot-ipa` artifact from the latest [Build](../../actions/workflows/build.yml) run.
2. Install it in [LiveContainer](https://github.com/LiveContainer/LiveContainer) on your iPhone.
3. Add your API keys under **Models**, and download any on-device models you want.

To build on a Mac instead: `brew install xcodegen && xcodegen generate`, then open `Earshot.xcodeproj`.

## Set up the Action Button

In the app, **Settings › Shortcuts › Action Button** walks you through it: a Shortcut that runs
*Take Screenshot*, *Copy to Clipboard* and *Open URLs* with

```
livecontainer://livecontainer-launch?bundle-name=com.mahmudkarim.Earshot.app
```

Then assign that Shortcut to the Action Button in iOS Settings.

## Project layout

```
App/
  Brain/          Model providers and the Hermes client
  Ears/           Speech to text: Apple, Fluid on-device, servers, voice filter
  Voice/          Text to speech providers
  Conversation/   The talk loop, streaming and chat history
  Views/          SwiftUI screens
Packages/
  LocalVoices/    On-device TTS and STT engines (MLX + Core ML)
```

## Contributing

Earshot is an open-source community project. Issues and pull requests are welcome, whether that's a
new model provider, a new on-device voice or a bug fix.

## License

MIT. Individual on-device models keep their own licenses, listed above. Check them before any
commercial use.
