"""Transcribe voice-check's WAVs with Whisper and compare them with the text they were given.

    python check_speech.py <voice-out folder>

Prints one line per clip and writes summary.md. Exits 1 when a clip is silent or its words are
mostly wrong, so a broken voice fails the CI job.
"""
import json
import pathlib
import re
import sys

from faster_whisper import WhisperModel

NUMBERS = {"58": "fifty eight", "12": "twelve", "6": "six"}


def words(text: str) -> list[str]:
    text = text.lower()
    for digits, spoken in NUMBERS.items():
        text = re.sub(rf"\b{digits}\b", spoken, text)
    text = text.replace("-", " ")
    return re.findall(r"[a-z']+", text)


def wer(reference: list[str], hypothesis: list[str]) -> float:
    rows = [[0] * (len(hypothesis) + 1) for _ in range(len(reference) + 1)]
    for i in range(len(reference) + 1):
        rows[i][0] = i
    for j in range(len(hypothesis) + 1):
        rows[0][j] = j
    for i in range(1, len(reference) + 1):
        for j in range(1, len(hypothesis) + 1):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            rows[i][j] = min(rows[i - 1][j] + 1, rows[i][j - 1] + 1, rows[i - 1][j - 1] + cost)
    return rows[-1][-1] / max(len(reference), 1)


def main(folder: pathlib.Path) -> int:
    whisper = WhisperModel("small.en", device="cpu", compute_type="int8")
    lines = ["| clip | audio s | made in s | x real time | WER | heard |", "|---|---|---|---|---|---|"]
    failed = False
    for report_file in sorted(folder.glob("*-report.json")):
        report = json.loads(report_file.read_text())
        lines.append(f"| **{report['model']}** load {report['load_seconds']} s, peak memory {report.get('peak_memory_mb')} MB | | | | | |")
        for sentence in report["sentences"]:
            segments, _ = whisper.transcribe(str(folder / sentence["file"]), language="en", beam_size=5)
            heard = " ".join(segment.text.strip() for segment in segments)
            score = wer(words(sentence["text"]), words(heard))
            speed = sentence["generate_seconds"] / max(sentence["audio_seconds"], 0.001)
            bad = sentence["peak"] < 0.01 or score > 0.35
            failed |= bad
            mark = " ❌" if bad else ""
            lines.append(
                f"| {sentence['file']}{mark} | {sentence['audio_seconds']:.2f} | {sentence['generate_seconds']:.2f} "
                f"| {speed:.2f} | {score:.0%} | {heard} |"
            )
            print(f"{sentence['file']}: WER {score:.0%}, {speed:.2f}x real time: {heard}")
    summary = "\n".join(lines) + "\n"
    (folder / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(pathlib.Path(sys.argv[1])))
