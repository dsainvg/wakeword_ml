"""Prove edge-tts can synthesise the keyword before a long corpus build runs.

The corpus builder needs a pool of TTS 'amaze' waveforms to make positives. If
TTS is blocked or every voice fails, the build either dies or -- worse -- quietly
produces a corpus with no positives. Test it in isolation first.

    py check_tts.py
"""

import asyncio
import os
import sys

import numpy as np

OUT = []


def emit(s=""):
    OUT.append(str(s))
    print(s, flush=True)


async def main():
    import edge_tts
    emit("edge-tts %s" % getattr(edge_tts, "__version__", "?"))

    voices = ["en-US-AriaNeural", "en-US-GuyNeural", "en-GB-SoniaNeural",
              "en-US-JennyNeural", "en-AU-NatashaNeural"]
    ok = 0
    emit("")
    emit("%-24s %8s %6s" % ("voice", "samples", "sec"))
    emit("-" * 42)
    for v in voices:
        try:
            comm = edge_tts.Communicate("amaze", v)
            audio = b""
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    audio += chunk["data"]
            import io
            import soundfile as sf
            data, sr = sf.read(io.BytesIO(audio))
            emit("%-24s %8d %6.2f  sr=%d" % (v, len(data), len(data) / sr, sr))
            ok += 1
        except Exception as e:
            emit("%-24s FAILED: %s" % (v, str(e)[:60]))
    emit("")
    emit("voices working: %d/%d" % (ok, len(voices)))

    # Also confirm the offline fallbacks are absent, so a mid-build failure is
    # attributable to edge-tts rather than a missing engine.
    for mod in ("gtts", "pyttsx3"):
        try:
            __import__(mod)
            emit("%s: available" % mod)
        except Exception:
            emit("%s: NOT available (edge-tts is the only engine)" % mod)

    open("tts_check.out", "w", encoding="utf-8").write("\n".join(OUT))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))