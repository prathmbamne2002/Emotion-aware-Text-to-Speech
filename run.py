"""
run.py  –  Quick-start CLI for the Empathy Engine
──────────────────────────────────────────────────
Usage:
    python run.py "I just got the best news ever!"
    python run.py "I cannot believe they did that." --output angry.mp3
    python run.py --interactive
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


def run_once(text: str, output_path: str) -> None:
    from app.emotion_detector import detect
    from app.tts_engine import synthesize, get_applied_params

    print(f"\n📝 Text: {text!r}\n")
    print("🔍 Detecting emotion…")
    result = detect(text)
    params = get_applied_params(result)

    print(f"  Emotion    : {params['emoji']}  {result['primary_emotion'].upper()}")
    print(f"  Confidence : {result['confidence']:.1%}")
    print(f"  Intensity  : {result['intensity']:.1%}")
    print(f"\n  Scores:")
    for emotion, score in sorted(result["all_scores"].items(), key=lambda x: -x[1]):
        bar = "█" * int(score * 20)
        print(f"    {emotion:<10} {bar:<20} {score:.1%}")

    print(f"\n🎛️  Applied voice parameters:")
    print(f"    Rate       : {params['rate']}×")
    print(f"    Pitch      : {params['pitch_steps']:+.1f} semitones")
    print(f"    Volume     : {params['volume_db']:+.1f} dB")

    print("\n🎙️  Synthesizing audio…")
    mp3_bytes = synthesize(text, result)

    with open(output_path, "wb") as f:
        f.write(mp3_bytes)

    print(f"✅  Audio saved to: {output_path}  ({len(mp3_bytes):,} bytes)\n")


def interactive_mode() -> None:
    print("\n🎙️  Empathy Engine — Interactive Mode")
    print("   Type text and press Enter to synthesize. Type 'quit' to exit.\n")
    counter = 0
    while True:
        try:
            text = input("Enter text > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not text:
            continue
        if text.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        counter += 1
        run_once(text, f"output_{counter:03d}.mp3")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Empathy Engine CLI")
    parser.add_argument("text", nargs="?", help="Text to synthesize")
    parser.add_argument("--output", "-o", default="output.mp3", help="Output MP3 path")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
    elif args.text:
        run_once(args.text, args.output)
    else:
        parser.print_help()
