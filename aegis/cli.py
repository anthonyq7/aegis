"""CLI for Aegis"""

import argparse
import json
from argparse import RawDescriptionHelpFormatter
from math import e
from pathlib import Path
from typing import Optional

from transformers import logging as hf_logging

from aegis.predictor import Predictor

hf_logging.set_verbosity_error()


#gets code from --text or --file, Argparse enforces at least on is provided
def get_code_input(text: Optional[str], file_path: Optional[str]) -> str:
    if text:
        return text

    #argparse ensures path is provided
    file = Path(file_path).expanduser()
    if not file.exists():
        raise SystemExit(f"Error: File not found: {file_path}")
    return file.read_text(encoding="utf-8")


def main() -> None:
    #sets up command line args
    parser = argparse.ArgumentParser(
        prog="aegis",
        description="Detect whether code is human-written or AI-generated",
        formatter_class=RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
        "  aegis --file path/to/code.py\n"
        "  aegis --text 'def add(a, b): return a + b'\n"
        "  aegis --file script.py --json > result.json\n"
        "  aegis --threshold 0.7\n"
        )
    )

    #creates a mutually exclusive input group where only either --text or --file is allowed
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", type=str, help="Code snippet as text (quote if it contains spaces or special characters).")
    input_group.add_argument("--file", type=str, help="Path to code file")

    #output format
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

    #allows users to adjust a threshold
    parser.add_argument("--threshold", type=float, help="Custom threshold to classify as AI")

    #parses args
    args = parser.parse_args()

    #gets the code to analyze
    code = get_code_input(args.text, args.file)

    #loads model and makes a prediction
    if args.threshold is None:
        predictor = Predictor()
    elif args.threshold:
        predictor = Predictor(threshold=args.threshold)
    elif args.threshold == 0:
        predictor = Predictor(threshold=0)
    else:
        raise ValueError("Threshold must be a float between 0 and 1")
        

    result = predictor.predict(code)

    #display results
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(f"Prediction: {result['prediction']} | human={result['human']} | ai={result['ai']}")


if __name__ == "__main__":
    main()
