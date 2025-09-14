import argparse
import os
from ruaccent import RUAccent

def process_text(accentizer: RUAccent, text: str) -> str:
    return accentizer.process_all(text)

def process_file(accentizer: RUAccent, file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return process_text(accentizer, text)

def main():
    parser = argparse.ArgumentParser(description="RUAccent CLI wrapper")
    parser.add_argument("--text", type=str, help="Input text")
    parser.add_argument("--file", type=str, help="Path to input file")
    parser.add_argument("--out", type=str, help="Path to output file (optional)")

    args = parser.parse_args()

    if not args.text and not args.file:
        parser.error("You must specify either --text or --file")

    # Load model
    accentizer = RUAccent()
    accentizer.load(omograph_model_size='turbo3.1', use_dictionary=True, tiny_mode=False)

    if args.text:
        result = process_text(accentizer, args.text)
    else:
        result = process_file(accentizer, args.file)

    if args.out:
        out_path = args.out
        # Самомодификация при совпадении
        if args.file and os.path.abspath(args.file) == os.path.abspath(args.out):
            out_path = args.file
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(result)
    else:
        print(result)

if __name__ == "__main__":
    main()
