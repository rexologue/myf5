import gdown
import argparse

GDRIVE_RECORD_ID = "1dJsyVRrG1ZKm-_8ABKrmFxucJ3W_9Ub6"

def main(output_file: str) -> None:
    gdown.download(id=GDRIVE_RECORD_ID, output=output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for downloading Ekaterina's audio record.")

    parser.add_argument(
       "--out",
       required=True,
       type=str,
       help="Output path, where record will be downloaded"
    )

    args = parser.parse_args()

    main(args.out)