import argparse
import json
import tomllib
from pathlib import Path

from .dataset import DatasetConfig, generate_dataset, verify_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate or verify exact-distance datasets")
    parser.add_argument("action", choices=["generate", "verify"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.action == "generate":
        settings = tomllib.loads(args.config.read_text()) if args.config else {}
        print(json.dumps(generate_dataset(args.output, DatasetConfig(**settings)), indent=2))
    else:
        print(json.dumps(verify_dataset(args.output), indent=2))


if __name__ == "__main__":
    main()
