"""Download a CoastSeg zoo model by model name.

This script supports selecting one of the six configured model names either
interactively or via ``--model-name``.
"""

# Usage:
#   python download_zoo_model.py [-m NAME] [-i MODE] [-o DIR] [-l]
#   python download_zoo_model.py [--model-name NAME] [--implementation MODE]
#                                [--output-dir DIR] [--list-models]
#
# Parameters:
#   -m, --model-name NAME
#       Model to download. If omitted, you will be prompted to choose.
#   -i, --implementation {BEST,ENSEMBLE}
#       Download mode passed to Zoo_Model.download_model(). Default: BEST.
#   -o, --output-dir DIR
#       Optional parent directory for model storage.
#       Models are saved to: <DIR>/<model-name>
#   -l, --list-models
#       Print the allowed model names and exit.
#
# Examples:
#   python download_zoo_model.py --list-models
#   python download_zoo_model.py --model-name global_segformer_RGB_4class_14036903 --implementation BEST --output-dir /path/to/save/models
#   Recommended example:
#   python download_zoo_model.py --model-name global_segformer_RGB_4class_14036903
#   - This saves the global segformer RGB model to the default location (CoastSeg/models/global_segformer_RGB_4class_14036903) and downloads only the best weights for the model.
#
# Default save location:
#   We recommend running this script from the repository root and using the default save location. By default, models are saved to:
#   CoastSeg/models/<model-name>
#   If --output-dir is not provided, models are saved under:
#   CoastSeg/models/<model-name>

from __future__ import annotations

import argparse
import sys
from pathlib import Path


RGB_MODELS = [
    "global_segformer_RGB_4class_14036903",
    "AK_segformer_RGB_4class_14037041",
]
MNDWI_MODELS = [
    "global_segformer_MNDWI_4class_14183366",
    "AK_segformer_MNDWI_4class_14187478",
]
NDWI_MODELS = [
    "global_segformer_NDWI_4class_14172182",
    "AK_segformer_NDWI_4class_14183210",
]

ALLOWED_MODELS = RGB_MODELS + MNDWI_MODELS + NDWI_MODELS
_MODEL_LOOKUP = {model.strip(): model for model in ALLOWED_MODELS}


def ensure_src_on_path() -> Path:
    """Add repository and src paths to ``sys.path``.

    Returns:
        Path: Absolute path to repository root.
    """
    repo_root = Path(__file__).resolve().parent
    src_path = repo_root / "src"

    for candidate in (repo_root, src_path):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)

    return repo_root


def parse_args() -> argparse.Namespace:
    """Parse command-line options.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Download one CoastSeg zoo model by model name."
    )
    parser.add_argument(
        "-m",
        "--model-name",
        type=str,
        default="",
        help="Model name to download. If omitted, interactive selection is used.",
    )
    parser.add_argument(
        "-i",
        "--implementation",
        choices=["BEST", "ENSEMBLE", "best", "ensemble"],
        default="BEST",
        help="Download mode passed to Zoo_Model.download_model().",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="",
        help=(
            "Optional parent output directory (defaults to CoastSeg/models). If provided, "
            "model files are saved to <output-dir>/<model-name>."
        ),
    )
    parser.add_argument(
        "-l",
        "--list-models",
        action="store_true",
        help="Print allowed model names and exit.",
    )
    return parser.parse_args()


def print_model_groups() -> None:
    """Print the six allowed model names grouped by image type."""
    print("RGB_models = [")
    for name in RGB_MODELS:
        print(f'    "{name}",')
    print("]")

    print("MNDWI_models = [")
    for name in MNDWI_MODELS:
        print(f'    "{name}",')
    print("]")

    print("NDWI_models = [")
    for name in NDWI_MODELS:
        print(f'    "{name}",')
    print("]")


def choose_model_interactively() -> str:
    """Prompt the user to select one model from the allowed list.

    Returns:
        str: Selected model name.
    """
    print("Select a model to download:\n")
    indexed_models: list[str] = []

    print("RGB models:")
    for name in RGB_MODELS:
        indexed_models.append(name)
        print(f"  {len(indexed_models)}. {name}")

    print("\nMNDWI models:")
    for name in MNDWI_MODELS:
        indexed_models.append(name)
        print(f"  {len(indexed_models)}. {name}")

    print("\nNDWI models:")
    for name in NDWI_MODELS:
        indexed_models.append(name)
        print(f"  {len(indexed_models)}. {name}")

    while True:
        user_input = input("\nEnter the number of the model to download: ").strip()
        if not user_input.isdigit():
            print("Please enter a valid number.")
            continue

        selected_index = int(user_input)
        if 1 <= selected_index <= len(indexed_models):
            return indexed_models[selected_index - 1]

        print(f"Please enter a number from 1 to {len(indexed_models)}.")


def resolve_model_name(raw_model_name: str) -> str:
    """Normalize and validate model name.

    Args:
        raw_model_name (str): User-provided model name.

    Returns:
        str: Canonical model name.

    Raises:
        ValueError: If model name is not one of the allowed values.
    """
    candidate = raw_model_name.strip()
    if candidate in _MODEL_LOOKUP:
        return _MODEL_LOOKUP[candidate]

    allowed = "\n".join(f"- {name}" for name in ALLOWED_MODELS)
    raise ValueError(
        f"Unsupported model name: '{raw_model_name}'.\n" f"Choose one of:\n{allowed}"
    )


def get_target_directory(
    model_name: str,
    output_dir: str,
    zoo: "Zoo_Model",
) -> Path:
    """Resolve where model files should be downloaded.

    Args:
        model_name (str): Canonical model name.
        output_dir (str): Optional parent output directory.
        zoo (Zoo_Model): Zoo model helper instance.

    Returns:
        Path: Download target directory.
    """
    if output_dir.strip():
        target_directory = Path(output_dir).expanduser().resolve() / model_name
        target_directory.mkdir(parents=True, exist_ok=True)
        return target_directory

    return Path(zoo.get_model_directory(model_name)).expanduser().resolve()


def main() -> int:
    """Run model download workflow.

    Returns:
        int: Process exit code.
    """
    repo_root = ensure_src_on_path()

    from coastseg.zoo_model import Zoo_Model, retrieve_model_weights

    args = parse_args()

    if args.list_models:
        print_model_groups()
        return 0

    try:
        selected_model = (
            resolve_model_name(args.model_name)
            if args.model_name
            else choose_model_interactively()
        )
    except (KeyboardInterrupt, EOFError):
        print("\nSelection cancelled.")
        return 1
    except ValueError as exc:
        print(str(exc))
        return 1

    default_models_root = (repo_root / "models").resolve()
    mode = str(args.implementation).upper()
    zoo = Zoo_Model()
    target_dir = get_target_directory(
        selected_model,
        args.output_dir or str(default_models_root),
        zoo,
    )

    print(f"Downloading model: {selected_model}")
    print(f"Mode: {mode}")
    print(f"Target directory: {target_dir}")

    try:
        zoo.download_model(mode, selected_model, str(target_dir))
        weight_files = retrieve_model_weights(str(target_dir), mode)
    except Exception as exc:
        print(f"Model download failed: {exc}")
        return 1

    print("Download complete.")
    print("Resolved weight files:")
    for path in weight_files:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
