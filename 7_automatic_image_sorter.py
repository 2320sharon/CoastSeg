import argparse

from coastseg import classifier

# -----------------------------------------------------------------------------
# Automatic image sorter CLI
#
# - Uses CoastSeg's image classifier to sort images in an ROI RGB directory.
# - Classifies each image as good or bad using the selected threshold.
# - Moves bad images into a 'bad' subdirectory in the input directory.
#
# How to use:
# - Provide the ROI RGB directory as the required positional argument.
# - Optionally set --threshold (default: 0.40).
# - Optionally pass --quiet to suppress status messages.
#
# Examples:
# 1) Sort with the default threshold (0.40):
#    python 7_automatic_image_sorter.py "C:/CoastSeg/data/ID_12_datetime06-05-23__04_16_45/jpg_files/preprocessed/RGB"
#
# 2) Sort with a custom threshold and no status output:
#    python 7_automatic_image_sorter.py "C:/CoastSeg/data/ID_12_datetime06-05-23__04_16_45/jpg_files/preprocessed/RGB" --threshold 0.55 --quiet
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for automatic image sorting."""
    parser = argparse.ArgumentParser(
        description=(
            "Automatically sort ROI RGB images as good or bad using the "
            "classifier model."
        )
    )
    parser.add_argument(
        "input_directory",
        help=(
            "Path to the ROI RGB directory, for example: "
            "CoastSeg/data/<ROI_NAME>/jpg_files/preprocessed/RGB"
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.40,
        help="Classification threshold. Start with a low value such as 0.40.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable status printing during sorting.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the automatic image sorter from command-line arguments."""
    args = parse_args()
    if not args.quiet:
        print(f"Sorting images in: {args.input_directory}")
    classifier.sort_images_with_model(
        args.input_directory,
        threshold=args.threshold,
        print_status=not args.quiet,
    )


if __name__ == "__main__":
    main()
