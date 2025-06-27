import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        help="root to dataset folder",
    )
    parser.add_argument(
        "--ext",
        default="jpg",
        type=str,
        help="file extension to filter",
    )

    parser.add_argument(
        "--out",
        default="out.txt",
        type=str,
        help="output dataset file",
    )

    args = parser.parse_args()

    with open(args.out, "a+") as f:
        for root, dirs, files in os.walk(args.root):
            for file in sorted(files):
                path = os.path.join(root, file)
                if path.endswith(args.ext):
                    f.write(path + "\n")
