"""Command-line interface for SIMPL."""

import argparse
import sys
from pathlib import Path

_NOTEBOOK_URL = "https://raw.githubusercontent.com/TomGeorge1234/SIMPL/main/examples/simpl_demo.ipynb"
_NOTEBOOK_NAME = "simpl_demo.ipynb"


def demo(_args):
    """Download the demo notebook into the current directory."""
    import urllib.request

    dest = Path.cwd() / _NOTEBOOK_NAME

    if dest.exists():
        answer = input(f"{_NOTEBOOK_NAME} already exists. Overwrite? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            return

    # Check local source tree first (editable installs)
    local_path = Path(__file__).resolve().parent.parent.parent / "examples" / _NOTEBOOK_NAME
    if local_path.is_file():
        import shutil

        shutil.copy2(local_path, dest)
    else:
        print(f"Downloading {_NOTEBOOK_NAME} ...")
        try:
            urllib.request.urlretrieve(_NOTEBOOK_URL, dest)
        except Exception as e:
            dest.unlink(missing_ok=True)
            print(f"Error downloading notebook: {e}", file=sys.stderr)
            sys.exit(1)

    print(f"\nDemo notebook saved to ./{_NOTEBOOK_NAME}")

    answer = input(f"Run `jupyter notebook {_NOTEBOOK_NAME}`? [y/N] ").strip().lower()
    if answer == "y":
        import subprocess

        subprocess.run(["jupyter", "notebook", str(dest)])
    else:
        print(f"\n  jupyter notebook {_NOTEBOOK_NAME}\n")


def main():
    parser = argparse.ArgumentParser(prog="simpl", description="SIMPL command-line tools")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("demo", help="Download the demo notebook into the current directory")

    args = parser.parse_args()
    if args.command == "demo":
        demo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
