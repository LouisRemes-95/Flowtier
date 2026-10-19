import sys
import argparse
from pathlib import Path

from rich.traceback import install
from rich_argparse import RichHelpFormatter

install(show_locals=True)

from flowtier.common import console


class RichArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        console.print(f"[bold red]✖ Argument error:[/bold red] {message}\n")
        self.print_help()
        raise SystemExit(2)


def build_parser() -> RichArgumentParser:
    parser = RichArgumentParser(
        prog="Flowtier",
        description="Interface to segment filaments by sketching the filament center cross-section given the filament center lines",
        formatter_class=RichHelpFormatter,
    )

    parser.add_argument(
        "surface-path",
        type=Path,
        default=None,
        help="Path to the input triangulation in a .ply file, representing the solid-void interface",
    )
    parser.add_argument(
        "-clp"
        "--center_lines_path",
        type=Path,
        required=True,
        default=None,
        help="Path to the filament center lines in a .npz file",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Path to the directory where outputs should be saved",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default=None,
        help="Name of the output file without suffix",
    )


def main() -> int:
    parser = build_parser()

    return 0


if __name__ == "__main__":
    sys.exit(main())