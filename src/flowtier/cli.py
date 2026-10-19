import sys
import argparse
from pathlib import Path

from rich.traceback import install
from rich.panel import Panel
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
    # parser.add_argument(
    #     "--mode",
    #     type=Mode,
    #     choices=list(Mode),
    #     default=Mode.RESUME,
    #     help=(
    #         "reset: start fresh in GUI; "
    #         "resume: load previous state in GUI; "
    #         "silent: apply state without opening GUI"
    #     ),
    # )


def main() -> int:
    parser = build_parser()

    console.print(
        Panel(
            "[bold cyan]Flowtier[/bold cyan]\n"
            "Filament segmentation interface",
            border_style="cyan",
            expand=False,
        )
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())