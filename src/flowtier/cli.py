import sys
import argparse
from pathlib import Path

from rich.traceback import install
from rich.panel import Panel
from rich_argparse import RichHelpFormatter

install(show_locals=True)

from flowtier.common import UserError, console
from flowtier.studio import Mode, AppConfig, run_flowtier_studio


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
        "surface_path",
        type=Path,
        default=None,
        help="Path to the input triangulation in a .ply file, representing the solid-void interface",
    )
    parser.add_argument(
        "-clp",
        "--center-lines-path",
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
    parser.add_argument(
        "-m",
        "--mode",
        type=Mode,
        choices=list(Mode),
        default=Mode.RESUME,
        help=(
            "reset: start fresh in GUI; "
            "resume: load previous state in GUI; "
            "silent: apply state without opening GUI"
        ),
    )

    return parser


def resolve_inputs(surface_path: Path, center_lines_path: Path) -> tuple[Path, Path]:
    surface_path = surface_path.resolve()
    if not surface_path.exists() or not surface_path.is_file():
        raise UserError("surface-path must be an existing file")
    
    center_lines_path = center_lines_path.resolve()
    if not center_lines_path.exists() or not center_lines_path.is_file():
        raise UserError("center-lines-path must be an existing file")

    return surface_path, center_lines_path


def resolve_output(input_path: Path, output_dir: Path | None, name: str | None) -> Path:
    output_dir = (output_dir or input_path.parent).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    if name is None:
        name = input_path.name.removesuffix("".join(input_path.suffixes))

    return (output_dir / name).with_suffix(".npz")


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

    args = parser.parse_args()

    try:
        surface_path, center_lines_path = resolve_inputs(args.surface_path, args.center_lines_path)
        output_path = resolve_output(surface_path, args.output_dir, args.name)
        app_config = AppConfig(surface_path, center_lines_path, output_path, args.mode)

        return run_flowtier_studio(app_config)

    
    except UserError as e:
        console.print(f"[bold red]✖ Error:[/bold red] {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    sys.exit(main())