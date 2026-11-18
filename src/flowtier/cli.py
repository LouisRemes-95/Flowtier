import argparse
import sys
from pathlib import Path
from typing import NoReturn, override

from rich.panel import Panel
from rich.traceback import install
from rich_argparse import RichHelpFormatter

from flowtier.common import UserError, console
from flowtier.studio import AlgConfig, AppConfig, Mode, run_flowtier_studio

install(show_locals=True)


class RichArgumentParser(argparse.ArgumentParser):
    @override
    def error(self, message: str) -> NoReturn:
        console.print(f"[bold red]✖ Argument error:[/bold red] {message}\n")
        self.print_help()
        raise SystemExit(2)


def build_parser() -> RichArgumentParser:
    parser = RichArgumentParser(
        prog="Flowtier",
        description="Interface to segment filaments by sketching the filament center cross-section"
        "given the filament center lines",
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
        "--centerlines-path",
        type=Path,
        required=True,
        default=None,
        help="Path to the filament center lines in a .npz file",
    )
    parser.add_argument(
        "-cs",
        "--cross-section-path",
        type=Path,
        default=None,
        help="Path to a cross section in a .npz file",
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
        help=("reset: start fresh in GUI; resume: load previous state in GUI; silent: apply state without opening GUI"),
    )
    parser.add_argument(
        "--precision",
        type=float,
        default=1,
        help="Voxel grid precision in mm",
    )

    return parser


def resolve_surface_path(surface_path: Path) -> Path:
    surface_path = surface_path.resolve()
    if not surface_path.exists() or not surface_path.is_file():
        raise UserError("surface-path must be an existing file")

    return surface_path


def resolve_centerlines_path(centerlines_path: Path) -> Path:
    centerlines_path = centerlines_path.resolve()
    if not centerlines_path.exists() or not centerlines_path.is_file():
        raise UserError("center-lines-path must be an existing file")

    return centerlines_path


def resolve_cross_section_path(cross_section_path: Path, fallback_dir: Path) -> Path:
    cross_section_path = (cross_section_path or fallback_dir / "cross_section").resolve()

    cross_section_path.parent.mkdir(parents=True, exist_ok=True)

    return cross_section_path.with_name(
        cross_section_path.name.removesuffix("".join(cross_section_path.suffixes)) + ".npz"
    )


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
            "[bold cyan]Flowtier[/bold cyan]\nFilament segmentation interface",
            border_style="cyan",
            expand=False,
        )
    )

    args = parser.parse_args()

    try:
        surface_path = resolve_surface_path(args.surface_path)
        centerlines_path = resolve_centerlines_path(args.centerlines_path)
        cross_section_path = resolve_cross_section_path(args.cross_section_path)
        output_path = resolve_output(surface_path, args.output_dir, args.name)

        app_config = AppConfig(surface_path, centerlines_path, cross_section_path, output_path, args.mode)
        alg_config = AlgConfig(args.precision)

        return run_flowtier_studio(app_config, alg_config)

    except UserError as e:
        console.print(f"[bold red]✖ Error:[/bold red] {e}")
        raise SystemExit(1) from None


if __name__ == "__main__":
    sys.exit(main())
