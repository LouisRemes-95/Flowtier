import argparse
import sys
from pathlib import Path
from typing import NoReturn, override

import numpy as np
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
        allow_abbrev=False,
    )

    parser.add_argument(
        "surface_path",
        type=Path,
        help="Path to the input triangulation in a .ply file, representing the solid-void interface",
    )
    parser.add_argument(
        "centerlines_path",
        type=Path,
        help="Path to the filament center lines in a .npz file",
    )
    parser.add_argument(
        "-css",
        "--cross-section-state-path",
        type=Path,
        help="Path to a cross section state in a .npz file",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Path to the directory where outputs should be saved",
    )
    parser.add_argument(
        "-n",
        "--output-name",
        type=str,
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
        "-p",
        "--precision",
        type=float,
        default=1,
        help=r"Voxel grid precision in \[mm]",
    )
    parser.add_argument(
        "-bd",
        "--build-direction",
        type=float,
        nargs=3,
        default=1,
        required=True,
        help="Build direction when 3D printing",
    )

    return parser


def resolve_input_path(path: Path, name: str) -> Path:
    path = path.resolve()

    if not path.is_file():
        raise UserError(f"{name} must be an existing file: {path}")

    return path


def resolve_cross_section_state_path(path: Path | None, fallback_dir: Path) -> Path:
    path = (path or fallback_dir / "cross_section_state").resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    return path.with_name(path.name.split(".")[0] + ".npz")


def resolve_output(input_path: Path, dir: Path | None, name: str | None) -> Path:
    dir = (dir or input_path.parent).resolve()

    dir.mkdir(parents=True, exist_ok=True)

    if name is None:
        name = input_path.name.removesuffix("".join(input_path.suffixes))

    return (dir / name).with_suffix(".npz")


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
        surface_path = resolve_input_path(args.surface_path, "Surface")
        centerlines_path = resolve_input_path(args.centerlines_path, "Centerlines")
        cross_section_state_path = resolve_cross_section_state_path(
            args.cross_section_state_path, args.centerlines_path.parent
        )
        output_path = resolve_output(surface_path, args.output_dir, args.output_name)

        app_config = AppConfig(surface_path, centerlines_path, cross_section_state_path, output_path, args.mode)
        alg_config = AlgConfig(args.precision, np.array(args.build_direction))

        return run_flowtier_studio(app_config, alg_config)

    except UserError as e:
        console.print(f"[bold red]✖ Error:[/bold red] {e}")
        raise SystemExit(1) from None


if __name__ == "__main__":
    sys.exit(main())
