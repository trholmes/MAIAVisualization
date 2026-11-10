#!/usr/bin/env python3
"""Make a publication-ready MAIA nozzle r-z cross-section.

The nozzle profiles are read from the same DD4hep compact XML geometry used by
``make_maia_schematics.py``.  Both PNG and vector PDF outputs are written.

Usage:
    python3 make_maia_nozzle.py
    python3 make_maia_nozzle.py --unit mm --side right
    python3 make_maia_nozzle.py --geometry-dir path/to/MAIA_v0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from make_maia_schematics import DEFAULT_GEOMETRY_DIR, load_geometry, unit_factor


SCRIPT_DIR = Path(__file__).resolve().parent

MATERIAL_STYLE = {
    "BCH2": ("#4f63c6", "Borated polyethylene", 0.72),
    "Tungsten_Alloy": ("#4a4a4a", "Tungsten alloy", 0.82),
}


def display_name(name: str, material: str) -> str:
    return MATERIAL_STYLE.get(material, ("#777777", material or name, 0.75))[1]


def profile_polygon(
    planes: tuple[tuple[float, float, float], ...],
    direction: float,
) -> list[tuple[float, float]]:
    outer = [(direction * z, r_max) for z, _, r_max in planes]
    inner = [(direction * z, r_min) for z, r_min, _ in reversed(planes)]
    return outer + inner


def draw_nozzle(
    geometry_dir: Path,
    output_dir: Path,
    unit: str,
    side: str,
    dpi: int,
) -> tuple[Path, Path]:
    model = load_geometry(geometry_dir).scaled(unit_factor(unit))
    if not model.nozzles:
        raise ValueError(f"No right-side nozzle profiles found in {geometry_dir}")

    direction = -1.0 if side == "left" else 1.0
    z_extent = max(z for nozzle in model.nozzles for z, _, _ in nozzle.z_planes)
    r_extent = max(r_max for nozzle in model.nozzles for _, _, r_max in nozzle.z_planes)
    z_margin = 0.045 * z_extent
    r_margin = 0.10 * r_extent

    fig, ax = plt.subplots(figsize=(10.5, 3.0), constrained_layout=True)

    # Draw the outer layers first so every material boundary remains visible.
    ordered_nozzles = sorted(
        model.nozzles,
        key=lambda nozzle: ("Cludding" not in nozzle.name, nozzle.material != "BCH2"),
    )
    for nozzle in ordered_nozzles:
        color, _, alpha = MATERIAL_STYLE.get(
            nozzle.material,
            ("#777777", nozzle.material or nozzle.name, 0.75),
        )
        ax.add_patch(
            Polygon(
                profile_polygon(nozzle.z_planes, direction),
                closed=True,
                facecolor=color,
                edgecolor=color,
                linewidth=1.25,
                alpha=alpha,
                zorder=2,
            )
        )

    tungsten = next(nozzle for nozzle in model.nozzles if nozzle.name == "NozzleW")

    # A compact in-region label complements the material legend.
    w_z0, _, w_r0 = tungsten.z_planes[-1]
    w_z1, _, w_r1 = tungsten.z_planes[-2]
    w_mid = (direction * (0.68 * w_z0 + 0.32 * w_z1), 0.48 * (0.68 * w_r0 + 0.32 * w_r1))
    ax.text(*w_mid, "W", fontsize=13, color="#202020", ha="center", va="center")

    legend_handles = []
    seen_labels: set[str] = set()
    for nozzle in model.nozzles:
        if "Cludding" in nozzle.name:
            continue
        color, _, _ = MATERIAL_STYLE.get(
            nozzle.material,
            ("#777777", nozzle.material or nozzle.name, 0.75),
        )
        label = display_name(nozzle.name, nozzle.material)
        if label in seen_labels:
            continue
        seen_labels.add(label)
        legend_handles.append(Line2D([0], [0], color=color, lw=5, label=label))

    ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=9)
    ax.set_title("MAIA nozzle profile", fontsize=15, pad=10)
    ax.set_xlabel(f"z [{unit}]", fontsize=13)
    ax.set_ylabel(f"r [{unit}]", fontsize=13)
    if side == "left":
        ax.set_xlim(-z_extent - z_margin, z_margin * 0.18)
    else:
        ax.set_xlim(-z_margin * 0.18, z_extent + z_margin)
    ax.set_ylim(0.0, r_extent + r_margin)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#ededed", linewidth=0.8)
    ax.minorticks_on()
    ax.tick_params(which="major", direction="out", length=4, color="#555555")
    ax.tick_params(which="minor", direction="out", length=2, color="#777777")
    ax.spines[["top", "right"]].set_visible(False)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"maia_nozzle_{side}.{unit}"
    png_path = Path(f"{stem}.png")
    pdf_path = Path(f"{stem}.pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return png_path, pdf_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geometry-dir",
        type=Path,
        default=DEFAULT_GEOMETRY_DIR,
        help="Path to the MAIA_v0 DD4hep compact XML geometry directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "maia_schematics",
        help="Directory for the PNG and PDF outputs.",
    )
    parser.add_argument("--unit", choices=("mm", "cm"), default="mm", help="Plot unit.")
    parser.add_argument(
        "--side",
        choices=("left", "right"),
        default="left",
        help="Which longitudinal side to draw; left matches the reference figure.",
    )
    parser.add_argument("--dpi", type=int, default=220, help="PNG resolution.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    paths = draw_nozzle(args.geometry_dir, args.output_dir, args.unit, args.side, args.dpi)
    print(f"Read nozzle geometry from {args.geometry_dir}")
    for path in paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
