#!/usr/bin/env python3
"""Make MAIA detector schematic plots from DD4hep compact XML geometry.

The script reads the MAIA_v0 geometry directory, evaluates the XML constants,
and produces:

  1. An r-z overview with pseudorapidity guide lines.
  2. An x-y overview with detector barrel radii labeled.
  3. A tracker-only r-z view showing Vertex, Inner Tracker, and Outer Tracker
     barrel layers and endcap rings explicitly.
  4. A tracker-only perspective schematic approximating the x-y/3D barrel view.

Usage:
    python3 make_maia_schematics.py
    python3 make_maia_schematics.py --geometry-dir path/to/MAIA_v0
    python3 make_maia_schematics.py --output-dir figures --unit cm
"""

from __future__ import annotations

import argparse
import math
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_GEOMETRY_DIR = SCRIPT_DIR / "external" / "detector-simulation" / "geometries" / "MAIA_v0"

CACHE_DIR = Path(tempfile.gettempdir()) / "maia_schematic_cache"
(CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
(CACHE_DIR / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR / "xdg"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon, Rectangle, Wedge


SUBSYSTEM_COLORS = {
    "Vertex Detector": "#2c8a3d",
    "Inner Tracker": "#2368aa",
    "Outer Tracker": "#e08e2e",
    "Solenoid": "#707070",
    "ECAL": "#c84d4d",
    "HCAL": "#7b5aa6",
    "Muon Detector": "#2f9a9a",
}

SUBSYSTEM_LABELS = {
    "Vertex Detector": "VTX",
    "Inner Tracker": "IT",
    "Outer Tracker": "OT",
    "Solenoid": "Solenoid",
    "ECAL": "ECAL",
    "HCAL": "HCAL",
    "Muon Detector": "Muon",
}

TRACKER_SUBSYSTEMS = ("Vertex Detector", "Inner Tracker", "Outer Tracker")


@dataclass(frozen=True)
class DetectorRegion:
    subsystem: str
    region: str
    r_min: float
    r_max: float
    z_low: float
    z_high: float
    material: str = ""

    def scaled(self, factor: float) -> "DetectorRegion":
        return DetectorRegion(
            subsystem=self.subsystem,
            region=self.region,
            r_min=self.r_min * factor,
            r_max=self.r_max * factor,
            z_low=self.z_low * factor,
            z_high=self.z_high * factor,
            material=self.material,
        )


@dataclass(frozen=True)
class BarrelLayer:
    subsystem: str
    detector: str
    layer_id: str
    radius: float
    z_half: float

    def scaled(self, factor: float) -> "BarrelLayer":
        return BarrelLayer(
            subsystem=self.subsystem,
            detector=self.detector,
            layer_id=self.layer_id,
            radius=self.radius * factor,
            z_half=self.z_half * factor,
        )


@dataclass(frozen=True)
class EndcapRing:
    subsystem: str
    detector: str
    layer_id: str
    z: float
    r_min: float
    r_max: float
    module: str

    def scaled(self, factor: float) -> "EndcapRing":
        return EndcapRing(
            subsystem=self.subsystem,
            detector=self.detector,
            layer_id=self.layer_id,
            z=self.z * factor,
            r_min=self.r_min * factor,
            r_max=self.r_max * factor,
            module=self.module,
        )


@dataclass(frozen=True)
class NozzleProfile:
    name: str
    material: str
    z_planes: tuple[tuple[float, float, float], ...]

    def scaled(self, factor: float) -> "NozzleProfile":
        return NozzleProfile(
            name=self.name,
            material=self.material,
            z_planes=tuple((z * factor, r_min * factor, r_max * factor) for z, r_min, r_max in self.z_planes),
        )


@dataclass(frozen=True)
class GeometryModel:
    constants: dict[str, float]
    regions: list[DetectorRegion]
    barrel_layers: list[BarrelLayer]
    endcap_rings: list[EndcapRing]
    nozzles: list[NozzleProfile]

    def scaled(self, factor: float) -> "GeometryModel":
        return GeometryModel(
            constants={key: value * factor for key, value in self.constants.items()},
            regions=[region.scaled(factor) for region in self.regions],
            barrel_layers=[layer.scaled(factor) for layer in self.barrel_layers],
            endcap_rings=[ring.scaled(factor) for ring in self.endcap_rings],
            nozzles=[nozzle.scaled(factor) for nozzle in self.nozzles],
        )


def unit_factor(unit: str) -> float:
    if unit == "mm":
        return 1.0
    if unit == "cm":
        return 0.1
    raise ValueError(f"Unsupported unit: {unit}")


def fmt_radius(value: float, unit: str) -> str:
    if abs(value - round(value)) < 1e-9:
        return f"R = {int(round(value))} {unit}"
    return f"R = {value:.1f} {unit}"


def strip_int_casts(expr: str) -> str:
    return re.sub(r"\(int\)\s*", "", expr.strip())


def eval_expr(expr: str, constants: dict[str, float]) -> float:
    """Evaluate a compact-XML numeric expression in millimeters."""
    expr = strip_int_casts(expr)
    namespace = {
        **constants,
        "mm": 1.0,
        "cm": 10.0,
        "m": 1000.0,
        "um": 0.001,
        "rad": 1.0,
        "deg": math.pi / 180.0,
        "tesla": 1.0,
        "pi": math.pi,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "sqrt": math.sqrt,
    }
    return float(eval(expr, {"__builtins__": {}}, namespace))


def included_xml_paths(root: ET.Element, geometry_dir: Path) -> Iterable[Path]:
    for elem in root.findall(".//include") + root.findall(".//gdmlFile"):
        ref = elem.attrib.get("ref", "")
        if not ref or "$" in ref:
            continue
        path = geometry_dir / ref
        if path.suffix == ".xml" and path.exists():
            yield path.resolve()


def load_xml_roots(geometry_dir: Path, main_file: str = "MAIA_v0.xml") -> list[ET.Element]:
    main_path = (geometry_dir / main_file).resolve()
    if not main_path.exists():
        raise FileNotFoundError(f"Could not find geometry XML: {main_path}")

    roots: list[ET.Element] = []
    seen: set[Path] = set()

    def visit(path: Path) -> None:
        if path in seen:
            return
        seen.add(path)
        root = ET.parse(path).getroot()
        roots.append(root)
        for include_path in included_xml_paths(root, geometry_dir):
            visit(include_path)

    visit(main_path)

    for path in sorted(geometry_dir.glob("*.xml")):
        visit(path.resolve())

    return roots


def load_constants(roots: list[ET.Element]) -> dict[str, float]:
    constants: dict[str, float] = {}
    pending: list[tuple[str, str]] = []

    for root in roots:
        for elem in root.findall(".//constant"):
            name = elem.attrib.get("name")
            value = elem.attrib.get("value")
            if not name or value is None or elem.attrib.get("type") == "string":
                continue
            pending.append((name, value))

    while pending:
        progressed = False
        next_pending: list[tuple[str, str]] = []
        for name, value in pending:
            try:
                constants[name] = eval_expr(value, constants)
                progressed = True
            except Exception:
                next_pending.append((name, value))
        if not progressed:
            unresolved = ", ".join(name for name, _ in next_pending[:10])
            raise ValueError(f"Could not resolve geometry constants: {unresolved}")
        pending = next_pending

    return constants


def subsystem_for_detector(detector_name: str) -> str | None:
    if detector_name.startswith("Vertex"):
        return "Vertex Detector"
    if detector_name.startswith("InnerTracker"):
        return "Inner Tracker"
    if detector_name.startswith("OuterTracker"):
        return "Outer Tracker"
    return None


def module_shapes(detector: ET.Element, constants: dict[str, float]) -> dict[str, tuple[str, float]]:
    """Return module radial interpretation.

    For TrackerEndcap modules, `trd y` is the radial width and ring `r` is the
    inner edge. For VertexEndcap modules, `trd z` is the radial half-width and
    ring `r` is the center.
    """
    shapes: dict[str, tuple[str, float]] = {}
    for module in detector.findall("module"):
        name = module.attrib.get("name")
        trd = module.find("trd")
        if not name or trd is None:
            continue
        if "z" in trd.attrib:
            shapes[name] = ("center_half", eval_expr(trd.attrib["z"], constants))
        elif "y" in trd.attrib:
            shapes[name] = ("inner_width", eval_expr(trd.attrib["y"], constants))
    return shapes


def barrel_module_lengths(detector: ET.Element, constants: dict[str, float]) -> dict[str, float]:
    lengths: dict[str, float] = {}
    for module in detector.findall("module"):
        name = module.attrib.get("name")
        envelope = module.find("module_envelope")
        if name and envelope is not None and "length" in envelope.attrib:
            lengths[name] = eval_expr(envelope.attrib["length"], constants)
    return lengths


def parse_tracker_layers(roots: list[ET.Element], constants: dict[str, float]) -> tuple[list[BarrelLayer], list[EndcapRing]]:
    barrel_layers: list[BarrelLayer] = []
    endcap_rings: list[EndcapRing] = []

    for root in roots:
        for detector in root.findall(".//detector"):
            detector_name = detector.attrib.get("name", "")
            subsystem = subsystem_for_detector(detector_name)
            if subsystem is None:
                continue

            if detector_name.endswith("Barrel"):
                module_lengths = barrel_module_lengths(detector, constants)
                for layer in detector.findall("layer"):
                    layer_id = layer.attrib.get("id", "?")
                    rphi_layout = layer.find("rphi_layout")
                    z_layout = layer.find("z_layout")
                    if rphi_layout is not None and z_layout is not None:
                        radius = eval_expr(rphi_layout.attrib["rc"], constants)
                        z0 = eval_expr(z_layout.attrib["z0"], constants)
                        module_name = layer.attrib.get("module", "")
                        z_half = z0 + 0.5 * module_lengths.get(module_name, 0.0)
                        barrel_layers.append(BarrelLayer(subsystem, detector_name, layer_id, radius, z_half))
                        continue

                    distances: list[float] = []
                    lengths: list[float] = []
                    for child in list(layer):
                        if child.tag not in {"sensitive", "ladder"}:
                            continue
                        if "distance" in child.attrib:
                            distances.append(eval_expr(child.attrib["distance"], constants))
                        if "length" in child.attrib:
                            lengths.append(eval_expr(child.attrib["length"], constants))
                    if distances and lengths:
                        barrel_layers.append(
                            BarrelLayer(subsystem, detector_name, layer_id, max(distances), max(lengths))
                        )

            if detector_name.endswith("Endcap"):
                shapes = module_shapes(detector, constants)
                for layer in detector.findall("layer"):
                    layer_id = layer.attrib.get("id", "?")
                    for ring in layer.findall("ring"):
                        module_name = ring.attrib.get("module", "")
                        radius = eval_expr(ring.attrib["r"], constants)
                        z = eval_expr(ring.attrib["zstart"], constants)
                        mode, size = shapes.get(module_name, ("point", 0.0))
                        if mode == "center_half":
                            r_min, r_max = radius - size, radius + size
                        elif mode == "inner_width":
                            r_min, r_max = radius, radius + size
                        else:
                            r_min, r_max = radius, radius
                        endcap_rings.append(
                            EndcapRing(subsystem, detector_name, layer_id, z, r_min, r_max, module_name)
                        )

    return barrel_layers, endcap_rings


def parse_nozzles(roots: list[ET.Element], constants: dict[str, float]) -> list[NozzleProfile]:
    nozzles: list[NozzleProfile] = []
    for root in roots:
        for detector in root.findall(".//detector"):
            name = detector.attrib.get("name", "")
            if not name.startswith("Nozzle") or not name.endswith("_right"):
                continue
            z_planes = []
            for zplane in detector.findall("zplane"):
                z = abs(eval_expr(zplane.attrib["z"], constants))
                r_min = eval_expr(zplane.attrib["rmin"], constants)
                r_max = eval_expr(zplane.attrib["rmax"], constants)
                z_planes.append((z, r_min, r_max))
            if z_planes:
                material_node = detector.find("material")
                material = material_node.attrib.get("name", "") if material_node is not None else ""
                nozzles.append(
                    NozzleProfile(
                        name=name.replace("_right", ""),
                        material=material,
                        z_planes=tuple(sorted(z_planes, key=lambda item: item[0])),
                    )
                )
    return nozzles


def tracker_regions(barrel_layers: list[BarrelLayer], endcap_rings: list[EndcapRing]) -> list[DetectorRegion]:
    regions: list[DetectorRegion] = []
    for subsystem in TRACKER_SUBSYSTEMS:
        barrels = [layer for layer in barrel_layers if layer.subsystem == subsystem]
        rings = [ring for ring in endcap_rings if ring.subsystem == subsystem]
        if barrels:
            regions.append(
                DetectorRegion(
                    subsystem=subsystem,
                    region="Barrel",
                    r_min=min(layer.radius for layer in barrels),
                    r_max=max(layer.radius for layer in barrels),
                    z_low=0.0,
                    z_high=max(layer.z_half for layer in barrels),
                    material="Si",
                )
            )
        if rings:
            regions.append(
                DetectorRegion(
                    subsystem=subsystem,
                    region="Endcap",
                    r_min=min(ring.r_min for ring in rings),
                    r_max=max(ring.r_max for ring in rings),
                    z_low=min(ring.z for ring in rings),
                    z_high=max(ring.z for ring in rings),
                    material="Si",
                )
            )
    return regions


def detector_envelope_regions(constants: dict[str, float]) -> list[DetectorRegion]:
    return [
        DetectorRegion(
            "Solenoid",
            "Barrel",
            constants["Solenoid_inner_radius"],
            constants["Solenoid_outer_radius"],
            0.0,
            constants["Solenoid_half_length"],
            "Al",
        ),
        DetectorRegion(
            "ECAL",
            "Barrel",
            constants["ECalBarrel_inner_radius"],
            constants["ECalBarrel_outer_radius"],
            0.0,
            constants["ECalBarrel_half_length"],
            "W + Si",
        ),
        DetectorRegion(
            "ECAL",
            "Endcap",
            constants["ECalEndcap_inner_radius"],
            constants["ECalEndcap_outer_radius"],
            constants["ECalEndcap_min_z"],
            constants["ECalEndcap_max_z"],
            "W + Si",
        ),
        DetectorRegion(
            "HCAL",
            "Barrel",
            constants["HCalBarrel_inner_radius"],
            constants["HCalBarrel_outer_radius"],
            0.0,
            constants["HCalBarrel_half_length"],
            "Fe + PS",
        ),
        DetectorRegion(
            "HCAL",
            "Endcap",
            constants["HCalEndcap_inner_radius"],
            constants["HCalEndcap_outer_radius"],
            constants["HCalEndcap_min_z"],
            constants["HCalEndcap_max_z"],
            "Fe + PS",
        ),
        DetectorRegion(
            "Muon Detector",
            "Barrel",
            constants["YokeBarrel_inner_radius"],
            constants["YokeBarrel_outer_radius"],
            0.0,
            constants["YokeBarrel_half_length"],
            "Air + RPC",
        ),
        DetectorRegion(
            "Muon Detector",
            "Endcap",
            constants["YokeEndcap_inner_radius"],
            constants["YokeEndcap_outer_radius"],
            constants["YokeEndcap_min_z"],
            constants["YokeEndcap_max_z"],
            "Air + RPC",
        ),
    ]


def load_geometry(geometry_dir: Path) -> GeometryModel:
    roots = load_xml_roots(geometry_dir)
    constants = load_constants(roots)
    barrel_layers, endcap_rings = parse_tracker_layers(roots, constants)
    nozzles = parse_nozzles(roots, constants)
    regions = tracker_regions(barrel_layers, endcap_rings) + detector_envelope_regions(constants)
    return GeometryModel(constants, regions, barrel_layers, endcap_rings, nozzles)


def eta_slope(eta: float) -> float | None:
    """Return r/z for a pseudorapidity ray in the positive-z, positive-r plane."""
    if abs(eta) < 1e-12:
        return None
    theta = 2.0 * math.atan(math.exp(-abs(eta)))
    return math.tan(theta)


def draw_eta_guides(ax: plt.Axes, etas: Iterable[float], z_max: float, r_max: float) -> None:
    for eta in etas:
        color = "#c7c7c7"
        if abs(eta) < 1e-12:
            ax.plot([0.0, 0.0], [0.0, r_max], color=color, lw=1.0, ls="--", zorder=0)
            ax.text(
                0.015 * z_max,
                0.96 * r_max,
                r"$\eta=0$",
                color="#555555",
                fontsize=9,
                va="top",
            )
            continue

        slope = eta_slope(eta)
        if slope is None:
            continue
        z_end = min(z_max, r_max / slope)
        r_end = slope * z_end
        if z_end < 0.015 * z_max or r_end < 0.015 * r_max:
            continue

        ax.plot([0.0, z_end], [0.0, r_end], color=color, lw=1.0, ls="--", zorder=0)
        angle = math.degrees(math.atan2(r_end, z_end))
        x_label = z_end + 0.012 * z_max
        y_label = r_end
        ha = "left"
        va = "center"
        if r_end > 0.92 * r_max:
            x_label = z_end
            y_label = r_end + 0.018 * r_max
            ha = "center"
            va = "bottom"
        ax.text(
            x_label,
            y_label,
            rf"$\eta={eta:g}$",
            color="#555555",
            fontsize=9,
            rotation=angle,
            ha=ha,
            va=va,
            clip_on=False,
        )


def matched_aspect_figure(z_limit: float, r_limit: float, width: float = 10.5) -> tuple[plt.Figure, plt.Axes]:
    height = max(3.0, width * (r_limit / z_limit))
    return plt.subplots(figsize=(width, height), constrained_layout=True)


def draw_region(ax: plt.Axes, region: DetectorRegion, z_max: float, r_max: float) -> None:
    color = SUBSYSTEM_COLORS.get(region.subsystem, "#555555")
    if region.region == "Barrel":
        x0, x1 = 0.0, region.z_high
    else:
        x0, x1 = region.z_low, region.z_high

    rect = Rectangle(
        (x0, region.r_min),
        x1 - x0,
        region.r_max - region.r_min,
        facecolor=color,
        edgecolor=color,
        lw=1.8,
        alpha=0.14 if region.region == "Barrel" else 0.20,
        zorder=2 if region.region == "Barrel" else 4,
    )
    ax.add_patch(rect)

    width = x1 - x0
    height = region.r_max - region.r_min
    if width * height > 0.018 * z_max * r_max:
        x_mid = 0.5 * (x0 + x1)
        y_mid = 0.5 * (region.r_min + region.r_max)
        label = f"{SUBSYSTEM_LABELS.get(region.subsystem, region.subsystem)} {region.region[0]}"
        ax.text(x_mid, y_mid, label, color=color, fontsize=8.5, ha="center", va="center")


def nozzle_style(nozzle: NozzleProfile) -> tuple[str, float]:
    if "BCH" in nozzle.name or nozzle.material == "BCH2":
        return "#4f63c6", 0.42
    return "#4a4a4a", 0.38


def clipped_nozzle_planes(nozzle: NozzleProfile, z_clip: float) -> list[tuple[float, float, float]]:
    planes = list(nozzle.z_planes)
    clipped: list[tuple[float, float, float]] = []
    for index, plane in enumerate(planes):
        z, r_min, r_max = plane
        if z <= z_clip:
            clipped.append(plane)
            continue
        if index == 0:
            break
        z0, r_min0, r_max0 = planes[index - 1]
        if z0 < z_clip and z != z0:
            t = (z_clip - z0) / (z - z0)
            clipped.append((z_clip, r_min0 + t * (r_min - r_min0), r_max0 + t * (r_max - r_max0)))
        break
    return clipped


def draw_nozzles(ax: plt.Axes, nozzles: list[NozzleProfile], z_clip: float, zorder: int = 6) -> None:
    for nozzle in nozzles:
        planes = clipped_nozzle_planes(nozzle, z_clip)
        if len(planes) < 2:
            continue
        outer = [(z, r_max) for z, _, r_max in planes]
        inner = [(z, r_min) for z, r_min, _ in reversed(planes)]
        color, alpha = nozzle_style(nozzle)
        ax.add_patch(
            Polygon(
                outer + inner,
                closed=True,
                facecolor=color,
                edgecolor=color,
                lw=1.0,
                alpha=alpha,
                zorder=zorder,
            )
        )


def draw_rz(model: GeometryModel, output_path: Path, unit: str, etas: list[float], dpi: int) -> None:
    z_max = max(region.z_high for region in model.regions)
    r_max = max(region.r_max for region in model.regions)
    z_limit = z_max * 1.08
    r_limit = r_max * 1.08
    fig, ax = matched_aspect_figure(z_limit, r_limit)

    draw_eta_guides(ax, etas, z_max, r_max)

    for region in model.regions:
        draw_region(ax, region, z_max, r_max)

    draw_nozzles(ax, model.nozzles, z_max)

    legend_items = [
        Line2D([0], [0], color=color, lw=3, label=name)
        for name, color in SUBSYSTEM_COLORS.items()
        if any(region.subsystem == name for region in model.regions)
    ]
    if model.nozzles:
        legend_items.extend(
            [
                Line2D([0], [0], color="#4a4a4a", lw=3, label="Nozzle W"),
                Line2D([0], [0], color="#4f63c6", lw=3, label="Nozzle BCH"),
            ]
        )
    ax.legend(handles=legend_items, loc="upper right", frameon=False, fontsize=9)

    ax.set_title("MAIA detector schematic: r-z view", fontsize=15, pad=12)
    ax.set_xlabel(f"|z| [{unit}]", fontsize=13)
    ax.set_ylabel(f"r [{unit}]", fontsize=13)
    ax.set_xlim(0.0, z_limit)
    ax.set_ylim(0.0, r_limit)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#ededed", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def boundary_radii(model: GeometryModel, source: str) -> list[float]:
    chosen_regions = [region for region in model.regions if source == "all" or region.region == "Barrel"]
    region_radii = {round(region.r_min, 6) for region in chosen_regions} | {
        round(region.r_max, 6) for region in chosen_regions
    }
    layer_radii = {round(layer.radius, 6) for layer in model.barrel_layers if layer.subsystem in TRACKER_SUBSYSTEMS}
    return [float(value) for value in sorted(region_radii | layer_radii)]


def draw_xy(model: GeometryModel, output_path: Path, unit: str, label_source: str, dpi: int) -> None:
    barrel_regions = [region for region in model.regions if region.region == "Barrel"]
    if not barrel_regions:
        raise ValueError("No barrel rows were found for the x-y view.")

    fig, ax = plt.subplots(figsize=(11.0, 11.0), constrained_layout=True)
    r_max = max(region.r_max for region in model.regions)

    for region in sorted(barrel_regions, key=lambda item: item.r_max, reverse=True):
        color = SUBSYSTEM_COLORS.get(region.subsystem, "#555555")
        width = max(region.r_max - region.r_min, 0.001 * r_max)
        annulus = Wedge(
            (0.0, 0.0),
            region.r_max,
            0.0,
            360.0,
            width=width,
            facecolor=color,
            edgecolor=color,
            lw=1.4,
            alpha=0.17,
        )
        ax.add_patch(annulus)
        ax.add_patch(Circle((0.0, 0.0), region.r_min, fill=False, edgecolor=color, lw=1.0, alpha=0.75))
        ax.add_patch(Circle((0.0, 0.0), region.r_max, fill=False, edgecolor=color, lw=1.6, alpha=0.95))

    for layer in model.barrel_layers:
        color = SUBSYSTEM_COLORS.get(layer.subsystem, "#555555")
        ax.add_patch(Circle((0.0, 0.0), layer.radius, fill=False, edgecolor=color, lw=0.9, alpha=0.9))

    radii = boundary_radii(model, label_source)
    label_x = -1.28 * r_max
    point_angle = math.radians(145.0)
    y_positions = np.linspace(0.92 * r_max, -0.92 * r_max, len(radii))

    for radius, y_text in zip(sorted(radii, reverse=True), y_positions):
        x_target = radius * math.cos(point_angle)
        y_target = radius * math.sin(point_angle)
        ax.plot([label_x * 0.96, x_target], [y_text, y_target], color="#2a2a2a", lw=0.75)
        ax.text(
            label_x,
            y_text,
            fmt_radius(radius, unit),
            ha="right",
            va="center",
            fontsize=8,
            color="#111111",
        )

    legend_items = [
        Line2D([0], [0], color=color, lw=4, label=name)
        for name, color in SUBSYSTEM_COLORS.items()
        if any(region.subsystem == name for region in barrel_regions)
    ]
    ax.legend(handles=legend_items, loc="upper right", frameon=False, fontsize=9)

    ax.axhline(0.0, color="#dddddd", lw=0.8, zorder=0)
    ax.axvline(0.0, color="#dddddd", lw=0.8, zorder=0)
    ax.set_title("MAIA detector schematic: x-y view", fontsize=15, pad=12)
    ax.set_xlabel(f"x [{unit}]", fontsize=13)
    ax.set_ylabel(f"y [{unit}]", fontsize=13)
    ax.set_xlim(-1.50 * r_max, 1.12 * r_max)
    ax.set_ylim(-1.08 * r_max, 1.08 * r_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#eeeeee", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def draw_tracker_layers(model: GeometryModel, output_path: Path, unit: str, etas: list[float], dpi: int) -> None:
    tracker_regions_only = [region for region in model.regions if region.subsystem in TRACKER_SUBSYSTEMS]
    z_max = max(
        [region.z_high for region in tracker_regions_only]
        + [layer.z_half for layer in model.barrel_layers]
        + [ring.z for ring in model.endcap_rings]
    )
    r_max = max(
        [region.r_max for region in tracker_regions_only]
        + [layer.radius for layer in model.barrel_layers]
        + [ring.r_max for ring in model.endcap_rings]
    )
    z_limit = z_max * 1.10
    r_limit = r_max * 1.13
    fig, ax = matched_aspect_figure(z_limit, r_limit, width=11.0)

    draw_eta_guides(ax, etas, z_max, r_max)

    for region in tracker_regions_only:
        color = SUBSYSTEM_COLORS[region.subsystem]
        if region.region == "Barrel":
            rect = Rectangle(
                (0.0, region.r_min),
                region.z_high,
                region.r_max - region.r_min,
                facecolor=color,
                edgecolor="none",
                alpha=0.045,
                zorder=1,
            )
        else:
            rect = Rectangle(
                (region.z_low, region.r_min),
                region.z_high - region.z_low,
                region.r_max - region.r_min,
                facecolor=color,
                edgecolor="none",
                alpha=0.055,
                zorder=1,
            )
        ax.add_patch(rect)

    draw_nozzles(ax, model.nozzles, z_limit, zorder=2)

    subsystem_order = {name: index for index, name in enumerate(TRACKER_SUBSYSTEMS)}

    def layer_key(layer: BarrelLayer | EndcapRing) -> tuple[int, float]:
        return (subsystem_order.get(layer.subsystem, 99), float(layer.layer_id))

    for layer in sorted(model.barrel_layers, key=layer_key):
        color = SUBSYSTEM_COLORS[layer.subsystem]
        ax.plot([0.0, layer.z_half], [layer.radius, layer.radius], color=color, lw=2.1, zorder=3)
        ax.plot([layer.z_half, layer.z_half], [layer.radius - 0.012 * r_max, layer.radius + 0.012 * r_max], color=color, lw=1.2)
        if layer.radius > 0.055 * r_max:
            ax.text(
                max(0.012 * z_max, layer.z_half - 0.020 * z_max),
                layer.radius,
                f"B{layer.layer_id}",
                color=color,
                fontsize=7.5,
                va="center",
                ha="right",
            )

    grouped_endcap_layers: dict[tuple[str, str], list[EndcapRing]] = {}
    for ring in model.endcap_rings:
        grouped_endcap_layers.setdefault((ring.subsystem, ring.layer_id), []).append(ring)

    for label_index, ((subsystem, layer_id), rings) in enumerate(
        sorted(grouped_endcap_layers.items(), key=lambda item: layer_key(item[1][0]))
    ):
        color = SUBSYSTEM_COLORS[subsystem]
        z = rings[0].z
        for ring in rings:
            ax.plot([ring.z, ring.z], [ring.r_min, ring.r_max], color=color, lw=2.0, solid_capstyle="butt", zorder=4)
        r_label = max(ring.r_max for ring in rings)
        compact_vertex = subsystem == "Vertex Detector"
        x_offset = ((label_index % 3) - 1) * 0.012 * z_max if compact_vertex else 0.0
        y_offset = (0.016 + 0.012 * (label_index % 2)) * r_max if compact_vertex else 0.018 * r_max
        ax.text(
            z + x_offset,
            r_label + y_offset,
            f"E{layer_id}",
            color=color,
            fontsize=7.5,
            ha="center",
            va="bottom",
            rotation=0 if compact_vertex else 90,
        )

    legend_items = [
        Line2D([0], [0], color=SUBSYSTEM_COLORS[name], lw=3, label=name)
        for name in TRACKER_SUBSYSTEMS
        if any(layer.subsystem == name for layer in model.barrel_layers)
    ]
    if model.nozzles:
        legend_items.append(Line2D([0], [0], color="#4a4a4a", lw=3, label="Nozzle"))
    ax.legend(handles=legend_items, loc="upper right", frameon=False, fontsize=9)

    ax.set_title("MAIA tracker layers: r-z view", fontsize=15, pad=12)
    ax.set_xlabel(f"|z| [{unit}]", fontsize=13)
    ax.set_ylabel(f"r [{unit}]", fontsize=13)
    ax.set_xlim(0.0, z_limit)
    ax.set_ylim(0.0, r_limit)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#eeeeee", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def tracker_layer_sort_key(layer: BarrelLayer | EndcapRing) -> tuple[int, float]:
    subsystem_order = {name: index for index, name in enumerate(TRACKER_SUBSYSTEMS)}
    return (subsystem_order.get(layer.subsystem, 99), float(layer.layer_id))


def surface_for_cylinder(radius: float, z_half: float, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, phi_grid = np.meshgrid(np.linspace(0.0, z_half, 8), phi)
    y = radius * np.cos(phi_grid)
    z = radius * np.sin(phi_grid)
    return x, y, z


def surface_for_annular_sector(
    x_position: float,
    r_min: float,
    r_max: float,
    phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radius_grid, phi_grid = np.meshgrid(np.linspace(r_min, r_max, 3), phi)
    x = np.full_like(radius_grid, x_position)
    y = radius_grid * np.cos(phi_grid)
    z = radius_grid * np.sin(phi_grid)
    return x, y, z


def draw_3d_radius_callouts(
    ax: plt.Axes,
    layers: list[BarrelLayer],
    z_max: float,
    r_max: float,
    unit: str,
    phi_anchor: float,
) -> None:
    label_x = -0.72 * z_max
    label_y = -1.42 * r_max
    label_z_positions = np.linspace(1.10 * r_max, -0.07 * r_max, len(layers))

    for layer, label_z in zip(sorted(layers, key=lambda item: item.radius, reverse=True), label_z_positions):
        target = (
            layer.z_half,
            layer.radius * math.cos(phi_anchor),
            layer.radius * math.sin(phi_anchor),
        )
        start = (label_x, label_y, label_z)
        ax.plot(
            [start[0], target[0]],
            [start[1], target[1]],
            [start[2], target[2]],
            color="#202020",
            lw=0.75,
            alpha=0.9,
            zorder=20,
        )
        ax.text(
            start[0],
            start[1],
            start[2],
            fmt_radius(layer.radius, unit),
            color="#111111",
            fontsize=7.5,
            ha="right",
            va="center",
            zorder=21,
        )


def draw_tracker_3d(model: GeometryModel, output_path: Path, unit: str, dpi: int) -> None:
    fig = plt.figure(figsize=(12.2, 8.4), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    tracker_layers = [layer for layer in model.barrel_layers if layer.subsystem in TRACKER_SUBSYSTEMS]
    tracker_rings = [ring for ring in model.endcap_rings if ring.subsystem in TRACKER_SUBSYSTEMS]
    z_max = max([layer.z_half for layer in tracker_layers] + [ring.z for ring in tracker_rings])
    r_max = max([layer.radius for layer in tracker_layers] + [ring.r_max for ring in tracker_rings])

    phi = np.linspace(math.radians(24), math.radians(152), 112)
    phi_anchor = math.radians(152)

    # Draw outer layers first so the smaller inner layers remain visible.
    for layer in sorted(tracker_layers, key=lambda item: item.radius, reverse=True):
        color = SUBSYSTEM_COLORS[layer.subsystem]
        x, y, z = surface_for_cylinder(layer.radius, layer.z_half, phi)
        ax.plot_surface(x, y, z, color=color, alpha=1.0, linewidth=0, shade=False, zorder=4)
        ax.plot(x[:, 0], y[:, 0], z[:, 0], color=color, lw=1.1, alpha=1.0, zorder=8)
        ax.plot(x[:, -1], y[:, -1], z[:, -1], color=color, lw=1.0, alpha=1.0, zorder=7)
        ax.plot(x[0, :], y[0, :], z[0, :], color=color, lw=0.8, alpha=1.0, zorder=7)
        ax.plot(x[-1, :], y[-1, :], z[-1, :], color=color, lw=0.8, alpha=1.0, zorder=7)

    grouped_rings: dict[tuple[str, str], list[EndcapRing]] = {}
    for ring in tracker_rings:
        grouped_rings.setdefault((ring.subsystem, ring.layer_id), []).append(ring)

    for (_, _), rings in sorted(grouped_rings.items(), key=lambda item: tracker_layer_sort_key(item[1][0])):
        color = SUBSYSTEM_COLORS[rings[0].subsystem]
        for ring in rings:
            x, y, z = surface_for_annular_sector(ring.z, ring.r_min, ring.r_max, phi)
            ax.plot_surface(x, y, z, color=color, alpha=1.0, linewidth=0, shade=False, zorder=5)
            ax.plot(x[:, 0], y[:, 0], z[:, 0], color=color, lw=0.8, alpha=1.0, zorder=8)
            ax.plot(x[:, -1], y[:, -1], z[:, -1], color=color, lw=1.0, alpha=1.0, zorder=9)

    # A slim beam line anchors the perspective without introducing non-tracker geometry.
    ax.plot([0.0, z_max * 1.03], [0.0, 0.0], [0.0, 0.0], color="#9a9a9a", lw=1.5, alpha=0.8)

    radius_layers = {
        round(layer.radius, 6): layer
        for layer in sorted(tracker_layers, key=lambda item: item.z_half, reverse=True)
    }
    draw_3d_radius_callouts(ax, list(radius_layers.values()), z_max, r_max, unit, phi_anchor)

    legend_items = [
        Line2D([0], [0], color=SUBSYSTEM_COLORS[name], lw=5, label=name)
        for name in TRACKER_SUBSYSTEMS
    ]
    ax.legend(handles=legend_items, loc="upper right", frameon=False, fontsize=10)

    ax.set_title("MAIA tracker schematic: perspective view", fontsize=16, pad=12)
    ax.set_xlim(-0.52 * z_max, 1.08 * z_max)
    ax.set_ylim(-1.40 * r_max, 1.02 * r_max)
    ax.set_zlim(-0.02 * r_max, 1.16 * r_max)
    ax.set_box_aspect((1.60 * z_max, 2.42 * r_max, 1.18 * r_max))
    ax.view_init(elev=19, azim=-122)
    ax.set_axis_off()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


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
        default=Path("maia_schematics"),
        help="Directory for the output images.",
    )
    parser.add_argument(
        "--unit",
        choices=("mm", "cm"),
        default="mm",
        help="Plot unit. The DD4hep XML is evaluated in millimeters.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        nargs="*",
        default=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
        help="Eta values to superimpose on the r-z views.",
    )
    parser.add_argument(
        "--xy-radii",
        choices=("barrel", "all"),
        default="barrel",
        help="Which detector-region R boundaries to label in the x-y plot.",
    )
    parser.add_argument("--dpi", type=int, default=220, help="Output image resolution.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = load_geometry(args.geometry_dir).scaled(unit_factor(args.unit))
    etas = sorted(set(abs(eta) for eta in args.eta))

    rz_path = args.output_dir / f"maia_rz.{args.unit}.png"
    xy_path = args.output_dir / f"maia_xy.{args.unit}.png"
    tracker_path = args.output_dir / f"maia_tracker_layers_rz.{args.unit}.png"
    tracker_3d_path = args.output_dir / f"maia_tracker_perspective.{args.unit}.png"

    draw_rz(model, rz_path, args.unit, etas, args.dpi)
    draw_xy(model, xy_path, args.unit, args.xy_radii, args.dpi)
    draw_tracker_layers(model, tracker_path, args.unit, etas, args.dpi)
    draw_tracker_3d(model, tracker_3d_path, args.unit, args.dpi)

    print(f"Read geometry from {args.geometry_dir}")
    print(f"Wrote {rz_path}")
    print(f"Wrote {xy_path}")
    print(f"Wrote {tracker_path}")
    print(f"Wrote {tracker_3d_path}")


if __name__ == "__main__":
    main()
