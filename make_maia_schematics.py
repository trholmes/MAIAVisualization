#!/usr/bin/env python3
"""Make MAIA detector schematic plots from DD4hep compact XML geometry.

The script reads the MAIA_v0 geometry directory, evaluates the XML constants,
and produces:

  1. PNG and PDF r-z overview with pseudorapidity guide lines.
  2. PNG and PDF x-y overview with detector barrel radii labeled.
  3. PNG and PDF tracker-only r-z view showing Vertex, Inner Tracker, and Outer Tracker
     barrel layers and endcap rings explicitly.
  4. PNG and PDF tracker-only perspective schematic approximating the x-y/3D barrel view.

Usage:
    python3 make_maia_schematics.py
    python3 make_maia_schematics.py --geometry-dir path/to/MAIA_v0
    python3 make_maia_schematics.py --output-dir figures --unit cm
    python3 make_maia_schematics.py --full-rz-tracker layers
    python3 make_maia_schematics.py --interactive-perspective
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
from matplotlib.colors import LightSource
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon, Wedge
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d


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
DEFAULT_PERSPECTIVE_ELEV = 10.0
DEFAULT_PERSPECTIVE_AZIM = -140.0
DEFAULT_PERSPECTIVE_ROLL = 0.0
DEFAULT_PERSPECTIVE_FOCAL_LENGTH: float | None = None
DEFAULT_PERSPECTIVE_DISTANCE = 10.0


@dataclass(frozen=True)
class DetectorRegion:
    subsystem: str
    region: str
    r_min: float
    r_max: float
    z_low: float
    z_high: float
    material: str = ""
    r_min_high: float | None = None

    def scaled(self, factor: float) -> "DetectorRegion":
        return DetectorRegion(
            subsystem=self.subsystem,
            region=self.region,
            r_min=self.r_min * factor,
            r_max=self.r_max * factor,
            z_low=self.z_low * factor,
            z_high=self.z_high * factor,
            material=self.material,
            r_min_high=None if self.r_min_high is None else self.r_min_high * factor,
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
            r_min_high=constants["HCalEndcap_inner_radius2"],
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
            r_min_high=constants["YokeEndcap_inner_radius2"],
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


def save_figure_outputs(fig: plt.Figure, output_path: Path, dpi: int, **savefig_kwargs: object) -> tuple[Path, Path]:
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(output_path, dpi=dpi, **savefig_kwargs)
    fig.savefig(pdf_path, **savefig_kwargs)
    return output_path, pdf_path


def nozzle_outer_radius_at(nozzles: list[NozzleProfile], z: float) -> float | None:
    outer_radii: list[float] = []
    for nozzle in nozzles:
        planes = nozzle.z_planes
        for z_plane, _, r_outer in planes:
            if abs(z - z_plane) < 1e-9:
                outer_radii.append(r_outer)
        for (z0, _, r0), (z1, _, r1) in zip(planes, planes[1:]):
            if abs(z1 - z0) < 1e-12:
                continue
            if min(z0, z1) <= z <= max(z0, z1):
                fraction = (z - z0) / (z1 - z0)
                outer_radii.append(r0 + fraction * (r1 - r0))
                break
    return max(outer_radii) if outer_radii else None


def region_inner_radius_at(region: DetectorRegion, z: float) -> float:
    if region.region != "Endcap" or region.r_min_high is None or abs(region.z_high - region.z_low) < 1e-12:
        return region.r_min
    fraction = (z - region.z_low) / (region.z_high - region.z_low)
    return region.r_min + fraction * (region.r_min_high - region.r_min)


def region_polygon_points(region: DetectorRegion, nozzles: list[NozzleProfile]) -> list[tuple[float, float]]:
    if region.region == "Barrel":
        z_low, z_high = 0.0, region.z_high
    else:
        z_low, z_high = region.z_low, region.z_high

    z_samples = {z_low, z_high}
    for nozzle in nozzles:
        for z_plane, _, _ in nozzle.z_planes:
            if z_low <= z_plane <= z_high:
                z_samples.add(z_plane)

    lower_edge: list[tuple[float, float]] = []
    for z in sorted(z_samples):
        nozzle_r = nozzle_outer_radius_at(nozzles, z)
        geometry_inner_radius = region_inner_radius_at(region, z)
        inner_radius = geometry_inner_radius if nozzle_r is None else max(geometry_inner_radius, nozzle_r)
        lower_edge.append((z, min(inner_radius, region.r_max)))

    return lower_edge + [(z_high, region.r_max), (z_low, region.r_max)]


def draw_region(
    ax: plt.Axes,
    region: DetectorRegion,
    z_max: float,
    r_max: float,
    nozzles: list[NozzleProfile],
) -> None:
    color = SUBSYSTEM_COLORS.get(region.subsystem, "#555555")
    if region.region == "Barrel":
        x0, x1 = 0.0, region.z_high
    else:
        x0, x1 = region.z_low, region.z_high

    polygon_points = region_polygon_points(region, nozzles)
    polygon = Polygon(
        polygon_points,
        closed=True,
        facecolor=color,
        edgecolor=color,
        lw=1.8,
        alpha=0.14 if region.region == "Barrel" else 0.20,
        zorder=2 if region.region == "Barrel" else 4,
    )
    ax.add_patch(polygon)

    width = x1 - x0
    height = region.r_max - region.r_min
    if width * height > 0.018 * z_max * r_max:
        x_mid = sum(point[0] for point in polygon_points) / len(polygon_points)
        y_mid = sum(point[1] for point in polygon_points) / len(polygon_points)
        label = f"{SUBSYSTEM_LABELS.get(region.subsystem, region.subsystem)} {region.region[0]}"
        ax.text(x_mid, y_mid, label, color=color, fontsize=8.5, ha="center", va="center")


def layer_radial_bands(
    radii: Iterable[float],
    fallback_width: float,
    min_radius: float | None = None,
    max_radius: float | None = None,
) -> dict[float, tuple[float, float]]:
    sorted_radii = sorted(set(round(radius, 9) for radius in radii))
    if not sorted_radii:
        return {}

    def clamped(radius: float, lower: float, upper: float) -> tuple[float, float]:
        if min_radius is not None:
            lower = max(min_radius, lower)
        if max_radius is not None:
            upper = min(max_radius, upper)
        if upper < lower:
            lower = upper = radius
        return lower, upper

    if len(sorted_radii) == 1:
        radius = sorted_radii[0]
        half_width = 0.5 * fallback_width
        return {radius: clamped(radius, max(0.0, radius - half_width), radius + half_width)}

    bands: dict[float, tuple[float, float]] = {}
    for index, radius in enumerate(sorted_radii):
        if index == 0:
            inner_gap = sorted_radii[1] - radius
        else:
            inner_gap = radius - sorted_radii[index - 1]
        if index == len(sorted_radii) - 1:
            outer_gap = radius - sorted_radii[index - 1]
        else:
            outer_gap = sorted_radii[index + 1] - radius
        bands[radius] = clamped(radius, max(0.0, radius - 0.45 * inner_gap), radius + 0.45 * outer_gap)
    return bands


def tracker_subsystem_radial_bounds(model: GeometryModel, subsystem: str, barrel_only: bool = False) -> tuple[float, float]:
    regions = [
        region
        for region in model.regions
        if region.subsystem == subsystem and (not barrel_only or region.region == "Barrel")
    ]
    if not regions:
        raise ValueError(f"No tracker regions found for {subsystem}")
    return min(region.r_min for region in regions), max(region.r_max for region in regions)


def nozzle_z_samples_for_radius(
    nozzles: list[NozzleProfile],
    z_low: float,
    z_high: float,
    radius: float,
) -> set[float]:
    z_samples = {z_low, z_high}
    for nozzle in nozzles:
        planes = nozzle.z_planes
        for z_plane, _, r_outer in planes:
            if z_low <= z_plane <= z_high:
                z_samples.add(z_plane)
            if abs(r_outer - radius) < 1e-9 and z_low <= z_plane <= z_high:
                z_samples.add(z_plane)

        for (z0, _, r0), (z1, _, r1) in zip(planes, planes[1:]):
            if z1 == z0 or max(z0, z1) < z_low or min(z0, z1) > z_high:
                continue
            delta0 = r0 - radius
            delta1 = r1 - radius
            if delta0 == 0.0:
                z_samples.add(min(max(z0, z_low), z_high))
            if delta0 * delta1 < 0.0:
                fraction = -delta0 / (delta1 - delta0)
                z_crossing = z0 + fraction * (z1 - z0)
                if z_low <= z_crossing <= z_high:
                    z_samples.add(z_crossing)
    return z_samples


def nozzle_clipped_slab_points(
    z_low: float,
    z_high: float,
    r_low: float,
    r_high: float,
    nozzles: list[NozzleProfile],
) -> list[tuple[float, float]]:
    z_samples = sorted(nozzle_z_samples_for_radius(nozzles, z_low, z_high, r_low))
    lower_edge: list[tuple[float, float]] = []
    for z in z_samples:
        nozzle_r = nozzle_outer_radius_at(nozzles, z)
        inner_radius = r_low if nozzle_r is None else max(r_low, nozzle_r)
        lower_edge.append((z, min(inner_radius, r_high)))
    upper_edge = [(z, r_high) for z in reversed(z_samples)]
    return lower_edge + upper_edge


def tracker_rz_envelope_polygons(
    layers: list[BarrelLayer],
    rings: list[EndcapRing],
    r_max: float,
    min_radius: float,
    max_radius: float,
    nozzles: list[NozzleProfile],
) -> list[list[tuple[float, float]]]:
    polygons: list[list[tuple[float, float]]] = []
    barrel_layer_pad = 0.006 * r_max

    for layer in layers:
        r_low = max(min_radius, layer.radius - barrel_layer_pad)
        r_high = min(max_radius, layer.radius + barrel_layer_pad)
        polygons.append(
            nozzle_clipped_slab_points(
                0.0,
                layer.z_half,
                r_low,
                r_high,
                nozzles,
            )
        )

    rings_by_disk: dict[tuple[str, float], list[EndcapRing]] = {}
    for ring in rings:
        rings_by_disk.setdefault((ring.layer_id, round(ring.z, 9)), []).append(ring)

    for (_, _), disk_rings in sorted(
        rings_by_disk.items(),
        key=lambda item: (float(item[1][0].layer_id), item[1][0].z),
    ):
        ring_z = disk_rings[0].z
        ring_r_min = min(ring.r_min for ring in disk_rings)
        ring_r_max = max(ring.r_max for ring in disk_rings)
        widest_ring = max(ring.r_max - ring.r_min for ring in disk_rings)
        radial_pad = max(0.0025 * r_max, 0.03 * max(widest_ring, 0.0))
        z_pad = max(0.003 * ring_z, 0.006 * r_max)
        r_low = max(min_radius, max(0.0, ring_r_min - radial_pad))
        r_high = min(max_radius, ring_r_max + radial_pad)
        polygons.append(
            nozzle_clipped_slab_points(
                max(0.0, ring_z - z_pad),
                ring_z + z_pad,
                r_low,
                r_high,
                nozzles,
            )
        )

    return polygons


def polygon_area_and_centroid(points: list[tuple[float, float]]) -> tuple[float, tuple[float, float]]:
    twice_area = 0.0
    centroid_x = 0.0
    centroid_y = 0.0
    for point_a, point_b in zip(points, points[1:] + points[:1]):
        cross = point_a[0] * point_b[1] - point_b[0] * point_a[1]
        twice_area += cross
        centroid_x += (point_a[0] + point_b[0]) * cross
        centroid_y += (point_a[1] + point_b[1]) * cross

    if abs(twice_area) < 1e-12:
        return 0.0, (
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
        )

    area = 0.5 * twice_area
    return abs(area), (centroid_x / (3.0 * twice_area), centroid_y / (3.0 * twice_area))


def draw_tracker_rz_envelopes(
    ax: plt.Axes,
    model: GeometryModel,
    z_max: float,
    r_max: float,
    alpha: float = 0.16,
    zorder: int = 3,
    label: bool = True,
) -> None:
    for subsystem in TRACKER_SUBSYSTEMS:
        layers = [layer for layer in model.barrel_layers if layer.subsystem == subsystem]
        rings = [ring for ring in model.endcap_rings if ring.subsystem == subsystem]
        min_radius, max_radius = tracker_subsystem_radial_bounds(model, subsystem)
        polygons = tracker_rz_envelope_polygons(layers, rings, r_max, min_radius, max_radius, model.nozzles)
        polygons = [polygon for polygon in polygons if len(polygon) >= 3]
        if not polygons:
            continue

        color = SUBSYSTEM_COLORS[subsystem]
        for polygon_points in polygons:
            ax.add_patch(
                Polygon(
                    polygon_points,
                    closed=True,
                    facecolor=color,
                    edgecolor=color,
                    lw=0.8,
                    alpha=alpha,
                    zorder=zorder,
                )
            )

        if not label:
            continue

        areas_and_centroids = [polygon_area_and_centroid(polygon_points) for polygon_points in polygons]
        total_area = sum(area for area, _ in areas_and_centroids)
        if total_area > 0.018 * z_max * r_max:
            centroid_x = sum(area * centroid[0] for area, centroid in areas_and_centroids) / total_area
            centroid_y = sum(area * centroid[1] for area, centroid in areas_and_centroids) / total_area
            ax.text(
                centroid_x,
                centroid_y,
                SUBSYSTEM_LABELS.get(subsystem, subsystem),
                color=color,
                fontsize=8.5,
                ha="center",
                va="center",
            )


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


def draw_tracker_rz_layer_lines(
    ax: plt.Axes,
    model: GeometryModel,
    z_max: float,
    r_max: float,
    label: bool = True,
    barrel_zorder: int = 3,
    endcap_zorder: int = 4,
    linewidth_scale: float = 1.0,
) -> None:
    subsystem_order = {name: index for index, name in enumerate(TRACKER_SUBSYSTEMS)}

    def layer_key(layer: BarrelLayer | EndcapRing) -> tuple[int, float]:
        return (subsystem_order.get(layer.subsystem, 99), float(layer.layer_id))

    def label_vertex_barrel_layers(layers: list[BarrelLayer]) -> None:
        consumed: set[int] = set()
        sorted_layers = sorted(layers, key=lambda item: item.radius)
        for index, layer in enumerate(sorted_layers):
            if index in consumed:
                continue

            paired_layer: BarrelLayer | None = None
            for other_index in range(index + 1, len(sorted_layers)):
                other_layer = sorted_layers[other_index]
                if abs(other_layer.radius - layer.radius) < 0.010 * r_max:
                    paired_layer = other_layer
                    consumed.add(other_index)
                    break

            label_text = f"B{layer.layer_id}"
            label_radius = layer.radius
            if paired_layer is not None:
                label_text = f"B{layer.layer_id}/{paired_layer.layer_id}"
                label_radius = 0.5 * (layer.radius + paired_layer.radius)

            ax.text(
                -0.006 * z_max,
                label_radius,
                label_text,
                color=SUBSYSTEM_COLORS[layer.subsystem],
                fontsize=7.5,
                va="center",
                ha="right",
                clip_on=False,
            )

    def label_endcap_groups(
        subsystem: str,
        grouped_layers: list[tuple[str, list[EndcapRing]]],
    ) -> list[tuple[str, list[EndcapRing]]]:
        if subsystem != "Vertex Detector":
            return grouped_layers

        label_groups: list[tuple[str, list[EndcapRing]]] = []
        consumed: set[int] = set()
        for index, (layer_id, rings) in enumerate(grouped_layers):
            if index in consumed:
                continue

            match_index: int | None = None
            z = rings[0].z
            r_min = min(ring.r_min for ring in rings)
            r_max_local = max(ring.r_max for ring in rings)
            for other_index in range(index + 1, len(grouped_layers)):
                other_layer_id, other_rings = grouped_layers[other_index]
                other_z = other_rings[0].z
                other_r_min = min(ring.r_min for ring in other_rings)
                other_r_max = max(ring.r_max for ring in other_rings)
                same_span = abs(other_r_min - r_min) < 1e-9 and abs(other_r_max - r_max_local) < 1e-9
                if same_span and abs(other_z - z) < 0.020 * z_max:
                    match_index = other_index
                    break

            if match_index is None:
                label_groups.append((layer_id, rings))
                continue

            other_layer_id, other_rings = grouped_layers[match_index]
            consumed.add(match_index)
            label_groups.append((f"{layer_id}/{other_layer_id}", rings + other_rings))

        return label_groups

    sorted_barrel_layers = sorted(model.barrel_layers, key=layer_key)
    for layer in sorted_barrel_layers:
        color = SUBSYSTEM_COLORS[layer.subsystem]
        ax.plot(
            [0.0, layer.z_half],
            [layer.radius, layer.radius],
            color=color,
            lw=2.1 * linewidth_scale,
            zorder=barrel_zorder,
        )
        if not label:
            continue
        if layer.subsystem == "Vertex Detector":
            continue
        elif layer.radius > 0.055 * r_max:
            ax.text(
                max(0.012 * z_max, layer.z_half - 0.020 * z_max),
                layer.radius + 0.010 * r_max,
                f"B{layer.layer_id}",
                color=color,
                fontsize=7.5,
                va="bottom",
                ha="right",
            )

    if label:
        label_vertex_barrel_layers([layer for layer in sorted_barrel_layers if layer.subsystem == "Vertex Detector"])

    grouped_endcap_layers: dict[tuple[str, str], list[EndcapRing]] = {}
    for ring in model.endcap_rings:
        grouped_endcap_layers.setdefault((ring.subsystem, ring.layer_id), []).append(ring)

    grouped_by_subsystem: dict[str, list[tuple[str, list[EndcapRing]]]] = {}
    for (subsystem, layer_id), rings in grouped_endcap_layers.items():
        grouped_by_subsystem.setdefault(subsystem, []).append((layer_id, rings))

    label_items: list[tuple[int, str, str, list[EndcapRing]]] = []
    for subsystem, grouped_layers in grouped_by_subsystem.items():
        ordered_layers = sorted(grouped_layers, key=lambda item: layer_key(item[1][0]))
        for layer_id, rings in label_endcap_groups(subsystem, ordered_layers):
            label_items.append((subsystem_order.get(subsystem, 99), subsystem, layer_id, rings))

    for label_index, (_, subsystem, layer_id, rings) in enumerate(
        sorted(label_items, key=lambda item: (item[0], float(item[2].split("/")[0])))
    ):
        color = SUBSYSTEM_COLORS[subsystem]
        for ring in rings:
            ax.plot(
                [ring.z, ring.z],
                [ring.r_min, ring.r_max],
                color=color,
                lw=2.0 * linewidth_scale,
                solid_capstyle="butt",
                zorder=endcap_zorder,
            )
        if not label:
            continue
        z = sum(ring.z for ring in rings) / len(rings)
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


def draw_rz(
    model: GeometryModel,
    output_path: Path,
    unit: str,
    etas: list[float],
    dpi: int,
    tracker_style: str,
) -> tuple[Path, Path]:
    z_max = max(region.z_high for region in model.regions)
    r_max = max(region.r_max for region in model.regions)
    z_limit = z_max * 1.08
    r_limit = r_max * 1.08
    fig, ax = matched_aspect_figure(z_limit, r_limit)

    draw_eta_guides(ax, etas, z_max, r_max)

    for region in model.regions:
        if region.subsystem in TRACKER_SUBSYSTEMS:
            continue
        draw_region(ax, region, z_max, r_max, model.nozzles)
    if tracker_style == "layers":
        draw_tracker_rz_layer_lines(
            ax,
            model,
            z_max,
            r_max,
            label=False,
            barrel_zorder=5,
            endcap_zorder=5,
            linewidth_scale=0.85,
        )
    else:
        draw_tracker_rz_envelopes(ax, model, z_max, r_max)

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
    saved_paths = save_figure_outputs(fig, output_path, dpi)
    plt.close(fig)
    return saved_paths


def boundary_radii(model: GeometryModel, source: str) -> list[float]:
    chosen_regions = [region for region in model.regions if source == "all" or region.region == "Barrel"]
    region_radii = {round(region.r_min, 6) for region in chosen_regions} | {
        round(region.r_max, 6) for region in chosen_regions
    }
    layer_radii = {round(layer.radius, 6) for layer in model.barrel_layers if layer.subsystem in TRACKER_SUBSYSTEMS}
    return [float(value) for value in sorted(region_radii | layer_radii)]


def draw_tracker_xy_layer_bands(ax: plt.Axes, model: GeometryModel, r_max: float) -> None:
    for subsystem in TRACKER_SUBSYSTEMS:
        layers = [layer for layer in model.barrel_layers if layer.subsystem == subsystem]
        if not layers:
            continue

        color = SUBSYSTEM_COLORS[subsystem]
        min_radius, max_radius = tracker_subsystem_radial_bounds(model, subsystem, barrel_only=True)
        bands = layer_radial_bands(
            (layer.radius for layer in layers),
            0.006 * r_max,
            min_radius=min_radius,
            max_radius=max_radius,
        )
        for radius in sorted(bands):
            r_inner, r_outer = bands[radius]
            width = max(r_outer - r_inner, 0.001 * r_max)
            ax.add_patch(
                Wedge(
                    (0.0, 0.0),
                    r_outer,
                    0.0,
                    360.0,
                    width=width,
                    facecolor=color,
                    edgecolor=color,
                    lw=0.7,
                    alpha=0.13,
                )
            )


def draw_xy(model: GeometryModel, output_path: Path, unit: str, label_source: str, dpi: int) -> tuple[Path, Path]:
    barrel_regions = [region for region in model.regions if region.region == "Barrel"]
    if not barrel_regions:
        raise ValueError("No barrel rows were found for the x-y view.")

    fig, ax = plt.subplots(figsize=(11.0, 11.0), constrained_layout=True)
    r_max = max(region.r_max for region in model.regions)

    for region in sorted(barrel_regions, key=lambda item: item.r_max, reverse=True):
        if region.subsystem in TRACKER_SUBSYSTEMS:
            continue
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

    draw_tracker_xy_layer_bands(ax, model, r_max)

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
    saved_paths = save_figure_outputs(fig, output_path, dpi)
    plt.close(fig)
    return saved_paths


def draw_tracker_layers(model: GeometryModel, output_path: Path, unit: str, etas: list[float], dpi: int) -> tuple[Path, Path]:
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

    draw_tracker_rz_envelopes(ax, model, z_max, r_max, alpha=0.07, zorder=1, label=False)

    draw_nozzles(ax, model.nozzles, z_limit, zorder=2)

    draw_tracker_rz_layer_lines(ax, model, z_max, r_max)

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
    saved_paths = save_figure_outputs(fig, output_path, dpi)
    plt.close(fig)
    return saved_paths


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


def surface_grid_faces(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> list[list[tuple[float, float, float]]]:
    faces: list[list[tuple[float, float, float]]] = []
    for row in range(x.shape[0] - 1):
        for col in range(x.shape[1] - 1):
            faces.append(
                [
                    (float(x[row, col]), float(y[row, col]), float(z[row, col])),
                    (float(x[row + 1, col]), float(y[row + 1, col]), float(z[row + 1, col])),
                    (float(x[row + 1, col + 1]), float(y[row + 1, col + 1]), float(z[row + 1, col + 1])),
                    (float(x[row, col + 1]), float(y[row, col + 1]), float(z[row, col + 1])),
                ]
            )
    return faces


def add_tracker_surface_collection(
    ax: plt.Axes,
    faces: list[list[tuple[float, float, float]]],
    facecolors: list[str],
    lightsource: LightSource,
) -> None:
    surface_collection = Poly3DCollection(
        np.asarray(faces, dtype=float),
        facecolors=facecolors,
        linewidths=0,
        alpha=1.0,
        antialiaseds=False,
        shade=True,
        lightsource=lightsource,
        zsort="average",
    )
    ax.add_collection3d(surface_collection)


def projected_leftmost_barrel_point(
    ax: plt.Axes,
    layer: BarrelLayer,
    phi: np.ndarray,
) -> tuple[float, float, float]:
    x_samples = np.linspace(0.0, layer.z_half, 72)
    candidate_x = np.concatenate(
        [
            np.zeros_like(phi),
            np.full_like(phi, layer.z_half),
            x_samples,
            x_samples,
        ]
    )
    candidate_phi = np.concatenate(
        [
            phi,
            phi,
            np.full_like(x_samples, phi[0]),
            np.full_like(x_samples, phi[-1]),
        ]
    )
    candidate_y = layer.radius * np.cos(candidate_phi)
    candidate_z = layer.radius * np.sin(candidate_phi)
    projected_x, projected_y, _ = proj3d.proj_transform(candidate_x, candidate_y, candidate_z, ax.get_proj())
    display_xy = ax.transData.transform(np.column_stack([projected_x, projected_y]))
    index = int(np.argmin(display_xy[:, 0]))
    return float(candidate_x[index]), float(candidate_y[index]), float(candidate_z[index])


def project_point_to_axes(ax: plt.Axes, point: tuple[float, float, float]) -> tuple[float, float]:
    projected_x, projected_y, _ = proj3d.proj_transform(*point, ax.get_proj())
    display_xy = ax.transData.transform((projected_x, projected_y))
    axes_xy = ax.transAxes.inverted().transform(display_xy)
    return float(axes_xy[0]), float(axes_xy[1])


def draw_3d_radius_callouts(
    ax: plt.Axes,
    layers: list[BarrelLayer],
    unit: str,
    phi: np.ndarray,
) -> None:
    sorted_layers = sorted(layers, key=lambda item: item.radius, reverse=True)
    target_positions = [
        project_point_to_axes(ax, projected_leftmost_barrel_point(ax, layer, phi)) for layer in sorted_layers
    ]
    target_xs, target_ys = zip(*target_positions)
    label_x = max(0.02, min(target_xs) - 0.035)
    label_y_top = min(0.86, max(target_ys) + 0.08)
    label_y_bottom = max(0.08, min(target_ys) - 0.14)
    label_y_positions = np.linspace(label_y_top, label_y_bottom, len(sorted_layers))

    for layer, target, label_y in zip(sorted_layers, target_positions, label_y_positions):
        ax.annotate(
            fmt_radius(layer.radius, unit),
            xy=target,
            xycoords=ax.transAxes,
            xytext=(label_x, float(label_y)),
            textcoords=ax.transAxes,
            color="#111111",
            fontsize=7.5,
            ha="right",
            va="center",
            annotation_clip=False,
            arrowprops={
                "arrowstyle": "-",
                "color": "#202020",
                "lw": 0.75,
                "shrinkA": 0.0,
                "shrinkB": 0.0,
            },
            zorder=30,
        )


def backend_supports_interactive_window() -> bool:
    backend = plt.get_backend().lower()
    noninteractive_backends = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}
    return backend not in noninteractive_backends and "inline" not in backend


def format_angle(value: float) -> str:
    return f"{0.0 if abs(value) < 5e-7 else value:.6g}"


def positive_float(value: str) -> float:
    numeric_value = float(value)
    if numeric_value <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return numeric_value


def perspective_view_options(ax: plt.Axes, focal_length: float | None, distance: float) -> str:
    options = (
        f"--perspective-elev {format_angle(float(ax.elev))} "
        f"--perspective-azim {format_angle(float(ax.azim))} "
        f"--perspective-roll {format_angle(float(getattr(ax, 'roll', 0.0)))} "
        f"--perspective-distance {format_angle(distance)}"
    )
    if focal_length is not None:
        options += f" --perspective-focal-length {format_angle(focal_length)}"
    return options


def pause_for_interactive_view(fig: plt.Figure) -> None:
    try:
        fig.canvas.manager.set_window_title("Adjust MAIA tracker perspective")
    except AttributeError:
        pass
    print("Interactive perspective mode: rotate the 3D window, then press Enter here to save.")
    plt.show(block=False)
    plt.pause(0.1)
    try:
        input("Press Enter to save the perspective view...")
    except EOFError:
        print("No terminal input was available; saving the current perspective view.")


def draw_tracker_3d(
    model: GeometryModel,
    output_path: Path,
    unit: str,
    dpi: int,
    interactive: bool = False,
    perspective_elev: float = DEFAULT_PERSPECTIVE_ELEV,
    perspective_azim: float = DEFAULT_PERSPECTIVE_AZIM,
    perspective_roll: float = DEFAULT_PERSPECTIVE_ROLL,
    perspective_focal_length: float | None = DEFAULT_PERSPECTIVE_FOCAL_LENGTH,
    perspective_distance: float = DEFAULT_PERSPECTIVE_DISTANCE,
) -> tuple[Path, Path]:
    fig = plt.figure(figsize=(12.2, 8.4), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    tracker_layers = [layer for layer in model.barrel_layers if layer.subsystem in TRACKER_SUBSYSTEMS]
    tracker_rings = [ring for ring in model.endcap_rings if ring.subsystem in TRACKER_SUBSYSTEMS]
    z_max = max([layer.z_half for layer in tracker_layers] + [ring.z for ring in tracker_rings])
    r_max = max([layer.radius for layer in tracker_layers] + [ring.r_max for ring in tracker_rings])

    phi = np.linspace(math.radians(24), math.radians(152), 112)
    tracker_light = LightSource(azdeg=315, altdeg=75)
    surface_faces: list[list[tuple[float, float, float]]] = []
    surface_facecolors: list[str] = []

    # Put every tracker face in one collection so opaque layers depth-sort by
    # where they sit in space, not by which layer loop created them.
    for layer in sorted(tracker_layers, key=lambda item: item.radius, reverse=True):
        color = SUBSYSTEM_COLORS[layer.subsystem]
        x, y, z = surface_for_cylinder(layer.radius, layer.z_half, phi)
        faces = surface_grid_faces(x, y, z)
        surface_faces.extend(faces)
        surface_facecolors.extend([color] * len(faces))

    grouped_rings: dict[tuple[str, str], list[EndcapRing]] = {}
    for ring in tracker_rings:
        grouped_rings.setdefault((ring.subsystem, ring.layer_id), []).append(ring)

    for (_, _), rings in sorted(grouped_rings.items(), key=lambda item: tracker_layer_sort_key(item[1][0])):
        color = SUBSYSTEM_COLORS[rings[0].subsystem]
        for ring in rings:
            x, y, z = surface_for_annular_sector(ring.z, ring.r_min, ring.r_max, phi)
            faces = surface_grid_faces(x, y, z)
            surface_faces.extend(faces)
            surface_facecolors.extend([color] * len(faces))

    add_tracker_surface_collection(ax, surface_faces, surface_facecolors, tracker_light)

    # A slim beam line anchors the perspective without introducing non-tracker geometry.
    ax.plot([0.0, z_max * 1.03], [0.0, 0.0], [0.0, 0.0], color="#9a9a9a", lw=1.5, alpha=0.8)

    radius_layers = {
        round(layer.radius, 6): layer
        for layer in sorted(tracker_layers, key=lambda item: item.z_half, reverse=True)
    }

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
    ax.view_init(elev=perspective_elev, azim=perspective_azim, roll=perspective_roll)
    ax._dist = perspective_distance
    if perspective_focal_length is not None:
        ax.set_proj_type("persp", focal_length=perspective_focal_length)
    ax.set_axis_off()
    fig.canvas.draw()

    if interactive:
        if backend_supports_interactive_window():
            pause_for_interactive_view(fig)
            fig.canvas.draw()
        else:
            print(
                f"Requested --interactive-perspective, but Matplotlib backend "
                f"{plt.get_backend()!r} cannot open an interactive window. Saving the default view."
            )

    draw_3d_radius_callouts(ax, list(radius_layers.values()), unit, phi)
    saved_paths = save_figure_outputs(fig, output_path, dpi, bbox_inches="tight", pad_inches=0.16)
    if interactive:
        print("Perspective options for this saved view:")
        print(f"  {perspective_view_options(ax, perspective_focal_length, perspective_distance)}")
    plt.close(fig)
    return saved_paths


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
    parser.add_argument(
        "--full-rz-tracker",
        choices=("boxes", "layers"),
        default="layers",
        help="Draw tracker envelopes or explicit tracker layer lines in the full r-z detector view.",
    )
    parser.add_argument("--dpi", type=int, default=220, help="Output image resolution.")
    parser.add_argument(
        "--interactive-perspective",
        action="store_true",
        help="Open the tracker perspective plot for manual rotation before saving it.",
    )
    parser.add_argument(
        "--perspective-elev",
        type=float,
        default=DEFAULT_PERSPECTIVE_ELEV,
        help="Elevation angle for the tracker perspective view.",
    )
    parser.add_argument(
        "--perspective-azim",
        type=float,
        default=DEFAULT_PERSPECTIVE_AZIM,
        help="Azimuth angle for the tracker perspective view.",
    )
    parser.add_argument(
        "--perspective-roll",
        type=float,
        default=DEFAULT_PERSPECTIVE_ROLL,
        help="Roll angle for the tracker perspective view.",
    )
    parser.add_argument(
        "--perspective-focal-length",
        type=positive_float,
        default=DEFAULT_PERSPECTIVE_FOCAL_LENGTH,
        help="Perspective focal length. Smaller values give a stronger wide-angle/inside-camera effect.",
    )
    parser.add_argument(
        "--perspective-distance",
        type=positive_float,
        default=DEFAULT_PERSPECTIVE_DISTANCE,
        help="3D camera distance for the tracker perspective view. Smaller values feel closer/inside.",
    )
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

    written_paths = [
        *draw_rz(model, rz_path, args.unit, etas, args.dpi, args.full_rz_tracker),
        *draw_xy(model, xy_path, args.unit, args.xy_radii, args.dpi),
        *draw_tracker_layers(model, tracker_path, args.unit, etas, args.dpi),
        *draw_tracker_3d(
            model,
            tracker_3d_path,
            args.unit,
            args.dpi,
            args.interactive_perspective,
            args.perspective_elev,
            args.perspective_azim,
            args.perspective_roll,
            args.perspective_focal_length,
            args.perspective_distance,
        ),
    ]

    print(f"Read geometry from {args.geometry_dir}")
    for path in written_paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
