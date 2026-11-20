"""Synthetic lifestyle illustration using hand-defined values; no measured radar results."""
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POINTNET_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


@dataclass
class Zone:
    name: str
    x: float
    y: float
    w: float
    h: float
    fill: str


ZONES = [
    Zone("Bed area", 760, 110, 270, 150, "#f0f0f0"),
    Zone("Desk area", 760, 330, 270, 130, "#e8e8e8"),
    Zone("Chair area", 150, 390, 230, 120, "#eeeeee"),
    Zone("Open area", 420, 210, 280, 250, "#fafafa"),
    Zone("Transition", 420, 500, 610, 90, "#f7f7f7"),
]


NORMAL_DWELL_HOURS = {
    "Bed area": [8.1, 7.8, 8.4, 7.9, 8.2, 8.0, 8.3],
    "Desk area": [1.1, 1.3, 1.0, 1.2, 1.0, 1.4, 1.2],
    "Chair area": [2.0, 2.3, 2.1, 1.9, 2.2, 2.4, 2.1],
    "Open area": [3.2, 3.0, 3.4, 3.1, 3.3, 3.2, 3.0],
    "Transition": [0.9, 1.1, 1.0, 0.8, 1.0, 0.9, 1.1],
}


ANOMALY_DAY_HOURS = {
    "Bed area": 11.4,
    "Desk area": 0.3,
    "Chair area": 0.5,
    "Open area": 0.7,
    "Transition": 3.1,
}


NORMAL_PATH = [
    (890, 180),
    (620, 270),
    (500, 340),
    (270, 445),
    (520, 535),
    (880, 390),
    (900, 180),
]


ANOMALY_PATH = [
    (890, 180),
    (830, 520),
    (620, 545),
    (520, 535),
    (590, 535),
    (520, 535),
    (590, 535),
    (520, 535),
    (500, 540),
]


def svg_start(width: int, height: int) -> List[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        "<style>",
        ".title{font:700 28px 'Times New Roman',serif;fill:#111}",
        ".label{font:20px 'Times New Roman',serif;fill:#111}",
        ".small{font:16px 'Times New Roman',serif;fill:#222}",
        ".tiny{font:14px 'Times New Roman',serif;fill:#333}",
        ".room{fill:#fff;stroke:#111;stroke-width:2.4}",
        ".zone{stroke:#222;stroke-width:2;rx:10;ry:10}",
        ".axis{stroke:#222;stroke-width:1.8}",
        ".grid{stroke:#d0d0d0;stroke-width:1}",
        ".bar{fill:#d9e6f2;stroke:#222;stroke-width:1.6}",
        ".alarm{fill:#f5d6d0;stroke:#111;stroke-width:2}",
        ".normal{fill:none;stroke:#222;stroke-width:3}",
        ".abnormal{fill:none;stroke:#b22222;stroke-width:3;stroke-dasharray:9 6}",
        "</style>",
        '<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#222"/></marker>',
        '<marker id="arrow-red" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#b22222"/></marker>',
        "</defs>",
        '<rect width="100%" height="100%" fill="white"/>',
    ]


def svg_end(parts: List[str]) -> str:
    parts.append('<text x="15" y="20" font-size="12" fill="#9b2323">SYNTHETIC ILLUSTRATION - hand-defined values, not measured radar results</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def write_svg(name: str, parts: List[str]) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg_end(parts))
    return path


def draw_room(parts: List[str], x: int = 110, y: int = 95):
    parts.append(f'<rect x="{x}" y="{y}" width="1030" height="540" rx="12" ry="12" class="room"/>')
    for zone in ZONES:
        parts.append(
            f'<rect x="{zone.x}" y="{zone.y}" width="{zone.w}" height="{zone.h}" class="zone" fill="{zone.fill}"/>'
        )
        parts.append(
            f'<text x="{zone.x + zone.w / 2}" y="{zone.y + zone.h / 2 - 8}" text-anchor="middle" class="label">{zone.name}</text>'
        )
    parts.append('<circle cx="430" cy="185" r="14" fill="#222"/>')
    parts.append('<text x="430" y="165" text-anchor="middle" class="small">Radar</text>')


def polyline(points: List[Tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def draw_path(parts: List[str], path: List[Tuple[float, float]], cls: str, arrow_id: str):
    parts.append(f'<polyline points="{polyline(path)}" class="{cls}" marker-end="url(#{arrow_id})"/>')
    for idx, (x, y) in enumerate(path):
        parts.append(f'<circle cx="{x}" cy="{y}" r="5" fill="#111"/>')
        if idx in (0, len(path) - 1):
            label = "start" if idx == 0 else "end"
            parts.append(f'<text x="{x + 8}" y="{y - 8}" class="tiny">{label}</text>')


def generate_zone_layout():
    parts = svg_start(1250, 760)
    parts.append('<text x="625" y="46" text-anchor="middle" class="title">Functional Zone Layout for Localisation-Based Logging</text>')
    draw_room(parts)
    parts.append('<text x="120" y="705" class="small">The monitored space is divided into functional zones. Estimated target positions are assigned to these zones for dwell-time accumulation.</text>')
    return write_svg("zone_layout.svg", parts)


def generate_dwell_time_profile():
    mean_hours = {k: sum(v) / len(v) for k, v in NORMAL_DWELL_HOURS.items()}
    parts = svg_start(1150, 720)
    parts.append('<text x="575" y="46" text-anchor="middle" class="title">Daily Dwell-Time Profile Learned from Routine Days</text>')
    x0, y0 = 150, 610
    chart_w, chart_h = 830, 430
    max_h = 12
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0 + chart_w}" y2="{y0}" class="axis"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 - chart_h}" class="axis"/>')
    for h in range(0, max_h + 1, 2):
        y = y0 - chart_h * h / max_h
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + chart_w}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{x0 - 18}" y="{y + 5:.1f}" text-anchor="end" class="tiny">{h}</text>')
    parts.append(f'<text x="75" y="{y0 - chart_h / 2}" text-anchor="middle" class="small" transform="rotate(-90 75 {y0 - chart_h / 2})">Hours per day</text>')
    zones = list(mean_hours.keys())
    bar_w = 95
    gap = 50
    for idx, zone in enumerate(zones):
        x = x0 + 38 + idx * (bar_w + gap)
        h = mean_hours[zone]
        bar_h = chart_h * h / max_h
        y = y0 - bar_h
        parts.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" class="bar"/>')
        parts.append(f'<text x="{x + bar_w / 2}" y="{y - 8:.1f}" text-anchor="middle" class="small">{h:.1f} h</text>')
        parts.append(f'<text x="{x + bar_w / 2}" y="{y0 + 28}" text-anchor="middle" class="tiny">{zone}</text>')
    parts.append('<text x="575" y="675" text-anchor="middle" class="small">The profile represents the baseline routine learned from normal daily occupancy patterns.</text>')
    return write_svg("dwell_time_profile.svg", parts)


def generate_habit_learning_process():
    parts = svg_start(1280, 720)
    parts.append('<text x="640" y="46" text-anchor="middle" class="title">Personalised Habit Learning from Multi-Day Zone Occupancy</text>')
    zones = list(NORMAL_DWELL_HOURS.keys())
    days = [f"Day {i}" for i in range(1, 8)] + ["New day"]
    x0, y0 = 170, 120
    cell_w, cell_h = 110, 68
    max_h = 12
    for i, day in enumerate(days):
        parts.append(f'<text x="{x0 + (i + 1) * cell_w + cell_w / 2}" y="{y0 - 18}" text-anchor="middle" class="small">{day}</text>')
    for r, zone in enumerate(zones):
        parts.append(f'<text x="{x0 - 12}" y="{y0 + r * cell_h + cell_h / 2 + 6}" text-anchor="end" class="small">{zone}</text>')
        for c in range(8):
            if c < 7:
                value = NORMAL_DWELL_HOURS[zone][c]
            else:
                value = ANOMALY_DAY_HOURS[zone]
            intensity = min(max(value / max_h, 0), 1)
            shade = int(245 - 120 * intensity)
            color = f"rgb({shade},{shade},{shade})" if c < 7 else f"rgb(245,{max(180, shade)}, {max(170, shade)})"
            x = x0 + (c + 1) * cell_w
            y = y0 + r * cell_h
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{color}" stroke="#222" stroke-width="1.3"/>')
            parts.append(f'<text x="{x + cell_w / 2}" y="{y + cell_h / 2 + 5}" text-anchor="middle" class="tiny">{value:.1f} h</text>')
    parts.append('<rect x="1050" y="520" width="170" height="70" rx="10" ry="10" class="alarm"/>')
    parts.append('<text x="1135" y="548" text-anchor="middle" class="label">Warning</text>')
    parts.append('<text x="1135" y="574" text-anchor="middle" class="small">routine deviation</text>')
    parts.append('<text x="640" y="655" text-anchor="middle" class="small">Normal days establish a personal baseline. The new day is compared against this learned baseline to detect unusual dwell-time patterns.</text>')
    return write_svg("habit_learning_heatmap.svg", parts)


def generate_trajectory_comparison():
    parts = svg_start(1300, 780)
    parts.append('<text x="650" y="46" text-anchor="middle" class="title">Normal and Abnormal Trajectory Comparison</text>')
    draw_room(parts)
    draw_path(parts, NORMAL_PATH, "normal", "arrow")
    draw_path(parts, ANOMALY_PATH, "abnormal", "arrow-red")
    parts.append('<rect x="120" y="655" width="470" height="70" rx="10" ry="10" fill="#fff" stroke="#222" stroke-width="2"/>')
    parts.append('<line x1="150" y1="680" x2="210" y2="680" class="normal" marker-end="url(#arrow)"/>')
    parts.append('<text x="225" y="686" class="small">normal routine trajectory</text>')
    parts.append('<line x1="150" y1="708" x2="210" y2="708" class="abnormal" marker-end="url(#arrow-red)"/>')
    parts.append('<text x="225" y="714" class="small">abnormal repeated transition / reduced activity</text>')
    parts.append('<text x="700" y="705" class="small">The abnormal path shows prolonged transition-zone staying and reduced visits to normal daily zones.</text>')
    return write_svg("trajectory_anomaly_comparison.svg", parts)


def compute_baseline_stats():
    stats = {}
    for zone, values in NORMAL_DWELL_HOURS.items():
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)
        stats[zone] = {"mean_hours": mean, "std_hours": math.sqrt(var)}
    return stats


def compute_anomaly_scores(stats: Dict[str, Dict[str, float]]):
    scores = {}
    for zone, today in ANOMALY_DAY_HOURS.items():
        std = max(stats[zone]["std_hours"], 0.15)
        z = (today - stats[zone]["mean_hours"]) / std
        scores[zone] = {"today_hours": today, "z_score": z, "abnormal": abs(z) > 2.0}
    return scores


def write_summary(paths: Dict[str, str]):
    stats = compute_baseline_stats()
    scores = compute_anomaly_scores(stats)
    summary = {
        "purpose": "Demonstration of localisation-based daily habit learning and abnormal trajectory detection.",
        "outputs": paths,
        "baseline_stats": stats,
        "new_day": ANOMALY_DAY_HOURS,
        "anomaly_scores": scores,
        "rule": "A zone is marked abnormal when the absolute z-score of dwell time is greater than 2.0.",
    }
    path = os.path.join(OUTPUT_DIR, "lifestyle_demo_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return path


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outputs = {
        "zone_layout": generate_zone_layout(),
        "dwell_time_profile": generate_dwell_time_profile(),
        "habit_learning_heatmap": generate_habit_learning_process(),
        "trajectory_anomaly_comparison": generate_trajectory_comparison(),
    }
    summary_path = write_summary(outputs)
    print("Generated lifestyle demonstration figures:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")
    print(f"- summary: {summary_path}")


if __name__ == "__main__":
    main()
