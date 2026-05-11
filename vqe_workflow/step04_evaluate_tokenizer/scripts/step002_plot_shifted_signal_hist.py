import argparse
import gzip
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASES = "ACGT"


BASE_ORDER = "ATGC"
PANEL_ORDER = "GACT"
COLORS = {
    "A": "#e76f51",
    "T": "#f4a261",
    "G": "#2a9d8f",
    "C": "#457b9d",
}


@dataclass
class Record:
    data: dict
    pattern: str
    spans: List[Tuple[int, int]]
    signal: List[float]


def open_text(path: str, mode: str):
    if path.endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def iter_records(path: str, max_reads: Optional[int] = None) -> Iterator[Record]:
    count = 0
    with open_text(path, "r") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            pattern = obj.get("pattern")
            spans = obj.get("base_sample_spans_rel")
            signal = obj.get("signal")
            if not isinstance(pattern, str) or not isinstance(spans, list) or not isinstance(signal, list):
                continue
            if len(pattern) != len(spans):
                continue
            parsed_spans: List[Tuple[int, int]] = []
            valid = True
            for pair in spans:
                if not isinstance(pair, list) or len(pair) != 2:
                    valid = False
                    break
                parsed_spans.append((int(pair[0]), int(pair[1])))
            if not valid:
                continue
            yield Record(
                data=obj,
                pattern=pattern,
                spans=parsed_spans,
                signal=[float(x) for x in signal],
            )
            count += 1
            if max_reads is not None and count >= max_reads:
                break


def build_prefix(signal: Sequence[float]) -> List[float]:
    prefix = [0.0]
    total = 0.0
    for value in signal:
        total += float(value)
        prefix.append(total)
    return prefix


def segment_mean(prefix: Sequence[float], start: int, end: int) -> float:
    return (prefix[end] - prefix[start]) / max(1, end - start)


def output_stem(path: str) -> str:
    input_name = os.path.basename(path)
    if input_name.endswith(".jsonl.gz"):
        return input_name[:-9]
    if input_name.endswith(".jsonl"):
        return input_name[:-6]
    return input_name


def collect_base_means(path: str) -> Dict[str, List[float]]:
    values: Dict[str, List[float]] = defaultdict(list)
    for record in iter_records(path):
        prefix = build_prefix(record.signal)
        for base, (start, end) in zip(record.pattern, record.spans):
            if base not in BASES or end <= start:
                continue
            values[base].append(segment_mean(prefix, start, end))
    return values


def plot_histogram(values: Dict[str, List[float]], output_path: str, dataset_label: str, bins: int, alpha: float) -> None:
    pooled: List[float] = []
    for base in BASE_ORDER:
        pooled.extend(values[base])
    if not pooled:
        raise RuntimeError("No base-level signal means collected.")

    xmin = min(pooled)
    xmax = max(pooled)
    if xmin == xmax:
        xmin -= 1.0
        xmax += 1.0
    pad = 0.03 * (xmax - xmin)
    xmin -= pad
    xmax += pad

    fig = plt.figure(figsize=(16, 13), dpi=160, constrained_layout=True)
    grid = fig.add_gridspec(4, 2, width_ratios=[1.55, 1.0], wspace=0.18, hspace=0.12)
    ax_overlay = fig.add_subplot(grid[:, 0])
    for base in BASE_ORDER:
        ax_overlay.hist(
            values[base],
            bins=bins,
            range=(xmin, xmax),
            alpha=alpha,
            color=COLORS[base],
            label=f"{base} (n={len(values[base])})",
            edgecolor="none",
        )
    ax_overlay.set_xlabel("Signal Value")
    ax_overlay.set_ylabel("Frequency")
    ax_overlay.set_xlim(xmin, xmax)
    ax_overlay.legend(frameon=False, loc="upper right")
    ax_overlay.grid(alpha=0.15, linewidth=0.6)
    ax_overlay.set_title(f"{dataset_label}  Overlay Histogram")

    for row, base in enumerate(PANEL_ORDER):
        ax = fig.add_subplot(grid[row, 1])
        base_values = values[base]
        ax.hist(
            base_values,
            bins=bins,
            range=(xmin, xmax),
            alpha=0.75,
            color=COLORS[base],
            edgecolor="none",
        )
        ax.set_xlim(xmin, xmax)
        ax.grid(alpha=0.15, linewidth=0.6)
        mean_text = f"{sum(base_values) / len(base_values):.3f}" if base_values else "NA"
        ax.set_title(f"{dataset_label}  {base} only  mean={mean_text}  n={len(base_values)}")
        ax.set_ylabel("Freq")
        if row == len(PANEL_ORDER) - 1:
            ax.set_xlabel("Signal Value")

    fig.suptitle(f"{dataset_label}  Base Signal Histogram")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot base-level signal histograms from a jsonl/jsonl.gz file.")
    parser.add_argument("--input", required=True, help="Input jsonl/jsonl.gz path.")
    parser.add_argument(
        "--output",
        help="Output png path. Defaults to <output-dir>/<input_stem>.hist.png.",
    )
    parser.add_argument(
        "--output-dir",
        default="/mnt/si003067jezr/default/dengyiting/workflow",
        help="Output directory used when --output is not provided.",
    )
    parser.add_argument("--bins", type=int, default=120, help="Histogram bin count.")
    parser.add_argument("--alpha", type=float, default=0.35, help="Overlay histogram transparency.")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise SystemExit(f"Input file not found: {args.input}")

    if args.output:
        output_path = args.output
        output_dir = os.path.dirname(os.path.abspath(output_path)) or "."
    else:
        output_dir = args.output_dir
        output_path = os.path.join(output_dir, f"{output_stem(args.input)}.hist.png")

    os.makedirs(output_dir, exist_ok=True)
    values = collect_base_means(args.input)
    plot_histogram(values, output_path, output_stem(args.input), args.bins, args.alpha)

    print(f"input={args.input}")
    print(f"wrote_plot={output_path}")


if __name__ == "__main__":
    main()
