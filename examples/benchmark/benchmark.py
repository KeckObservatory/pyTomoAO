"""Benchmark ``build_reconstructor`` against the installed package.

This replaces a set of scripts that carried their own forked copies of the covariance
kernels. Those drifted from the library and ended up reporting numbers for code that was no
longer shipped, so everything here goes through the public API.

Usage
-----
Run every bundled configuration on whichever backends are available::

    python examples/benchmark/benchmark.py

Record the current numbers as the reference to compare future runs against::

    python examples/benchmark/benchmark.py --save-baseline

Compare against the recorded reference and exit non-zero on a regression::

    python examples/benchmark/benchmark.py --check-baseline --tolerance 0.25

Timings depend on the machine, so a baseline is only meaningful next to results from the
same one; `baseline.json` records the CPU and platform it was taken on.
"""

import argparse
import json
import logging
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from pyTomoAO import backend, example_config, list_example_configs
from pyTomoAO.reconstructor import tomographicReconstructor

BASELINE = Path(__file__).with_name("baseline.json")


def time_build(config, force_cpu, repeats):
    """Time ``build_reconstructor`` and return the timings and the resulting matrix.

    The first build is discarded: Numba compiles the Bessel kernel on the CPU path and CuPy
    compiles its ElementwiseKernels on the GPU path, so including it would measure the
    compiler.
    """
    tomographicReconstructor(config, force_cpu=force_cpu).build_reconstructor()

    timings = []
    reconstructor = None
    for _ in range(repeats):
        rec = tomographicReconstructor(config, force_cpu=force_cpu)
        start = time.perf_counter()
        reconstructor = rec.build_reconstructor()
        timings.append(time.perf_counter() - start)
    return timings, reconstructor


def run(config_names, repeats):
    """Benchmark each named configuration on every available backend."""
    backends = [("cpu", True)]
    if backend.cuda_available():
        backends.append(("gpu", False))

    results = {}
    for name in config_names:
        config = example_config(name)
        entry = {}
        matrices = {}
        for label, force_cpu in backends:
            timings, matrix = time_build(config, force_cpu, repeats)
            entry[label] = {
                "min": min(timings),
                "median": statistics.median(timings),
                "shape": list(matrix.shape),
                "dtype": str(matrix.dtype),
            }
            matrices[label] = matrix

        if "cpu" in matrices and "gpu" in matrices:
            cpu, gpu = (matrices["cpu"].astype(np.float64), matrices["gpu"].astype(np.float64))
            scale = np.abs(cpu).max()
            entry["backend_agreement"] = float(np.abs(cpu - gpu).max() / scale) if scale else 0.0
        results[name] = entry
    return results


def report(results):
    """Print one row per configuration and backend."""
    header = f"{'configuration':<22} {'backend':<8} {'min (s)':>10} {'median (s)':>12}"
    print(f"\n{header} {'shape':>14}")
    print("-" * 70)
    for name, entry in results.items():
        for label in ("cpu", "gpu"):
            if label not in entry:
                continue
            row = entry[label]
            shape = "x".join(str(n) for n in row["shape"])
            print(f"{name:<22} {label:<8} {row['min']:>10.3f} {row['median']:>12.3f} {shape:>14}")
        if "backend_agreement" in entry:
            print(f"{'':<22} {'agree':<8} {entry['backend_agreement']:>10.2e} (relative)")


#: A run must be slower than the baseline by both the relative tolerance *and* this many
#: seconds to count as a regression. Without the absolute floor, the sub-100 ms
#: configurations trip on ordinary timing jitter -- a 0.09 s build drifting to 0.12 s is
#: +30%, which says nothing about the code.
MIN_REGRESSION_SECONDS = 0.25


def compare(results, tolerance):
    """Compare against the committed baseline. Returns True if nothing regressed."""
    if not BASELINE.exists():
        print(f"\nNo baseline at {BASELINE}. Record one with --save-baseline.")
        return False

    reference = json.loads(BASELINE.read_text())
    if reference.get("platform", {}).get("processor") != platform.processor():
        print(
            "\nWarning: the baseline was recorded on a different processor "
            f"({reference.get('platform', {}).get('processor')!r}); timings are not comparable."
        )

    ok = True
    print(f"\n{'configuration':<22} {'backend':<8} {'baseline':>10} {'now':>10} {'change':>10}")
    print("-" * 64)
    for name, entry in results.items():
        for label in ("cpu", "gpu"):
            if label not in entry:
                continue
            was = reference.get("results", {}).get(name, {}).get(label, {}).get("min")
            if was is None:
                print(f"{name:<22} {label:<8} {'-':>10} {entry[label]['min']:>10.3f}   (new)")
                continue
            now = entry[label]["min"]
            change = now / was - 1.0
            flag = ""
            if change > tolerance and now - was > MIN_REGRESSION_SECONDS:
                flag = "  REGRESSION"
                ok = False
            elif change > tolerance:
                flag = "  (noise)"
            print(f"{name:<22} {label:<8} {was:>10.3f} {now:>10.3f} {change:>+9.1%}{flag}")
    return ok


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--configs",
        nargs="+",
        default=list_example_configs(),
        help="bundled configurations to benchmark (default: all)",
    )
    parser.add_argument("--repeats", type=int, default=3, help="timed runs per backend")
    parser.add_argument("--save-baseline", action="store_true", help="record these numbers")
    parser.add_argument("--check-baseline", action="store_true", help="compare and exit non-zero")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.25,
        help="fractional slowdown tolerated by --check-baseline (default 0.25)",
    )
    args = parser.parse_args(argv)

    # The reconstructor logs at INFO on every build; that would drown the table.
    logging.getLogger("pyTomoAO").setLevel(logging.WARNING)

    print(f"backend available: cpu{', gpu' if backend.cuda_available() else ''}")
    results = run(args.configs, args.repeats)
    report(results)

    if args.save_baseline:
        BASELINE.write_text(
            json.dumps(
                {
                    "platform": {
                        "processor": platform.processor(),
                        "machine": platform.machine(),
                        "python": platform.python_version(),
                    },
                    "repeats": args.repeats,
                    "results": results,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nBaseline written to {BASELINE}")

    if args.check_baseline and not compare(results, args.tolerance):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
