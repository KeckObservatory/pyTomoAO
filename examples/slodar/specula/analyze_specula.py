"""Stage 3b: SLODAR profile from SPECULA LTAO open-loop slopes.

Usage:
    python analyze_specula.py OUTDIR SLOPES.npz [SLOPES2.npz ...]

Several chunk files (independent SPECULA runs with different seeds, see run_chunks.sh) are
combined: each chunk's temporal mean is removed and the frames are concatenated.

Fits the 7-layer SPECULA input with pyTomoAO.slodar under several analysis choices and
writes figures plus a JSON summary. The slopes are those exported by run_specula.py.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import slodar_sim as S

from pyTomoAO.slodar import slodarProfiler


def tvd(a, b):
    return 0.5 * np.abs(np.asarray(a) - np.asarray(b)).sum()


def main(outdir, *npz):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    z = dict(np.load(npz[0]))
    chunks = [np.load(f)["slopes"] for f in npz]
    slopes = np.concatenate([c - c.mean(0) for c in chunks])
    alts, fr = z["altitude"], z["fraction"]
    r0_true = 0.9759 * 0.5e-6 / (float(z["seeing"]) / 206264.806)
    L0_true = float(z["L0"])
    dt = float(z["time_step"])
    T = slopes.shape[0]
    print(f"{T} frames, dt = {dt} s, r0 true = {r0_true:.4f} m, L0 = {L0_true} m")

    def profiler(**kw):
        args = {
            "validLLMap": z["validLLMap"],
            "D": float(z["D"]),
            "gsDirections": z["gsDirections"],
            "gsHeight": float(z["gsHeight"]),
            "L0": L0_true,
            "subapOrder": "C",
        }
        args.update(kw)
        return slodarProfiler(**args)

    prof = profiler()
    grid = prof.default_altitudes(hMax=20e3)
    truth = S.bin_profile(alts, fr, grid)
    summary = {
        "inputs": [str(f) for f in npz],
        "nFrames": int(T),
        "time_step_s": dt,
        "r0_true": r0_true,
        "L0_true": L0_true,
        "altitude_grid_m": grid.tolist(),
        "truth_binned": truth.tolist(),
        "runs": {},
    }

    def record(name, res, extra=None):
        t = S.bin_profile(alts, fr, res.altitude)
        d = {
            "r0": res.r0,
            "r0_err": res.r0 / r0_true - 1,
            "tvd": tvd(res.fractionalCn2, t),
            "L0": res.L0,
            "residual": res.residualRms,
            "altitude_m": res.altitude.tolist(),
            "fraction": res.fractionalCn2.tolist(),
            "truth_binned": t.tolist(),
            "ground_fraction": float(res.fractionalCn2[res.altitude < 250].sum()),
        }
        if extra:
            d.update(extra)
        summary["runs"][name] = d
        print(
            f"  {name:34s} r0 {res.r0:.4f} ({d['r0_err']:+.1%})  TVD {d['tvd']:.3f}  "
            f"L0 {res.L0:.0f}  resid {res.residualRms:.3f}"
        )
        return d

    # Baseline: tip-tilt removed (LGS), all 15 pairs, xx+yy, auto-covariance with noise term.
    prof.compute_covariance_maps(slopes, removeTT=True)
    base = prof.fit_profile(altitudes=grid)
    record("baseline (TT removed, L0=25)", base)
    print(base)

    # Covariance-map figure for one long baseline, data vs model.
    model = {}
    for i, h in enumerate(base.altitude):
        for key, m in prof.response_maps(h, base.L0).items():
            model[key] = model.get(key, 0) + base.r0m53[i] * m
    fig = prof.plot_covariance_maps((0, 3), components=("xx", "yy"), model=model)
    fig.savefig(out / "fig6_specula_maps_pair03.png", dpi=110)
    plt.close(fig)

    # 1-D cut along the (0, 3) baseline (along x): data vs model per layer.
    N = prof.nLenslet
    fig, ax = plt.subplots(figsize=(7, 4))
    k = np.arange(-(N - 1), N)
    dat = prof.maps[(0, 3)][0][N - 1] + prof.maps[(0, 3)][3][N - 1]
    mod = model[(0, 3)][0][N - 1] + model[(0, 3)][3][N - 1]
    ax.plot(k, dat, "k.-", label="SPECULA (xx+yy)")
    ax.plot(k, mod, "C0-", label="SLODAR fit")
    for i, h in enumerate(base.altitude):
        if base.r0m53[i] > 0:
            R = prof.response_maps(h, base.L0)[(0, 3)]
            ax.plot(k, base.r0m53[i] * (R[0][N - 1] + R[3][N - 1]), lw=0.7, alpha=0.7)
    ax.set_xlabel(r"offset $\delta_x$ along the 120$''$ baseline [subap]")
    ax.set_ylabel(r"covariance [rad$^2$]")
    ax.legend(fontsize=8)
    ax.set_title("Cross-covariance cut, WFS pair (0, 3); thin lines = layer contributions")
    fig.tight_layout()
    fig.savefig(out / "fig7_specula_cut_pair03.png", dpi=110)
    plt.close(fig)

    # Variants.
    prof_tt = profiler()
    prof_tt.compute_covariance_maps(slopes, removeTT=False)
    record("TT kept", prof_tt.fit_profile(altitudes=grid))
    resL0 = prof.fit_profile(altitudes=grid, fitL0=True, L0Grid=np.geomspace(5, 200, 14))
    record(
        "L0 fitted (grid search)",
        resL0,
        {"L0Grid": resL0.L0Grid.tolist(), "L0Cost": resL0.L0Cost.tolist()},
    )
    record("L0 fixed 100 m (mismatch)", prof.fit_profile(altitudes=grid, L0=100.0))
    record("cross-covariance only", prof.fit_profile(altitudes=grid, useAuto=False))
    record("xx,xy,yx,yy", prof.fit_profile(altitudes=grid, components=("xx", "xy", "yx", "yy")))
    record("long./transverse", prof.fit_profile(altitudes=grid, components=("ll", "tt")))
    nb = prof.pairs
    longest = [p for p in nb if abs(prof.separation(*p) - prof.separation(0, 3)) < 1e-9]
    record('longest baselines only (120")', prof.fit_profile(altitudes=grid, pairs=longest))
    shortest = [p for p in nb if abs(prof.separation(*p) - prof.separation(0, 1)) < 1e-9]
    record('shortest baselines only (60")', prof.fit_profile(altitudes=grid, pairs=shortest))

    # Ignoring the cone effect: analyse as if NGS.
    prof_ngs = profiler(gsHeight=np.inf)
    prof_ngs.compute_covariance_maps(slopes, removeTT=True)
    grid_ngs = prof_ngs.default_altitudes(hMax=20e3)
    record("cone effect ignored (NGS model)", prof_ngs.fit_profile(altitudes=grid_ngs))

    # Grid placed on the true layers (best case: no binning error).
    record("grid = true layer altitudes", prof.fit_profile(altitudes=alts))

    # Frame-count convergence on contiguous sub-sequences.
    conv = {"nFrames": [], "r0_err": [], "tvd": []}
    for n in sorted({100, 200, 400, 800, 1600, T}):
        if n > T:
            continue
        errs, tvs = [], []
        for start in range(0, T - n + 1, max(n, T // 4)):
            p2 = profiler()
            p2.compute_covariance_maps(slopes[start : start + n], removeTT=True)
            r = p2.fit_profile(altitudes=grid)
            errs.append(r.r0 / r0_true - 1)
            tvs.append(tvd(r.fractionalCn2, truth))
        conv["nFrames"].append(n)
        conv["r0_err"].append([float(e) for e in errs])
        conv["tvd"].append([float(t) for t in tvs])
        print(
            f"  frames {n:5d}: r0 err {np.mean(errs):+.2%} "
            f"(spread {np.std(errs):.2%}), TVD {np.mean(tvs):.3f}"
        )
    summary["convergence"] = conv

    # Profile figure.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    ax = axes[0]
    ax.barh(grid / 1e3, truth, height=0.3, color="0.75", label="input, binned to grid")
    ax.plot(base.fractionalCn2, base.altitude / 1e3, "o-", label="SLODAR baseline")
    ax.plot(
        resL0.fractionalCn2,
        resL0.altitude / 1e3,
        "s--",
        ms=4,
        label=f"L0 fitted ({resL0.L0:.0f} m)",
    )
    ax.plot(fr, alts / 1e3, "x", color="C3", ms=9, mew=2, label="SPECULA input layers")
    ax.set_xlabel("fractional $C_n^2$")
    ax.set_ylabel("altitude [km]")
    ax.set_title(
        f'SPECULA LTAO, 6 LGS @ 60", {T} frames\n'
        f"r0 {base.r0:.3f} m (input {r0_true:.3f} m), "
        f"TVD {summary['runs']['baseline (TT removed, L0=25)']['tvd']:.2f}",
        fontsize=9,
    )
    ax.legend(fontsize=7)

    ax = axes[1]
    r_true = summary["runs"]["grid = true layer altitudes"]
    ax.barh(alts / 1e3 - 0.15, fr, height=0.3, color="0.6", label="input")
    ax.barh(
        alts / 1e3 + 0.15,
        r_true["fraction"],
        height=0.3,
        color="C1",
        label="SLODAR, grid = input altitudes",
    )
    ax.set_xlabel("fractional $C_n^2$")
    ax.set_ylabel("altitude [km]")
    ax.set_title(
        f"fit on the 7 input altitudes: r0 {r_true['r0']:.3f} m, TVD {r_true['tvd']:.2f}",
        fontsize=9,
    )
    ax.legend(fontsize=7)

    ax = axes[2]
    names = list(summary["runs"])
    y = np.arange(len(names))
    ax.barh(
        y - 0.2,
        [summary["runs"][n]["r0_err"] * 100 for n in names],
        height=0.4,
        label="r0 error [%]",
    )
    ax.barh(y + 0.2, [summary["runs"][n]["tvd"] * 100 for n in names], height=0.4, label="TVD x100")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.axvline(0, color="0.5", lw=0.8)
    ax.legend(fontsize=7)
    ax.set_title("Analysis variants", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig5_specula_profile.png", dpi=110)
    plt.close(fig)

    with open(out / "stage3_results.json", "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print("done ->", out)


if __name__ == "__main__":
    main(sys.argv[1], *sys.argv[2:])
