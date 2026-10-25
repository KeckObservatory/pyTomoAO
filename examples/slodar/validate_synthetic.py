"""Stage 2: validate pyTomoAO.slodar on synthetic slopes.

Two independent slope generators are used:

- ``gauss`` -- Gaussian frames drawn from the theoretical slope covariance of
  :meth:`slodarProfiler.slope_covariance` (tests the estimator alone; model-consistent);
- ``fft``   -- von Karman FFT phase screens + exact G-tilt Shack-Hartmann slopes from
  ``slodar_sim.py`` (shares no code with the response model).

Run:  python validate_synthetic.py [outdir]
"""

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import slodar_sim as S

from pyTomoAO.slodar import slodarProfiler

_args = [a for a in sys.argv[1:] if not a.startswith("--")]
OUT = Path(_args[0] if _args else "slodar_outputs/stage2")
OUT.mkdir(parents=True, exist_ok=True)

D = 10.0
VALID = S.kapa_valid_map()  # 20x20, 304 valid subapertures, d = 0.5 m
L0_TRUE = 25.0
H_LGS = 90e3
MKR = S.MKR_MEDIAN
WIDE = S.ring_asterism(6, 60.0)  # 6 LGS on a 60" radius ring (baselines 60", 104", 120")
KAPA = S.ring_asterism(4, 7.6)  # KAPA asterism, 4 LGS on a 7.6" radius ring
results = {}


def done(section):
    """Restart support: a section whose JSON checkpoint exists is loaded, not recomputed."""
    f = OUT / f"stage2_{section}.json"
    if f.exists() and "--force" not in sys.argv:
        with open(f) as fh:
            results[section] = json.load(fh)
        print(f"== {section}: loaded checkpoint {f.name}")
        return True
    return False


def save(section):
    with open(OUT / f"stage2_{section}.json", "w") as fh:
        json.dump(results[section], fh, indent=1, default=float)


def tvd(a, b):
    """Total-variation distance between two fractional profiles (0 = identical, 1 = disjoint)."""
    return 0.5 * np.abs(np.asarray(a) - np.asarray(b)).sum()


def run_case(prof, alts, fracs, r0, nFrames, gen, seed, noiseRms=0.0, L0data=L0_TRUE, fit=None):
    fit = fit or {}
    w = prof.profile_to_r0m53(r0, fracs)
    t0 = time.time()
    if gen == "gauss":
        s = prof.draw_slopes(alts, w, nFrames, noiseRms=noiseRms, L0=L0data, rng=seed)
    else:
        s = S.simulate_slopes(
            prof.validLLMap,
            prof.D,
            prof.gsDirections,
            alts,
            fracs,
            r0,
            L0data,
            nFrames,
            gsHeight=prof.gsHeight,
            noiseRms=noiseRms,
            seed=seed,
        )
    tsim = time.time() - t0
    removeTT = fit.pop("removeTT", False)
    prof.compute_covariance_maps(s, removeTT=removeTT)
    alt_grid = fit.pop("altitudes", None)
    if alt_grid is None:
        alt_grid = prof.default_altitudes(hMax=20e3)
    res = prof.fit_profile(altitudes=alt_grid, **fit)
    truth = S.bin_profile(alts, fracs, res.altitude)
    return res, truth, tsim


def profile_axes(ax, res, truth, alts, fracs, title):
    ax.barh(res.altitude / 1e3, truth, height=0.25, color="0.75", label="input, binned to grid")
    ax.plot(res.fractionalCn2, res.altitude / 1e3, "o-", color="C0", label="SLODAR")
    ax.plot(fracs, np.asarray(alts) / 1e3, "x", color="C3", ms=9, mew=2, label="input layers")
    ax.set_xlabel("fractional $C_n^2$")
    ax.set_ylabel("altitude [km]")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)


# ---------------------------------------------------------------------------
# 1. One- and two-layer tests, NGS and LGS geometry, both generators
def section_simple():
    print("== 1. simple profiles")
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    cases = [
        ("1 layer @ 5 km", [5000.0], [1.0]),
        ("2 layers 0 + 8 km", [0.0, 8000.0], [0.65, 0.35]),
    ]
    col = 0
    results["simple"] = []
    for gsH, gname in [(np.inf, "NGS"), (H_LGS, "LGS 90 km")]:
        prof = slodarProfiler(VALID, D, WIDE, gsHeight=gsH, L0=L0_TRUE)
        for label, alts0, fracs in cases:
            # Put the single layer exactly on a bin so peak placement can be judged.
            alts = alts0
            if len(alts0) == 1:
                grid = prof.default_altitudes(hMax=20e3)
                alts = [float(grid[np.argmin(np.abs(grid - alts0[0]))])]
            for r, gen in enumerate(["gauss", "fft"]):
                res, truth, _ = run_case(prof, alts, fracs, 0.16, 4000, gen, seed=10 + col)
                err = tvd(res.fractionalCn2, truth)
                results["simple"].append(
                    {
                        "geometry": gname,
                        "case": label,
                        "generator": gen,
                        "layers_m": [round(a, 1) for a in alts],
                        "r0_true": 0.16,
                        "r0_est": res.r0,
                        "tvd": err,
                        "residual": res.residualRms,
                    }
                )
                print(f"  {gname:9s} {label:18s} {gen:5s} r0={res.r0:.4f} TVD={err:.3f}")
                profile_axes(
                    axes[r, col],
                    res,
                    truth,
                    alts,
                    fracs,
                    f"{gname}, {label}, {gen}\nr0 {res.r0:.3f} (true 0.160), TVD {err:.2f}",
                )
            col += 1
    fig.tight_layout()
    fig.savefig(OUT / "fig1_simple_profiles.png", dpi=110)
    plt.close(fig)

    save("simple")


if not done("simple"):
    section_simple()

# ---------------------------------------------------------------------------
# 2. Seven-layer Maunakea median profile (KAON 503), LGS at 90 km
alts7, fr7, r07 = MKR["altitude"], MKR["fraction"], MKR["r0"]


def section_mkr7():
    print("== 2. 7-layer MKR median")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    results["mkr7"] = []
    for k, (ast, aname) in enumerate([(WIDE, '6 LGS @ 60"'), (KAPA, 'KAPA 4 LGS @ 7.6"')]):
        prof = slodarProfiler(VALID, D, ast, gsHeight=H_LGS, L0=L0_TRUE)
        gens = ["gauss", "fft"] if k == 0 else ["fft"]
        for gen in gens:
            res, truth, tsim = run_case(
                prof, alts7, fr7, r07, 8000, gen, seed=7, fit={"removeTT": True}
            )
            err = tvd(res.fractionalCn2, truth)
            print(f"  {aname:20s} {gen:5s} r0={res.r0:.4f} TVD={err:.3f} ({tsim:.0f} s sim)")
            print(res)
            results["mkr7"].append(
                {
                    "asterism": aname,
                    "generator": gen,
                    "r0_true": r07,
                    "r0_est": res.r0,
                    "tvd": err,
                    "altitude_m": res.altitude.tolist(),
                    "fraction_est": res.fractionalCn2.tolist(),
                    "fraction_true_binned": truth.tolist(),
                }
            )
            ax = axes[0 if gen == "gauss" else (1 if k == 0 else 2)]
            profile_axes(
                ax,
                res,
                truth,
                alts7,
                fr7,
                f"{aname}, {gen}, 8000 frames, TT removed\n"
                f"r0 {res.r0:.3f} (true {r07:.3f}), TVD {err:.2f}",
            )
            if k == 0 and gen == "fft":
                model = {}
                for i, h in enumerate(res.altitude):
                    R = prof.response_maps(h, res.L0)
                    for key, m in R.items():
                        model[key] = model.get(key, 0) + res.r0m53[i] * m
                f2 = prof.plot_covariance_maps((0, 3), components=("xx", "yy"), model=model)
                f2.savefig(OUT / "fig3_covariance_maps_pair03.png", dpi=110)
                plt.close(f2)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_mkr7_profiles.png", dpi=110)
    plt.close(fig)

    save("mkr7")


if not done("mkr7"):
    section_mkr7()

# ---------------------------------------------------------------------------
# 3. Statistics vs number of frames, noise and L0 (Gaussian draws: cheap and exact model)
prof = slodarProfiler(VALID, D, WIDE, gsHeight=H_LGS, L0=L0_TRUE)
grid = prof.default_altitudes(hMax=20e3)
truth7 = S.bin_profile(alts7, fr7, grid)
w7 = prof.profile_to_r0m53(r07, fr7)
C7 = {L0v: prof.slope_covariance(alts7, w7, L0v) for L0v in (10.0, 25.0, 50.0, 100.0)}
evd = {}


def draw(L0v, n, rng, noise=0.0):
    if L0v not in evd:
        e, v = np.linalg.eigh(C7[L0v])
        evd[L0v] = v * np.sqrt(np.clip(e, 0, None))
    s = rng.standard_normal((n, evd[L0v].shape[0])) @ evd[L0v].T
    if noise:
        s += noise * rng.standard_normal(s.shape)
    return s


sig_turb = float(np.sqrt(np.mean(np.diag(C7[25.0]))))
print(f"== 3. Monte Carlo (turbulent slope rms {sig_turb:.3e} rad)")

nSeeds = 8
frames_list = [250, 500, 1000, 2000, 4000, 8000, 16000]
mc_frames = {
    "nFrames": frames_list,
    "r0_err_mean": [],
    "r0_err_std": [],
    "tvd_mean": [],
    "tvd_std": [],
}
if not done("frames"):
    for n in frames_list:
        r0e, tv = [], []
        for sd in range(nSeeds):
            rng = np.random.default_rng(1000 * n + sd)
            prof.compute_covariance_maps(draw(25.0, n, rng), removeTT=True)
            res = prof.fit_profile(altitudes=grid)
            r0e.append(res.r0 / r07 - 1)
            tv.append(tvd(res.fractionalCn2, truth7))
        for key, v in (("r0_err", r0e), ("tvd", tv)):
            mc_frames[key + "_mean"].append(float(np.mean(v)))
            mc_frames[key + "_std"].append(float(np.std(v)))
        print(f"  N={n:6d}: r0 err {np.mean(r0e):+.3%} +- {np.std(r0e):.3%}, TVD {np.mean(tv):.3f}")
    results["frames"] = mc_frames
    save("frames")

noise_list = [0.0, 0.25, 0.5, 1.0, 2.0]
mc_noise = {"noise_over_turb": noise_list}
if not done("noise"):
    for fitNoise in (True, False):
        key = "fitNoise" if fitNoise else "noNoiseTerm"
        mc_noise[key] = {"r0_err_mean": [], "r0_err_std": [], "tvd_mean": [], "tvd_std": []}
        for nz in noise_list:
            r0e, tv = [], []
            for sd in range(nSeeds):
                rng = np.random.default_rng(77 + sd)
                prof.compute_covariance_maps(draw(25.0, 4000, rng, nz * sig_turb), removeTT=True)
                res = prof.fit_profile(altitudes=grid, fitNoise=fitNoise)
                r0e.append(res.r0 / r07 - 1)
                tv.append(tvd(res.fractionalCn2, truth7))
            mc_noise[key]["r0_err_mean"].append(float(np.mean(r0e)))
            mc_noise[key]["r0_err_std"].append(float(np.std(r0e)))
            mc_noise[key]["tvd_mean"].append(float(np.mean(tv)))
            mc_noise[key]["tvd_std"].append(float(np.std(tv)))
            print(
                f"  noise {nz:.2f} x turb, {key:12s}: "
                f"r0 err {np.mean(r0e):+.3%}, TVD {np.mean(tv):.3f}"
            )
    results["noise"] = mc_noise
    save("noise")

L0_list = [10.0, 25.0, 50.0, 100.0]
mc_L0 = {"L0_true": L0_list, "fixed25": {}, "fitL0": {}}
for key in ("fixed25", "fitL0"):
    mc_L0[key] = {
        "r0_err_mean": [],
        "r0_err_std": [],
        "tvd_mean": [],
        "L0_est_mean": [],
        "L0_est_std": [],
    }
if not done("L0"):
    for L0v in L0_list:
        for key in ("fixed25", "fitL0"):
            r0e, tv, l0e = [], [], []
            for sd in range(nSeeds):
                rng = np.random.default_rng(555 + sd)
                prof.compute_covariance_maps(draw(L0v, 4000, rng), removeTT=True)
                if key == "fitL0":
                    res = prof.fit_profile(
                        altitudes=grid, fitL0=True, L0Grid=np.geomspace(5, 200, 14)
                    )
                else:
                    res = prof.fit_profile(altitudes=grid, L0=25.0)
                r0e.append(res.r0 / r07 - 1)
                tv.append(tvd(res.fractionalCn2, truth7))
                l0e.append(res.L0)
            mc_L0[key]["r0_err_mean"].append(float(np.mean(r0e)))
            mc_L0[key]["r0_err_std"].append(float(np.std(r0e)))
            mc_L0[key]["tvd_mean"].append(float(np.mean(tv)))
            mc_L0[key]["L0_est_mean"].append(float(np.mean(l0e)))
            mc_L0[key]["L0_est_std"].append(float(np.std(l0e)))
            print(
                f"  L0 true {L0v:5.0f}, {key:8s}: r0 err {np.mean(r0e):+.2%}, "
                f"TVD {np.mean(tv):.3f}, "
                f"L0 est {np.mean(l0e):.1f}"
            )
    results["L0"] = mc_L0
    save("L0")

# Tip-tilt removal (needed for LGS) vs not
tt = {}
if not done("tiptilt"):
    for rtt in (False, True):
        r0e, tv = [], []
        for sd in range(nSeeds):
            rng = np.random.default_rng(999 + sd)
            prof.compute_covariance_maps(draw(25.0, 4000, rng), removeTT=rtt)
            res = prof.fit_profile(altitudes=grid)
            r0e.append(res.r0 / r07 - 1)
            tv.append(tvd(res.fractionalCn2, truth7))
        tt["removeTT" if rtt else "keepTT"] = {
            "r0_err_mean": float(np.mean(r0e)),
            "r0_err_std": float(np.std(r0e)),
            "tvd_mean": float(np.mean(tv)),
        }
        print(f"  removeTT={rtt}: r0 err {np.mean(r0e):+.3%}, TVD {np.mean(tv):.3f}")
    results["tiptilt"] = tt
    save("tiptilt")

mc_frames, mc_noise, mc_L0 = results["frames"], results["noise"], results["L0"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
ax = axes[0]
ax.errorbar(
    frames_list,
    np.array(mc_frames["r0_err_mean"]) * 100,
    np.array(mc_frames["r0_err_std"]) * 100,
    marker="o",
    label="r0 error [%]",
)
ax2 = ax.twinx()
ax2.errorbar(
    frames_list,
    mc_frames["tvd_mean"],
    mc_frames["tvd_std"],
    marker="s",
    color="C1",
    label="profile TVD",
)
ax.set_xscale("log")
ax.set_xlabel("independent frames")
ax.set_ylabel("r0 error [%]")
ax2.set_ylabel("profile TVD", color="C1")
ax.axhline(0, color="0.6", lw=0.8)
ax.set_title("Convergence with frame count (8 seeds)")
ax = axes[1]
for key, mk in (("fitNoise", "o"), ("noNoiseTerm", "s")):
    ax.errorbar(
        noise_list,
        np.array(mc_noise[key]["r0_err_mean"]) * 100,
        np.array(mc_noise[key]["r0_err_std"]) * 100,
        marker=mk,
        label=f"r0 err, {key}",
    )
ax.set_xlabel("noise rms / turbulent slope rms")
ax.set_ylabel("r0 error [%]")
ax.axhline(0, color="0.6", lw=0.8)
ax2 = ax.twinx()
for key, mk in (("fitNoise", "o"), ("noNoiseTerm", "s")):
    ax2.plot(
        noise_list,
        mc_noise[key]["tvd_mean"],
        mk + "--",
        color="C2" if key == "fitNoise" else "C3",
        label=f"TVD, {key}",
    )
ax2.set_ylabel("profile TVD")
ax.legend(fontsize=7, loc="upper left")
ax2.legend(fontsize=7, loc="upper right")
ax.set_title("White slope noise (4000 frames)")
ax = axes[2]
for key, mk in (("fixed25", "o"), ("fitL0", "s")):
    ax.errorbar(
        L0_list,
        np.array(mc_L0[key]["r0_err_mean"]) * 100,
        np.array(mc_L0[key]["r0_err_std"]) * 100,
        marker=mk,
        label=f"r0 err, {key}",
    )
ax.set_xscale("log")
ax.set_xlabel("true L0 [m]")
ax.set_ylabel("r0 error [%]")
ax.axhline(0, color="0.6", lw=0.8)
ax.legend(fontsize=7)
ax.set_title("Outer scale mismatch (fit L0 = 25 m vs grid search)")
fig.tight_layout()
fig.savefig(OUT / "fig4_statistics.png", dpi=110)
plt.close(fig)

with open(OUT / "stage2_results.json", "w") as f:
    json.dump(results, f, indent=1, default=float)
print("done ->", OUT)
