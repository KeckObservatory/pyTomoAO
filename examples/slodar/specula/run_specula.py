"""Stage 3a: run the SPECULA LTAO open-loop simulation and export the slopes for SLODAR.

Usage:
    python run_specula.py WORKDIR [total_time_s] [time_step_s] [seed] [--gpu]

``--gpu`` runs SPECULA on GPU 0 (needs CuPy); the default is the CPU.

WORKDIR receives ``calib/`` (SubapData), ``output/<TN>/`` (SPECULA DataStore) and
``slopes_<TN>.npz`` with:

- ``slopes``  (nFrames, 6*2*304) in radians, per WFS X block then Y block, subapertures
  row-major (use ``subapOrder="C"``);
- ``gsDirections`` (6, 2) [rad], ``gsHeight`` [m], ``validLLMap``, ``D``, ``time_step``.

Slope units: SPECULA's IdealDerivativeSensor returns
``slope = grad[nm/px] * 1e-9 / pitch * RAD2ASEC / (fov/2) * (np_sub-1)/np_sub``, so
``rad = slope * (fov/2) / RAD2ASEC * np_sub / (np_sub-1)`` (checked with a pure tilt).
"""

import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
RAD2ASEC = 180 * 3600 / np.pi


def main(workdir, total_time=400.0, time_step=0.25, seed=11, gpu=False):
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(HERE))
    import make_subapdata

    make_subapdata.main(workdir / "calib")
    params = workdir / "params_ltao_slodar.yml"
    shutil.copy(HERE / "params_ltao_slodar.yml", params)
    override = workdir / "override_run.yml"
    override.write_text(
        yaml.safe_dump(
            {
                "main_override": {
                    "total_time": float(total_time),
                    "time_step": float(time_step),
                },
                "atmo_override": {"seed": int(seed)},
            }
        )
    )

    import os

    import specula

    specula.init(0 if gpu else -1, precision=0)
    from importlib import import_module

    from astropy.io import fits

    Simul = import_module("specula.simul").Simul

    os.chdir(workdir)
    before = set((workdir / "output").glob("2*")) if (workdir / "output").exists() else set()
    t0 = time.time()
    Simul(str(params), str(override)).run()
    print(f"SPECULA run: {time.time() - t0:.0f} s")
    tn = sorted(set((workdir / "output").glob("2*")) - before)[-1]

    with open(params) as f:
        cfg = yaml.safe_load(f)
    fov = cfg["wfs1"]["fov"]
    np_sub = cfg["main"]["pixel_pupil"] // 20
    unit = (fov / 2) / RAD2ASEC * np_sub / (np_sub - 1)
    slopes = []
    gs = []
    for k in range(1, 7):
        with fits.open(tn / f"s{k}.fits") as h:
            slopes.append(np.asarray(h[0].data, float) * unit)
        r, az = cfg[f"lgs{k}"]["polar_coordinates"]
        gs.append([r / RAD2ASEC * np.cos(np.radians(az)), r / RAD2ASEC * np.sin(np.radians(az))])
    slopes = np.concatenate(slopes, axis=1)
    out = workdir / f"slopes_{tn.name}.npz"
    np.savez_compressed(
        out,
        slopes=slopes,
        gsDirections=np.array(gs),
        gsHeight=float(cfg["lgs1"]["height"]),
        validLLMap=np.load(workdir / "calib" / "validLLMap.npy"),
        D=cfg["main"]["pixel_pupil"] * cfg["main"]["pixel_pitch"],
        time_step=float(time_step),
        altitude=np.array(cfg["atmo"]["heights"], float),
        fraction=np.array(cfg["atmo"]["Cn2"], float),
        windSpeed=np.array(cfg["wind_speed"]["constant"], float),
        seeing=float(cfg["seeing"]["constant"]),
        L0=float(np.atleast_1d(cfg["atmo"]["L0"])[0]),
    )
    print("wrote", out, slopes.shape)
    return out


if __name__ == "__main__":
    gpu = "--gpu" in sys.argv[1:]
    a = [x for x in sys.argv[1:] if x != "--gpu"]
    main(
        a[0] if a else "./specula_run",
        float(a[1]) if len(a) > 1 else 400.0,
        float(a[2]) if len(a) > 2 else 0.25,
        int(a[3]) if len(a) > 3 else 11,
        gpu=gpu,
    )
