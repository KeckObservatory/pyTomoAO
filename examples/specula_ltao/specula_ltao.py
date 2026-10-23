"""Run a pyTomoAO tomographic reconstructor inside a SPECULA LTAO closed loop.

Companion script of the "pyTomoAO in a SPECULA closed loop" tutorial. Each step writes its
products to WORKDIR (default ``./specula_ltao_run``), so the steps can be rerun separately::

    python specula_ltao.py setup                 # SPECULA subapertures, DM, pyTomoAO config
    python specula_ltao.py calibrate             # interaction matrix, measured in SPECULA
    python specula_ltao.py build --alpha 1e-4    # pyTomoAO MMSE reconstructor -> SPECULA recmat
    python specula_ltao.py run --time 2 --gpu    # closed loop (500 Hz)
    python specula_ltao.py analyse               # residual WFE and H-band Strehl

Add ``--workdir DIR`` to any step. ``build --profile kapa`` builds the reconstructor on the
generic profile of pyTomoAO's KAPA example configuration instead of the simulated one, to
see what a profile mismatch costs.

Requires SPECULA (``pip install specula``); ``--gpu`` also needs CuPy.
"""

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import yaml

from pyTomoAO import example_config, specula_slope_scale
from pyTomoAO.reconstructor import tomographicReconstructor

HERE = Path(__file__).resolve().parent

# ---- System (must agree with params_ltao.yml) --------------------------------------------
PIXEL_PUPIL = 160  # SPECULA pupil sampling [px]
D = 10.0  # telescope diameter [m]
N_LENSLET = 20  # KAPA lenslet array
N_LGS = 6
LGS_RADIUS = 60.0  # [arcsec], azimuths 0, 60, ..., 300 deg as in pyTomoAO
LGS_HEIGHT = 90e3  # [m]
WFS_FOV = 4.0  # SPECULA sensor fov [arcsec]; sets its slope unit
DM_PITCH = 0.5  # [m]
DM_COUPLING = 0.15  # influence function value at the neighbouring actuator
# Atmosphere simulated by SPECULA: Maunakea Ridge median (KAON 503), r0 at 500 nm, L0 = 25 m.
ATM = {
    "altitude_m": [0.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0],
    "cn2": [0.517, 0.119, 0.063, 0.061, 0.105, 0.081, 0.054],
    "seeing_arcsec": 0.6288,
    "L0": 25.0,
}
R0 = 0.9759 * 0.5e-6 / (ATM["seeing_arcsec"] / 206264.806)

SUBAP_TAG = "kapa_n20_ideal"
IFUNC_TAG = "kapa_dm349"
IM_TAG = "kapa_dm349_6lgs_im"
REC_TAG = "pytomoao_rec"


def _init_specula(gpu=False, precision=0):
    import specula

    specula.init(0 if gpu else -1, precision=precision)
    return specula


def _kapa_config():
    with open(example_config("kapa")) as f:
        return yaml.safe_load(f)


def _actuators():
    """Actuator positions [m] (row-major over the KAPA 21x21 valid map)."""
    va = np.array(_kapa_config()["dm_parameters"]["validActuators"], dtype=bool)
    n = va.shape[0]
    r, c = np.nonzero(va)
    return (c - (n - 1) / 2) * DM_PITCH, (r - (n - 1) / 2) * DM_PITCH


def _influence(x, y):
    """Gaussian influence functions at points (x, y) [m]: (nPoints, nActuators), peak 1."""
    ax, ay = _actuators()
    sigma = DM_PITCH / np.sqrt(2 * np.log(1 / DM_COUPLING))
    return np.exp(-((x[:, None] - ax) ** 2 + (y[:, None] - ay) ** 2) / (2 * sigma**2))


def _pupil_grid(pupil):
    """Pupil-plane coordinates [m] of SPECULA's pixels; x along columns, y along rows."""
    c = -D / 2 + (np.arange(pupil.shape[0]) + 0.5) * D / pupil.shape[0]
    y, x = np.meshgrid(c, c, indexing="ij")
    return x, y


# ---- Step 1: setup -------------------------------------------------------------------------
def setup(workdir):
    """SPECULA subaperture data and DM influence functions, plus the pyTomoAO config."""
    _init_specula()
    from specula.data_objects.ifunc import IFunc
    from specula.data_objects.pupilstop import Pupilstop
    from specula.data_objects.simul_params import SimulParams
    from specula.data_objects.subap_data import SubapData

    calib = workdir / "calib"
    cfg = _kapa_config()

    # Subapertures, enumerated row-major (np.nonzero order): SPECULA slopes are "C" ordered.
    valid = np.array(cfg["lgs_wfs_parameters"]["validLLMap"], dtype=bool)
    npx = PIXEL_PUPIL // N_LENSLET
    rows, cols = np.nonzero(valid)
    idx = []
    for r, c in zip(rows, cols):
        rr, cc = np.mgrid[r * npx : (r + 1) * npx, c * npx : (c + 1) * npx]
        idx.append(np.ravel_multi_index((rr.ravel(), cc.ravel()), (PIXEL_PUPIL, PIXEL_PUPIL)))
    (calib / "subapdata").mkdir(parents=True, exist_ok=True)
    SubapData(
        idxs=np.array(idx), display_map=rows * N_LENSLET + cols, nx=N_LENSLET, ny=N_LENSLET
    ).save(str(calib / "subapdata" / f"{SUBAP_TAG}.fits"), overwrite=True)

    # DM influence functions on SPECULA's pupil (unit command = 1 nm at the actuator peak).
    pupil = np.asarray(
        Pupilstop(SimulParams(pixel_pupil=PIXEL_PUPIL, pixel_pitch=D / PIXEL_PUPIL)).A
    )
    pupil = pupil > 0.5
    x, y = _pupil_grid(pupil)
    ifunc = _influence(x[pupil], y[pupil]).T  # (nAct, nPupilPixels)
    (calib / "ifunc").mkdir(parents=True, exist_ok=True)
    IFunc(ifunc=ifunc, mask=pupil.astype(float)).save(
        str(calib / "ifunc" / f"{IFUNC_TAG}.fits"), overwrite=True
    )

    # pyTomoAO configuration matching the SPECULA system.
    n = len(ATM["altitude_m"])
    cn2 = np.array(ATM["cn2"])
    cfg["atmosphere_parameters"].update(
        nLayer=n,
        zenithAngleInDeg=0.0,
        altitude=[a / 1000 for a in ATM["altitude_m"]],  # km
        fractionnalR0=(cn2 / cn2.sum()).tolist(),
        r0=float(R0),
        L0=ATM["L0"],
        wavelength=5e-7,
        windSpeed=[10.0] * n,
        windDirection=[0.0] * n,
    )
    cfg["lgs_asterism"].update(radiusAst=LGS_RADIUS, nLGS=N_LGS, baseLGSHeight=LGS_HEIGHT)
    cfg["lgs_wfs_parameters"]["wfsLensletsRotation"] = [0.0] * N_LGS
    cfg["lgs_wfs_parameters"]["wfsLensletsOffset"] = [[0.0] * N_LGS, [0.0] * N_LGS]
    cfg["tomography_parameters"] = {"fovOptimization": 0, "nFitSrc": 1}  # on-axis
    with open(workdir / "pytomoao_config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)
    print(f"setup: {len(idx)} subapertures, {ifunc.shape[0]} actuators, config written")


# ---- Step 2: interaction matrix ------------------------------------------------------------
def calibrate(workdir, gpu=False):
    """Push-pull IM of the ground DM on one LGS WFS, tiled over the six sensors."""
    _init_specula(gpu)
    from specula.data_objects.intmat import Intmat
    from specula.simul import Simul

    shutil.copy(HERE / "calib_im.yml", workdir / "calib_im.yml")
    cwd = Path.cwd()
    os.chdir(workdir)
    try:
        Simul("calib_im.yml").run()
    finally:
        os.chdir(cwd)
    single = workdir / "calib" / "im" / "kapa_dm349_lgs_im.fits"
    one = np.asarray(Intmat.restore(str(single)).intmat)
    Intmat(np.vstack([one] * N_LGS)).save(
        str(workdir / "calib" / "im" / f"{IM_TAG}.fits"), overwrite=True
    )
    print(f"calibrate: IM {one.shape} per WFS, tiled to {(N_LGS * one.shape[0], one.shape[1])}")


# ---- Step 3: reconstructor -----------------------------------------------------------------
def build(workdir, alpha=1e-4, profile="true"):
    """pyTomoAO MMSE reconstructor + DM fitting, in SPECULA units, saved as a Recmat."""
    config = workdir / "pytomoao_config.yaml"
    if profile == "kapa":  # generic profile of the KAPA example (zenith set to 0)
        cfg = yaml.safe_load(config.read_text())
        cfg["atmosphere_parameters"] = dict(
            _kapa_config()["atmosphere_parameters"], zenithAngleInDeg=0.0
        )
        config = workdir / "pytomoao_config_kapa_profile.yaml"
        config.write_text(yaml.safe_dump(cfg))

    rec = tomographicReconstructor(str(config))
    rec.build_reconstructor(alpha=alpha)

    # 1. SPECULA slopes -> pyTomoAO slope convention (d x angle [m], gradient-operator order)
    P = rec.slope_conversion_matrix(
        subapOrder="C", slopeScale=specula_slope_scale(WFS_FOV, PIXEL_PUPIL // N_LENSLET)
    )
    # 2. slopes -> on-axis OPD [m] on the reconstruction grid
    R = rec.RecStatSA
    # 3. OPD -> DM commands: least-squares fit of the same influence functions on that grid
    #    (the grid spans DSupport, which grid_coordinates() accounts for)
    gx, gy = rec.grid_coordinates()
    F = np.linalg.pinv(_influence(gx, gy), rcond=1e-6)
    recmat = 1e9 * F @ R @ P  # SPECULA slopes -> commands [nm]

    _init_specula()
    from specula.data_objects.recmat import Recmat

    (workdir / "calib" / "rec").mkdir(parents=True, exist_ok=True)
    Recmat(recmat).save(str(workdir / "calib" / "rec" / f"{REC_TAG}.fits"), overwrite=True)
    print(f"build: profile={profile}, alpha={alpha:g}, recmat {recmat.shape}")


# ---- Step 4: closed loop -------------------------------------------------------------------
def run(workdir, total_time=2.0, gain=0.5, seed=311, gpu=False, precision=0):
    """SPECULA closed loop; stores the on-axis residual phase every 5th step."""
    _init_specula(gpu, precision)
    from astropy.io import fits
    from specula.simul import Simul

    shutil.copy(HERE / "params_ltao.yml", workdir / "params_ltao.yml")
    override = {
        "main_override": {"total_time": float(total_time)},
        "atmo_override": {"seed": int(seed)},
        "control_override": {"int_gain": [float(gain)]},
    }
    (workdir / "override.yml").write_text(yaml.safe_dump(override))
    out = workdir / "output"
    before = set(out.glob("2*")) if out.exists() else set()
    cwd = Path.cwd()
    os.chdir(workdir)
    try:
        Simul("params_ltao.yml", "override.yml").run()
    finally:
        os.chdir(cwd)
    tn = sorted(set(out.glob("2*")) - before)[-1]
    with fits.open(tn / "sci.fits") as h:
        ef = np.asarray(h[0].data, float)  # (nSamples, 2, N, N): amplitude, phase [nm]
    dt = 5 * yaml.safe_load((HERE / "params_ltao.yml").read_text())["main"]["time_step"]
    np.savez_compressed(
        workdir / "closed_loop.npz",
        resPhaseNm=ef[:, 1],
        pupil=ef[0, 0] > 0.5,
        t=np.arange(ef.shape[0]) * dt,
    )
    print(f"run: {ef.shape[0]} samples saved from {tn.name}")


# ---- Step 5: analysis ----------------------------------------------------------------------
def analyse(workdir, settle=0.1, wavelength_nm=1650.0):
    """Residual WFE (piston and tip-tilt removed) and long-exposure Strehl on axis."""
    z = np.load(workdir / "closed_loop.npz")
    valid = np.array(_kapa_config()["lgs_wfs_parameters"]["validLLMap"], dtype=bool)
    npx = PIXEL_PUPIL // N_LENSLET
    # Evaluate where the WFSs sense the wavefront: KAPA's map has a central gap that no
    # reconstructor can correct, which would otherwise dominate the residual.
    m = z["pupil"] & np.kron(valid, np.ones((npx, npx), bool))
    x, y = _pupil_grid(m)
    B = np.stack([np.ones(m.sum()), x[m], y[m]], axis=1)
    ph = z["resPhaseNm"][z["t"] >= settle][:, m]
    res = ph - (B @ np.linalg.lstsq(B, ph.T, rcond=None)[0]).T
    wfe = float(np.sqrt(np.mean(res**2)))

    n = m.shape[0]
    pad = 4 * n
    amp = np.zeros((pad, pad))
    amp[:n, :n] = m
    psf = np.zeros((pad, pad))
    for r in res:
        e = np.zeros((pad, pad), complex)
        e[:n, :n][m] = np.exp(2j * np.pi * r / wavelength_nm)
        psf += np.abs(np.fft.fft2(e)) ** 2
    strehl = psf.max() / len(res) / (np.abs(np.fft.fft2(amp)) ** 2).max()
    print(
        f"analyse: {len(res)} samples, residual {wfe:.1f} nm rms (TT removed), "
        f"Strehl at {wavelength_nm / 1000:.2f} um = {strehl:.3f}"
    )
    return wfe, strehl


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=["setup", "calibrate", "build", "run", "analyse"])
    ap.add_argument("--workdir", default="specula_ltao_run")
    ap.add_argument("--gpu", action="store_true", help="run SPECULA on GPU 0 (needs CuPy)")
    ap.add_argument("--alpha", type=float, default=1e-4, help="build: regularisation")
    ap.add_argument("--profile", choices=["true", "kapa"], default="true", help="build")
    ap.add_argument("--time", type=float, default=2.0, help="run: simulated time [s]")
    ap.add_argument("--gain", type=float, default=0.5, help="run: integrator gain")
    ap.add_argument("--seed", type=int, default=311, help="run: atmosphere seed")
    ap.add_argument("--precision", type=int, default=0, help="run: 0 double, 1 single")
    a = ap.parse_args()
    wd = Path(a.workdir).resolve()
    wd.mkdir(parents=True, exist_ok=True)
    if a.step == "setup":
        setup(wd)
    elif a.step == "calibrate":
        calibrate(wd, a.gpu)
    elif a.step == "build":
        build(wd, a.alpha, a.profile)
    elif a.step == "run":
        run(wd, a.time, a.gain, a.seed, a.gpu, a.precision)
    else:
        analyse(wd)


if __name__ == "__main__":
    main()
