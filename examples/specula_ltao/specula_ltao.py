"""Run a pyTomoAO tomographic reconstructor inside a SPECULA LTAO closed loop.

Companion script of the "pyTomoAO in a SPECULA closed loop" tutorial. Each step writes its
products to WORKDIR (default ``./specula_ltao_run``), so the steps can be rerun separately::

    python specula_ltao.py setup                 # SPECULA subapertures, DM, pyTomoAO config
    python specula_ltao.py calibrate             # interaction matrix, measured in SPECULA
    python specula_ltao.py build --alpha 1e-4    # pyTomoAO MMSE reconstructor -> SPECULA recmat
    python specula_ltao.py run --time 2 --gpu    # closed loop (500 Hz)
    python specula_ltao.py analyse               # residual WFE and H-band Strehl

Add ``--workdir DIR`` to any step. ``setup --atmosphere atmosphere_5layer.yml`` simulates
another turbulence profile (default: the KAON 503 Maunakea median); later steps use it.
``build --profile kapa`` builds the reconstructor on the generic profile of pyTomoAO's KAPA
example configuration instead of the simulated one, to see what a profile mismatch costs.

Turbulence profiling from the closed loop (the on-sky cycle)::

    python specula_ltao.py run --time 20 --record --gpu  # also record slopes + commands
    python specula_ltao.py slodar                # pseudo-open-loop slopes -> SLODAR profile
    python specula_ltao.py build --profile slodar  # reconstructor on the measured profile

WFS noise: the default sensors are SPECULA's noiseless ideal derivative sensors.
``setup --wfs sh --photons 200`` uses physical Shack-Hartmanns with a noisy detector instead
(wfs_sh.yml: photon noise at 200 e-/subaperture/frame + 3 e- read noise; ``--photons 0`` gives a
noiseless detector). ``slodar`` measures the slope noise from the recorded telemetry and
``build --noise measured`` puts it into the reconstructor.

Requires SPECULA (``pip install specula``); ``--gpu`` also needs CuPy.
"""

# SPECULA is an optional dependency that is not always installed in the editor's interpreter.
# pyright: reportMissingImports=false

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import yaml

from pyTomoAO import example_config, slodarProfiler, specula_slope_scale
from pyTomoAO.reconstructor import tomographicReconstructor
from pyTomoAO.slodar import ARCSEC2RAD, pseudo_open_loop_slopes

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
# Default atmosphere: Maunakea Ridge median (KAON 503), r0 at 500 nm, L0 = 25 m. Pass another
# with ``setup --atmosphere FILE`` (same keys, see atmosphere_5layer.yml); it is copied to the
# work directory and every later step reads it from there.
DEFAULT_ATM = {
    "name": "Maunakea Ridge median, 7 layers (KAON 503)",
    "altitude_m": [0.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0],
    "cn2": [0.517, 0.119, 0.063, 0.061, 0.105, 0.081, 0.054],
    "seeing_arcsec": 0.6288,
    "L0": 25.0,
    "wind_speed": [6.8, 6.9, 7.1, 7.5, 10.0, 26.9, 18.5],
    "wind_direction": [0.0, 60.0, 135.0, 200.0, 260.0, 320.0, 30.0],
}
ATM_FILE = "atmosphere.yml"
WFS_FILE = "wfs.json"  # sensor choice (setup) and its measured slope unit (calibrate)
# Physical Shack-Hartmann (wfs_sh.yml): LGS magnitude giving 1000 detected photons per
# subaperture and 2 ms frame through its 1 nm band, and the detector read noise.
SH_MAG_1000 = 4.399
SH_READ_NOISE = 3.0  # [e- rms], as in wfs_sh.yml


def _r0(atm):
    """Fried parameter at 500 nm [m] from the seeing FWHM [arcsec]."""
    return 0.9759 * 0.5e-6 / (atm["seeing_arcsec"] * ARCSEC2RAD)


def _load_atmosphere(path):
    atm = dict(yaml.safe_load(Path(path).read_text()))
    n = len(atm["altitude_m"])
    for key in ("cn2", "wind_speed", "wind_direction"):
        if len(atm[key]) != n:
            raise ValueError(f"{path}: '{key}' has {len(atm[key])} entries for {n} layers")
    return atm


def _atmosphere(workdir):
    """The atmosphere of this work directory (written by setup)."""
    f = workdir / ATM_FILE
    return _load_atmosphere(f) if f.exists() else dict(DEFAULT_ATM)


SUBAP_TAG = "kapa_n20_ideal"
IFUNC_TAG = "kapa_dm349"
IM_TAG = "kapa_dm349_6lgs_im"
REC_TAG = "pytomoao_rec"


def _wfs(workdir):
    """Sensor of this work directory (written by setup, completed by calibrate)."""
    import json

    f = workdir / WFS_FILE
    return json.loads(f.read_text()) if f.exists() else {"sensor": "ideal"}


def _slope_scale(workdir):
    """SPECULA slope unit -> radians of angle of arrival, for this work directory's sensor.

    The ideal sensor's unit follows from its field of view. The physical Shack-Hartmann's
    centroids are in their own unit: calibrate measures it as the ratio of its interaction
    matrix to the ideal sensor's (the same DM pokes seen by both).
    """
    scale = specula_slope_scale(WFS_FOV, PIXEL_PUPIL // N_LENSLET)
    wfs = _wfs(workdir)
    if wfs["sensor"] == "ideal":
        return scale
    if "im_ratio" not in wfs:
        raise RuntimeError("Shack-Hartmann slope unit unknown: run the calibrate step first")
    return scale / wfs["im_ratio"]


def _host(a):
    """NumPy copy of an array that may live on the GPU (CuPy)."""
    return a.get() if hasattr(a, "get") else np.asarray(a)


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
def setup(workdir, atmosphere=None, wfs="ideal", photons=200.0):
    """SPECULA subaperture data and DM influence functions, plus the pyTomoAO config.

    ``wfs`` is "ideal" (noiseless derivative sensors) or "sh" (physical Shack-Hartmanns,
    wfs_sh.yml) with ``photons`` detected photons per subaperture and frame (0: noiseless
    detector).
    """
    import json

    atm = dict(DEFAULT_ATM) if atmosphere is None else _load_atmosphere(atmosphere)
    (workdir / ATM_FILE).write_text(yaml.safe_dump(atm, sort_keys=False))
    previous = _wfs(workdir)
    info = {"sensor": wfs}
    if wfs == "sh":
        info.update(photons=float(photons), read_noise=SH_READ_NOISE if photons > 0 else 0.0)
    if previous["sensor"] == wfs:  # same sensor: its IM (noiseless detector) is still valid
        info.update({k: previous[k] for k in ("im_ratio", "im_correlation") if k in previous})
    (workdir / WFS_FILE).write_text(json.dumps(info, indent=1))
    recalibrate = (
        previous["sensor"] != wfs and (workdir / "calib" / "im" / f"{IM_TAG}.fits").exists()
    )
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
    for r, c in zip(rows, cols, strict=True):
        rr, cc = np.mgrid[r * npx : (r + 1) * npx, c * npx : (c + 1) * npx]
        idx.append(np.ravel_multi_index((rr.ravel(), cc.ravel()), (PIXEL_PUPIL, PIXEL_PUPIL)))
    (calib / "subapdata").mkdir(parents=True, exist_ok=True)
    SubapData(
        idxs=np.array(idx), display_map=rows * N_LENSLET + cols, nx=N_LENSLET, ny=N_LENSLET
    ).save(str(calib / "subapdata" / f"{SUBAP_TAG}.fits"), overwrite=True)

    # DM influence functions on SPECULA's pupil (unit command = 1 nm at the actuator peak).
    pupil = _host(Pupilstop(SimulParams(pixel_pupil=PIXEL_PUPIL, pixel_pitch=D / PIXEL_PUPIL)).A)
    pupil = pupil > 0.5
    x, y = _pupil_grid(pupil)
    ifunc = _influence(x[pupil], y[pupil]).T  # (nAct, nPupilPixels)
    (calib / "ifunc").mkdir(parents=True, exist_ok=True)
    IFunc(ifunc=ifunc, mask=pupil.astype(float)).save(
        str(calib / "ifunc" / f"{IFUNC_TAG}.fits"), overwrite=True
    )

    # pyTomoAO configuration matching the SPECULA system.
    n = len(atm["altitude_m"])
    cn2 = np.array(atm["cn2"])
    cfg["atmosphere_parameters"].update(
        nLayer=n,
        zenithAngleInDeg=0.0,
        altitude=[a / 1000 for a in atm["altitude_m"]],  # km
        fractionnalR0=(cn2 / cn2.sum()).tolist(),
        r0=float(_r0(atm)),
        L0=float(atm["L0"]),
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
    label = atm.get("name") or f"custom, {n} layers"  # the bundled names give the count
    print(
        f"setup: {len(idx)} subapertures, {ifunc.shape[0]} actuators, config written; "
        f"atmosphere: {label}, r0 {_r0(atm):.3f} m, L0 {atm['L0']:g} m"
    )
    if wfs == "sh":
        noise = (
            f"{photons:g} photons + {SH_READ_NOISE:g} e- read noise" if photons > 0 else "noiseless"
        )
        print(f"setup: physical Shack-Hartmann WFSs, {noise}")
    if recalibrate:
        print(
            f"setup: sensor changed from {previous['sensor']!r} to {wfs!r}: the interaction "
            "matrix in calib/im/ is for the old sensor, run calibrate again"
        )


# ---- Step 2: interaction matrix ------------------------------------------------------------
def calibrate(workdir, gpu=False):
    """Push-pull IM of the ground DM on one LGS WFS, tiled over the six sensors.

    With Shack-Hartmann sensors, also measures their IM (noiseless detector) and stores its
    ratio to the ideal sensor's IM: the Shack-Hartmann slope unit, used by build and slodar.
    """
    import json

    _init_specula(gpu)
    from specula.data_objects.intmat import Intmat
    from specula.simul import Simul

    wfs = _wfs(workdir)
    files = ["calib_im.yml"] + (["calib_im_sh.yml"] if wfs["sensor"] == "sh" else [])
    for f in files:
        shutil.copy(HERE / f, workdir / f)
    cwd = Path.cwd()
    os.chdir(workdir)
    try:
        Simul("calib_im.yml").run()  # ideal sensor: the slope-unit reference
        if wfs["sensor"] == "sh":
            Simul(*files).run()
    finally:
        os.chdir(cwd)
    im_dir = workdir / "calib" / "im"
    one = _host(Intmat.restore(str(im_dir / "kapa_dm349_lgs_im.fits")).intmat)  # GPU -> host
    if wfs["sensor"] == "sh":
        ideal = one
        one = _host(Intmat.restore(str(im_dir / "kapa_dm349_lgs_im_sh.fits")).intmat)
        ratio = float(np.sum(one * ideal) / np.sum(ideal * ideal))
        corr = float(np.corrcoef(one.ravel(), ideal.ravel())[0, 1])
        wfs.update(im_ratio=ratio, im_correlation=corr)
        (workdir / WFS_FILE).write_text(json.dumps(wfs, indent=1))
        print(
            f"calibrate: Shack-Hartmann IM = {ratio:.3f} x ideal-sensor IM "
            f"(correlation {corr:.3f}): its slope unit is 1/{ratio:.3f} of the ideal sensor's"
        )
    Intmat(np.vstack([one] * N_LGS)).save(str(im_dir / f"{IM_TAG}.fits"), overwrite=True)
    print(f"calibrate: IM {one.shape} per WFS, tiled to {(N_LGS * one.shape[0], one.shape[1])}")


# ---- Step 3: reconstructor -----------------------------------------------------------------
def build(workdir, alpha=1e-4, profile="true", noise=None):
    """pyTomoAO MMSE reconstructor + DM fitting, in SPECULA units, saved as a Recmat.

    The noise model is ``alpha`` times the mean turbulent slope variance (a fixed
    regularisation), or with ``noise="measured"`` the per-WFS slope noise that the slodar step
    measured from closed-loop telemetry.
    """
    import json

    from pyTomoAO.slodar import slope_noise_covariance

    config = workdir / "pytomoao_config.yaml"
    if profile == "kapa":  # generic profile of the KAPA example (zenith set to 0)
        cfg = yaml.safe_load(config.read_text())
        cfg["atmosphere_parameters"] = dict(
            _kapa_config()["atmosphere_parameters"], zenithAngleInDeg=0.0
        )
        config = workdir / "pytomoao_config_kapa_profile.yaml"
        config.write_text(yaml.safe_dump(cfg))
    elif profile == "slodar":  # profile measured from closed-loop telemetry (slodar step)
        config = workdir / "pytomoao_config_slodar.yaml"
        if not config.exists():
            raise FileNotFoundError(f"{config} not found: run the slodar step first")

    rec = tomographicReconstructor(str(config))
    if noise == "measured":  # per-WFS noise [rad^2 of angle] from the slodar step
        f = workdir / "slodar.json"
        if not f.exists():
            raise FileNotFoundError(f"{f} not found: run the slodar step first")
        var = np.array(json.loads(f.read_text())["noise_variance_rad2"])
        Cn = slope_noise_covariance(var, reconstructor=rec)  # -> units of the slope covariance
        rec.build_reconstructor(noiseCovariance=Cn)
        how = f"measured noise {np.sqrt(var.mean()) / ARCSEC2RAD:.3f} arcsec rms"
    else:
        rec.build_reconstructor(alpha=alpha)
        how = f"alpha={alpha:g}"

    # 1. SPECULA slopes -> pyTomoAO slope convention (d x angle [m], gradient-operator order)
    P = rec.slope_conversion_matrix(subapOrder="C", slopeScale=_slope_scale(workdir))
    # 2. slopes -> on-axis OPD [m] on the reconstruction grid (a CuPy array when pyTomoAO
    #    uses the GPU backend, so copy it to the host)
    R = _host(rec.RecStatSA)
    # 3. OPD -> DM commands: least-squares fit of the same influence functions on that grid
    #    (the grid spans DSupport, which grid_coordinates() accounts for)
    gx, gy = rec.grid_coordinates()
    F = np.linalg.pinv(_influence(gx, gy), rcond=1e-6)
    recmat = 1e9 * F @ R @ P  # SPECULA slopes -> commands [nm]

    _init_specula()
    from specula.data_objects.recmat import Recmat

    (workdir / "calib" / "rec").mkdir(parents=True, exist_ok=True)
    Recmat(recmat).save(str(workdir / "calib" / "rec" / f"{REC_TAG}.fits"), overwrite=True)
    print(f"build: profile={profile}, {how}, recmat {recmat.shape}")


# ---- Step 4: closed loop -------------------------------------------------------------------
def run(workdir, *, total_time=2.0, gain=0.5, seed=311, gpu=False, precision=0, record=False):
    """SPECULA closed loop; stores the on-axis residual phase (about 200 samples).

    With ``record``, also stores the six WFSs' slopes and the DM commands at every loop step,
    as closed-loop telemetry for the ``slodar`` step.
    """
    _init_specula(gpu, precision)
    from astropy.io import fits
    from specula.simul import Simul

    atm = _atmosphere(workdir)
    wfs = _wfs(workdir)
    files = ["params_ltao.yml"] + (["wfs_sh.yml"] if wfs["sensor"] == "sh" else [])
    for f in files:
        shutil.copy(HERE / f, workdir / f)
    step = yaml.safe_load((HERE / "params_ltao.yml").read_text())["main"]["time_step"]
    # Science phase is 0.4 MB per sample: keep about 200 of them whatever the run length.
    sciEvery = max(5, round(total_time / step / 200))
    inputs = ["sci-prop.out_sci_ef", "comm-control.out_comm"]
    if record:
        inputs += [f"s{k}-wfs{k}.out_slopes" for k in range(1, N_LGS + 1)]
    override = {
        "main_override": {"total_time": float(total_time)},
        "atmo_override": {
            "seed": int(seed),
            "heights": [float(h) for h in atm["altitude_m"]],
            "Cn2": [float(c) for c in atm["cn2"]],
            "L0": [float(atm["L0"])] * len(atm["altitude_m"]),
        },
        "seeing_override": {"constant": float(atm["seeing_arcsec"])},
        "wind_speed_override": {"constant": [float(v) for v in atm["wind_speed"]]},
        "wind_direction_override": {"constant": [float(v) for v in atm["wind_direction"]]},
        "control_override": {"int_gain": [float(gain)]},
        "data_store_override": {
            "downsample_factor": 1,
            "downsample_factor_by_input": {"sci": sciEvery, **({} if record else {"comm": 5})},
            "inputs": {"input_list": inputs},
        },
    }
    if wfs["sensor"] == "sh":  # LGS flux and detector noise; noise seeds follow the run seed
        photons = wfs["photons"]
        mag = SH_MAG_1000 - 2.5 * np.log10(photons / 1000) if photons > 0 else SH_MAG_1000
        for k in range(1, N_LGS + 1):
            override[f"lgs{k}_override"] = {"magnitude": float(mag)}
            override[f"det{k}_override"] = {
                "photon_noise": photons > 0,
                "readout_noise": photons > 0,
                "photon_seed": int(1000 * seed + 10 * k + 1),
                "readout_seed": int(1000 * seed + 10 * k + 2),
            }
    (workdir / "override.yml").write_text(yaml.safe_dump(override))
    out = workdir / "output"
    before = set(out.glob("2*")) if out.exists() else set()
    cwd = Path.cwd()
    os.chdir(workdir)
    try:
        Simul(*files, "override.yml").run()
    finally:
        os.chdir(cwd)
    tn = sorted(set(out.glob("2*")) - before)[-1]
    with fits.open(tn / "sci.fits") as h:
        ef = np.asarray(h[0].data, float)  # (nSamples, 2, N, N): amplitude, phase [nm]
    np.savez_compressed(
        workdir / "closed_loop.npz",
        resPhaseNm=ef[:, 1],
        pupil=ef[0, 0] > 0.5,
        t=np.arange(ef.shape[0]) * sciEvery * step,
    )
    print(f"run: {ef.shape[0]} science samples saved from {tn.name}")
    if record:
        slopes = []
        for k in range(1, N_LGS + 1):
            with fits.open(tn / f"s{k}.fits") as h:
                slopes.append(np.asarray(h[0].data, np.float32))
        with fits.open(tn / "comm.fits") as h:
            comm = np.asarray(h[0].data, np.float32)
        slopes = np.concatenate(slopes, axis=1)  # SPECULA units, WFS blocks in LGS order
        np.savez_compressed(
            workdir / "telemetry.npz", slopes=slopes, commands=comm, time_step=step, gain=gain
        )
        print(f"run: telemetry {slopes.shape[0]} frames x {slopes.shape[1]} slopes saved")


# ---- Optional: SLODAR from the closed-loop telemetry ---------------------------------------
def slodar(workdir, settle=0.1, L0=None):
    """Turbulence profile and slope noise from closed-loop telemetry, as on sky.

    Rebuilds pseudo-open-loop slopes ``s(t) + M c(t - 1)`` from the recorded slopes and
    commands, fits the SLODAR profile (L0 fitted), estimates the slope noise from their
    temporal covariance, and writes a pyTomoAO configuration on the measured profile for
    ``build --profile slodar``.

    ``L0`` fixes the outer scale instead of fitting it. The L0 fit needs long telemetry: from
    a few seconds it scatters by a factor of two, and a wrong L0 is the most damaging profile
    error for the reconstructor.
    """
    import json

    _init_specula()
    from specula.data_objects.intmat import Intmat

    tel = np.load(workdir / "telemetry.npz")
    slopes, comm = tel["slopes"].astype(float), tel["commands"].astype(float)
    M = _host(Intmat.restore(str(workdir / "calib" / "im" / f"{IM_TAG}.fits")).intmat)
    # SPECULA's DM applies the opposite of the command (sign -1) while the IM was poked with
    # +c, so s = s_ol - M c(t-1): sign=+1, delay 1 frame (the DM layer enters with :-1).
    pol = pseudo_open_loop_slopes(slopes, comm, M, delay=1, sign=1)
    wrong = pseudo_open_loop_slopes(slopes, comm, M, delay=1, sign=-1)
    skip = round(settle / float(tel["time_step"]))
    pol, wrong = pol[skip:], wrong[skip:]
    incr = np.var(np.diff(wrong, axis=0)) / np.var(np.diff(pol, axis=0))
    print(
        f"slodar: {pol.shape[0]} pseudo-open-loop frames; sign check: wrong/right "
        f"increment variance = {incr:.1f} (should be > 1)"
    )

    rec = tomographicReconstructor(str(workdir / "pytomoao_config.yaml"))
    prof = slodarProfiler.from_reconstructor(rec, subapOrder="C", slopeScale=_slope_scale(workdir))
    grid = prof.default_altitudes(hMax=20e3)
    if L0 is None:
        res = prof.profile(pol, removeTT=True, fitL0=True, altitudes=grid)
    else:
        res = prof.profile(pol, removeTT=True, L0=float(L0), altitudes=grid)
    noise = prof.temporal_noise_variance(pol, removeTT=True)  # rad^2 per WFS

    # Input profile binned onto the SLODAR grid (nearest bin), for comparison
    atm = _atmosphere(workdir)
    R0 = _r0(atm)
    cn2 = np.array(atm["cn2"]) / np.sum(atm["cn2"])
    truth = np.zeros(grid.size)
    for h, f in zip(atm["altitude_m"], cn2, strict=True):
        truth[np.argmin(np.abs(grid - h))] += f
    tvd = 0.5 * np.abs(res.fractionalCn2 - truth).sum()
    print(res)
    if res.L0AtGridEdge:
        print(
            f"slodar: WARNING - fitted L0 = {res.L0:g} m is at the edge of the search grid, so "
            "L0 is not constrained by this telemetry. Record longer (or several runs) before "
            "trusting this profile."
        )
    print(
        f"slodar: r0 {res.r0:.4f} m (input {R0:.4f}), L0 {res.L0:.1f} m (input {atm['L0']:g}), "
        f"profile TVD {tvd:.3f}, temporal noise "
        f"{np.sqrt(noise.mean()) / ARCSEC2RAD:.4f} arcsec rms"
    )

    cfg = yaml.safe_load((workdir / "pytomoao_config.yaml").read_text())
    cfg["atmosphere_parameters"] = res.to_atmosphere_config(minFraction=0.01)
    (workdir / "pytomoao_config_slodar.yaml").write_text(yaml.safe_dump(cfg))
    (workdir / "slodar.json").write_text(
        json.dumps(
            {
                "frames": int(pol.shape[0]),
                "r0": res.r0,
                "r0_input": R0,
                "L0": res.L0,
                "L0_input": atm["L0"],
                "tvd": float(tvd),
                "sign_check_increment_ratio": float(incr),
                "altitude_m": grid.tolist(),
                "fraction": res.fractionalCn2.tolist(),
                "fraction_input_binned": truth.tolist(),
                "noise_variance_rad2": noise.tolist(),
            },
            indent=1,
        )
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("slodar: matplotlib not installed, no figure")
        return res
    fig, ax = plt.subplots(figsize=(6, 5))
    hk = grid / 1e3
    w = 0.35 * np.min(np.diff(hk)) if hk.size > 1 else 0.3
    ax.barh(hk - w / 2, truth, height=w, color="0.6", label="input, binned to the SLODAR grid")
    ax.barh(hk + w / 2, res.fractionalCn2, height=w, color="C0", label="SLODAR, closed-loop POL")
    ax.plot(cn2, np.array(atm["altitude_m"]) / 1e3, "rx", ms=8, label="input layers")
    ax.set_xlabel("fractional $C_n^2$")
    ax.set_ylabel("altitude [km]")
    ax.set_title(
        f"r0 {res.r0:.3f} m (input {R0:.3f}), L0 {res.L0:.0f} m, TVD {tvd:.2f}", fontsize=10
    )
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(workdir / "slodar_profile.png", dpi=120)
    plt.close(fig)
    print("slodar: wrote slodar_profile.png, slodar.json and pytomoao_config_slodar.yaml")
    return res


# ---- Step 5: analysis ----------------------------------------------------------------------
def analyse(workdir, settle=0.1, wavelength_nm=1650.0):
    """Residual WFE (piston and tip-tilt removed) and long-exposure Strehl on axis.

    Prints the numbers and writes ``closed_loop.png`` (residual vs time, residual maps and
    the long-exposure PSF) and ``closed_loop.json`` to the work directory.
    """
    import json

    z = np.load(workdir / "closed_loop.npz")
    valid = np.array(_kapa_config()["lgs_wfs_parameters"]["validLLMap"], dtype=bool)
    npx = PIXEL_PUPIL // N_LENSLET
    # Evaluate where the WFSs sense the wavefront: KAPA's map has a central gap that no
    # reconstructor can correct, which would otherwise dominate the residual.
    m = z["pupil"] & np.kron(valid, np.ones((npx, npx), bool))
    x, y = _pupil_grid(m)
    B = np.stack([np.ones(m.sum()), x[m], y[m]], axis=1)
    t = z["t"]
    ph_all = z["resPhaseNm"][:, m]
    res_all = ph_all - (B @ np.linalg.lstsq(B, ph_all.T, rcond=None)[0]).T
    rms_t = np.sqrt(np.mean(res_all**2, axis=1))  # per sample, TT removed
    keep = t >= settle
    res = res_all[keep]
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
    peak_dl = (np.abs(np.fft.fft2(amp)) ** 2).max()
    strehl = psf.max() / len(res) / peak_dl
    print(
        f"analyse: {len(res)} samples, residual {wfe:.1f} nm rms (TT removed), "
        f"Strehl at {wavelength_nm / 1000:.2f} um = {strehl:.3f}"
    )
    summary = {
        "residual_nm_rms_tt_removed": wfe,
        "strehl": float(strehl),
        "wavelength_nm": wavelength_nm,
        "n_samples": len(res),
        "settle_s": settle,
        "input_nm_rms_tt_removed_first_sample": float(rms_t[0]),
    }
    (workdir / "closed_loop.json").write_text(json.dumps(summary, indent=1))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("analyse: matplotlib not installed, no figure (pip install matplotlib)")
        return wfe, strehl

    def as_map(v):
        img = np.full(m.shape, np.nan)
        img[m] = v
        return img

    fig, ax = plt.subplots(1, 4, figsize=(17, 4))
    ax[0].semilogy(t, rms_t, lw=1)
    ax[0].axvline(settle, color="0.6", ls=":", label=f"settling ({settle:g} s, excluded)")
    ax[0].axhline(wfe, color="C1", ls="--", label=f"mean {wfe:.0f} nm")
    ax[0].set_xlabel("time [s]")
    ax[0].set_ylabel("residual, TT removed [nm rms]")
    ax[0].set_title("Closed-loop residual")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, which="both")

    last = res[-1]
    lim = 3 * np.std(last)
    im = ax[1].imshow(as_map(last), cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower")
    ax[1].set_title(f"Residual, last sample ({np.sqrt(np.mean(last**2)):.0f} nm rms)")
    plt.colorbar(im, ax=ax[1], fraction=0.046, label="nm")

    im = ax[2].imshow(as_map(np.sqrt(np.mean(res**2, axis=0))), cmap="viridis", origin="lower")
    ax[2].set_title("Residual rms over time")
    plt.colorbar(im, ax=ax[2], fraction=0.046, label="nm")
    for a_ in ax[1:3]:
        a_.set_xticks([])
        a_.set_yticks([])

    h = 40  # half-width of the PSF crop, in pixels of the 4x-padded FFT (lambda/4D)
    c = pad // 2
    img = np.fft.fftshift(psf / len(res) / peak_dl)[c - h : c + h, c - h : c + h]
    lamD = wavelength_nm * 1e-9 / D * 206265e3  # mas
    ext = np.array([-h, h, -h, h]) * lamD / 4
    im = ax[3].imshow(
        np.log10(img + 1e-6), vmin=-4, vmax=0, cmap="inferno", origin="lower", extent=ext
    )
    ax[3].set_title(f"Long-exposure PSF, {wavelength_nm / 1000:.2f} um, SR {strehl:.2f}")
    ax[3].set_xlabel("mas")
    plt.colorbar(im, ax=ax[3], fraction=0.046, label="log10 (peak = Strehl)")
    fig.tight_layout()
    out = workdir / "closed_loop.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"analyse: wrote {out} and closed_loop.json")
    return wfe, strehl


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=["setup", "calibrate", "build", "run", "slodar", "analyse"])
    ap.add_argument("--workdir", default="specula_ltao_run")
    ap.add_argument(
        "--atmosphere", default=None, help="setup: atmosphere YAML (default: KAON 503 median)"
    )
    ap.add_argument(
        "--wfs",
        choices=["ideal", "sh"],
        default="ideal",
        help="setup: noiseless ideal sensors or physical Shack-Hartmanns (wfs_sh.yml)",
    )
    ap.add_argument(
        "--photons",
        type=float,
        default=200.0,
        help="setup --wfs sh: photons per subaperture and frame (0: noiseless detector)",
    )
    ap.add_argument("--gpu", action="store_true", help="run SPECULA on GPU 0 (needs CuPy)")
    ap.add_argument("--alpha", type=float, default=1e-4, help="build: regularisation")
    ap.add_argument("--profile", choices=["true", "kapa", "slodar"], default="true", help="build")
    ap.add_argument(
        "--noise",
        choices=["measured"],
        default=None,
        help="build: use the slope noise measured by slodar instead of --alpha",
    )
    ap.add_argument(
        "--record", action="store_true", help="run: also save slopes + commands for slodar"
    )
    ap.add_argument("--L0", type=float, default=None, help="slodar: fix L0 [m] (default: fit)")
    ap.add_argument("--time", type=float, default=2.0, help="run: simulated time [s]")
    ap.add_argument("--gain", type=float, default=0.5, help="run: integrator gain")
    ap.add_argument("--seed", type=int, default=311, help="run: atmosphere seed")
    ap.add_argument("--precision", type=int, default=0, help="run: 0 double, 1 single")
    a = ap.parse_args()
    wd = Path(a.workdir).resolve()
    wd.mkdir(parents=True, exist_ok=True)
    if a.step == "setup":
        setup(wd, a.atmosphere, a.wfs, a.photons)
    elif a.step == "calibrate":
        calibrate(wd, a.gpu)
    elif a.step == "build":
        build(wd, a.alpha, a.profile, a.noise)
    elif a.step == "run":
        run(
            wd,
            total_time=a.time,
            gain=a.gain,
            seed=a.seed,
            gpu=a.gpu,
            precision=a.precision,
            record=a.record,
        )
    elif a.step == "slodar":
        slodar(wd, L0=a.L0)
    else:
        analyse(wd)


if __name__ == "__main__":
    main()
