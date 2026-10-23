# SLODAR end-to-end with SPECULA

Validates `pyTomoAO.slodar` on slopes from the SPECULA simulator, for a Keck-like LTAO
system:

- **Telescope:** 10 m pupil, sampled with 160 pixels.
- **Guide stars and sensors:** 6 sodium LGS at 90 km on a 60″ ring, each with a 20×20
  Shack–Hartmann on the KAPA valid map (SPECULA's ideal, noiseless derivative sensor).
- **Atmosphere:** 7-layer Maunakea Ridge median profile (KAON 503), r0 = 0.16 m at 500 nm,
  L0 = 25 m.

## Requirements

- pyTomoAO installed from this repository: `pip install -e ".[plot]"`
- SPECULA (`pip install specula`, tested with 1.1.0); CuPy as well for `--gpu`.

## Stages

| Script | What it does | Writes (in its output folder) |
|---|---|---|
| `make_subapdata.py ROOT` | SPECULA `SubapData` for the KAPA map, subapertures row-major (called by `run_specula.py`) | `subapdata/kapa_n20_ideal.fits`, `validLLMap.npy` |
| `run_specula.py WORKDIR [T] [dt] [seed] [--gpu]` | Runs the open-loop simulation for `T` s, sampled every `dt` s, and exports the slopes in radians | `calib/`, `output/<timestamp>/` (SPECULA data store), `slopes_<timestamp>.npz` |
| `run_chunks.sh ROOT N [T] [dt]` | Runs `N` independent chunks with seeds `101…`, skipping finished ones, so an interrupted campaign can be restarted (POSIX shell; see below for PowerShell) | `ROOT/chunk<k>/…`, `ROOT/chunk<k>.log` |
| `analyze_specula.py OUTDIR SLOPES.npz [...]` | Concatenates chunks (each chunk's mean removed) and fits the profile with several analysis choices | `fig5_specula_profile.png`, `fig6_*`, `fig7_*`, `stage3_results.json` |

`params_ltao_slodar.yml` is the SPECULA system. `dt = 0.25 s` keeps consecutive frames
close to independent for the ground layer while sampling the wind-blown high layers.

## Runtime

One simulated step (6 LGS, 7 layers, 160-pixel pupil) takes about 1 s on two CPU cores and
about 0.3 s on a desktop GPU. The run is dominated by per-step overhead rather than
arithmetic, so the GPU gain is modest. The validation used 1600 frames: 4 chunks of 100 s at
`dt = 0.25 s`, about 30 minutes on CPU and 9 minutes on GPU. The analysis takes about a
minute.

## Example

```sh
for k in 1 2 3 4; do python run_specula.py runs/chunk$k 100 0.25 $((100 + k)); done
python analyze_specula.py runs/analysis runs/chunk*/slopes_*.npz
```

PowerShell (Windows):

```powershell
foreach ($k in 1..4) { python run_specula.py runs\chunk$k 100 0.25 (100 + $k) --gpu }
$files = (Get-ChildItem runs\chunk*\slopes_*.npz).FullName
python analyze_specula.py runs\analysis @files
```

Expected result: r0 within about 1% of 0.160 m, and the altitude groups within about 0.02
in fractional Cn². The 0.5 and 1 km layers merge with the ground at this 0.85 km bin
resolution. See the SLODAR user guide for the full validation.
