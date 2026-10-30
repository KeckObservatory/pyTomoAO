"""Write the SPECULA SubapData calibration file for the KAPA 20x20 valid-subaperture map.

Subapertures are enumerated row-major (np.nonzero order), so the SPECULA slopes must be read
with ``slodarProfiler(..., subapOrder="C")``. display_map = row * N + col, which is how
IdealDerivativeSensor locates each subaperture on the pupil (X along columns).
"""

import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import yaml

from pyTomoAO import example_config


def main(root, pixel_pupil=160, tag="kapa_n20_ideal"):
    import specula

    specula.init(-1, precision=0)
    SubapData = import_module("specula.data_objects.subap_data").SubapData

    with open(example_config("kapa")) as f:
        valid = np.array(yaml.safe_load(f)["lgs_wfs_parameters"]["validLLMap"], dtype=bool)
    N = valid.shape[0]
    npx = pixel_pupil // N
    rows, cols = np.nonzero(valid)
    tot = N * npx
    idx = []
    for r, c in zip(rows, cols, strict=True):
        rr, cc = np.mgrid[r * npx : (r + 1) * npx, c * npx : (c + 1) * npx]
        idx.append(np.ravel_multi_index((rr.ravel(), cc.ravel()), (tot, tot)))
    sd = SubapData(idxs=np.array(idx), display_map=rows * N + cols, nx=N, ny=N)
    out = Path(root) / "subapdata"
    out.mkdir(parents=True, exist_ok=True)
    sd.save(str(out / f"{tag}.fits"), overwrite=True)
    np.save(Path(root) / "validLLMap.npy", valid)
    print("wrote", out / f"{tag}.fits", "with", len(idx), "subapertures")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "./calib")
