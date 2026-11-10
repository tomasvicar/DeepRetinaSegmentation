"""Extract paired stored RGB packets; no labels or network forwards."""

import argparse
from pathlib import Path

import numpy as np
from deepretina.legacy.matlab_source_auto import prepare_vesclass_matlab_source_auto
from scipy.io import loadmat

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--oracle-dir", type=Path, required=True)
args = parser.parse_args()
root = args.oracle_dir
c = loadmat(root / "source_auto_chain.mat", simplify_cells=True)["cases"][2]
p = prepare_vesclass_matlab_source_auto(c["image"], c["vessel"], dataset="diaretdb")
np.savez_compressed(
    root / "diaret_sensitivity_inputs.npz",
    raw_rgb=c["image"],
    matlab_rgb=c["stored_dicom12"],
    python_rgb=p.image,
)
print(p.image.shape)
