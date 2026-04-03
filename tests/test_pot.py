import os
from src.utils import read_pot, write_pot
from src.config import PROJECT_TESTS_DIR, SOURCE_DATA_DIR
import numpy as np

PATH_IN = SOURCE_DATA_DIR / "MNI152_E.pot"
PATH_OUT = PROJECT_TESTS_DIR / "MNI152_E_tmp.pot"

def _equality(a, b):
    return (a == b) | (np.isnan(a) & np.isnan(b))
    
def test_pot_integrity():
    # clear temp files
    if PATH_OUT.exists():
        PATH_OUT.unlink()
    
    mat_in = read_pot(PATH_IN)
    write_pot(PATH_OUT, mat_in)
    mat_out = read_pot(PATH_OUT)
    
    # clear temp files
    if PATH_OUT.exists():
        PATH_OUT.unlink()
    
    assert mat_out.shape == mat_in.shape, f"Shapes don't match: mat_in: {mat_in.shape}, mat_out: {mat_out.shape}"
    assert _equality(mat_in, mat_out).all(), f"Element-wise equality failed, mismatch element count: {mat_in[mat_out != mat_in].shape[0]}"
