from src.tetcor import unpack_tets, correct_tetrahedrons
import scipy.io as sio
from os.path import dirname, join as pjoin
import pytest
from src.config import SOURCE_DATA_DIR

MAT_PATH = SOURCE_DATA_DIR / "MNI152.mat"

@pytest.fixture
def sample_tet_data():
    mat_contents = sio.loadmat(MAT_PATH)
    pts, ids = unpack_tets(mat_contents, "Geometry")
    return pts, ids

def test_correct_tetrahedrons(sample_tet_data):
    pts, ids = sample_tet_data
    new_ids, n_corrected = correct_tetrahedrons(pts, ids)
    assert n_corrected == 1953198



