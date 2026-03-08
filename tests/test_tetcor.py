import src.tetcor as tetcor
from os.path import dirname, join as pjoin
import pytest

@pytest.fixture
def sample_tet_data():
    data_dir = pjoin(dirname(tetcor.__file__), "data")
    file_name = pjoin(data_dir, "MNI152.mat")
    pts, ids = tetcor.load_tetrahedrons(file_name)
    return pts, ids

def test_correct_tetrahedrons(sample_tet_data):
    pts, ids = sample_tet_data
    new_ids, n_corrected = tetcor.correct_tetrahedrons(pts, ids)
    assert n_corrected == 1953198



