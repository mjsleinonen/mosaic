import pytest
from app.model import Tiles

@pytest.fixture
def tyhja_model():
    t = Tiles()
    return t

def test_gui_init(tyhja_model):
    assert hasattr(tyhja_model, "variables")