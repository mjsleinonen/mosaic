import pytest
from app.model import Tiles

@pytest.fixture
def tyhja_model():
    t = Tiles()
    return t

@pytest.fixture
def variable_keys():
    varkeys = ["imagepath",
                "tileimage",
                "blend_original", 
                "close_to_clusters",
                "npixel", 
                "misize",
                "nclust"
    ]
    return varkeys

def test_model_init(tyhja_model):
    assert hasattr(tyhja_model, "variables")

def test_model_variable_keys(tyhja_model, variable_keys):
    assert hasattr(tyhja_model, "variables")

    for vk in variable_keys:
        assert vk in tyhja_model.variables

def test_model_init(tyhja_model, capsys):
    tyhja_model.get_picture("\\\\")
    kaapatut = capsys.readouterr()
    assert kaapatut.out == "...\n"