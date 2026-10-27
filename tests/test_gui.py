import pytest
from app.view_gui import Gui

@pytest.fixture
def tyhja_gui():
    gui = Gui()
    yield gui

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

def test_gui_init(tyhja_gui):
    assert hasattr(tyhja_gui, "root")

def test_view_variable_keys(tyhja_gui, variable_keys):
    assert hasattr(tyhja_gui, "variables")

    for vk in variable_keys:
        assert vk in tyhja_gui.variables

def test_gui_preview(tyhja_gui):
    r = tyhja_gui.set_preview()
    assert r is None