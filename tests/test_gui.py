import pytest
from app.view_gui import Gui

@pytest.fixture
def tyhja_gui():
    gui = Gui()
    yield gui

    gui.root.destroy()

def test_gui_init(tyhja_gui):
    assert hasattr(tyhja_gui, "root")