import pytest
from app.view_gui import Gui

from tkinter import *
from tkinter import Frame, Canvas, Label, Button, Message, LEFT, RIGHT, BOTTOM, ALL, Tk, Entry, BOTH, S, HORIZONTAL
from tkinter import filedialog
from random import randint
import tkinter as tk
from tkinter import ttk

@pytest.fixture
def tyhja_gui():
    """."""
    return Gui()

def gui_testi_init():
    w = Tk()
    lx,ly = 1000,500
    size = "{}x{}".format(lx,ly)
    w.title("Mosaic image creator")
    w.geometry(size)
    w.resizable(1, 1) 
    g = Gui(w)
    w.mainloop()

    assert hasattr(g, "root")