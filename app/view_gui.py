   
import os
import numpy as np

from tkinter import *
from tkinter import Frame, Canvas, Label, Button, Message, LEFT, RIGHT, BOTTOM, ALL, Tk, Entry, BOTH, S, HORIZONTAL
from tkinter import filedialog
from random import randint
import tkinter as tk
from tkinter import ttk
import tkinter.font as font

from PIL import Image, ImageTk
import PIL

from pathlib import Path

import time
import contextlib

@contextlib.contextmanager
def visit_dir(dirname):
    curdir = os.getcwd()
    
    os.chdir(dirname)
    try:
        yield
    finally:
        os.chdir(curdir)

def ext_check(name, exts):
    return name.split(".")[-1] in exts

def timestamp():
    return int(time.time())
    
def get_filename_dialog(root, initdir="/"):
    root.filename =  filedialog.askopenfilename(initialdir=initdir,title="Select file",filetypes=(("jpeg files","*.jpg"),("all files","*.*")))
    return root.filename
    
def get_foldername_dialog(root, initdir="/"):
    root.foldername =  filedialog.askdirectory(initialdir=initdir,title="Select file")
    return root.foldername

def arvioi_jpeg_koko_mb(leveys: int, korkeus: int, laatu: str = "normaali") -> float:
    # Määritetään kertoimet laadun mukaan
    kertoimet = {
        "korkea": 8,    # Laatu ~95
        "normaali": 20, # Laatu ~80
        "matala": 40    # Laatu ~50
    }
    
    kerroin = kertoimet.get(laatu, 20)
    
    # Lasketaan koko tavuina
    raakatavut = leveys * korkeus * 3
    jpeg_tavut = raakatavut / kerroin
    
    # Muunnetaan megatavuiksi (MB)
    koko_mb = jpeg_tavut / (1024 * 1024)
    return round(koko_mb, 2)

def main_window():

    w = Tk()

    lx,ly = 1000,500
    size = "{}x{}".format(lx,ly)

    w.title("Mosaic image creator")
    w.geometry(size)
    w.resizable(1, 1) 
    
    return w

EXTS = ["jpg","jpeg","png","bmp","tif"]
NO_ARGUMENT = object()
    
class Gui:
    
    def __init__(self, root=NO_ARGUMENT):

        if not root is NO_ARGUMENT:
            self.root = root
        else:
            self.root = main_window()

        self.set_up_gui()

    def set_root(self, root):
        self.root = root

    def set_up_gui(self):
        self.menubar = Menu(self.root)
        
        self.windowsize = self.root.size()
        self.fdx,self.fdy = self.windowsize
        
        "-- option menu widgets"
        
        self.menubar = Menu(self.root)
        self.filemenu = Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=self.filemenu)
        
        self.filemenu.add_command(label ='New File', command = None) 
        self.filemenu.add_command(label ='Open...', command = None) 
        self.filemenu.add_command(label ='Save', command = None) 
        self.filemenu.add_separator() 
        self.filemenu.add_command(label ='Exit', command = self.root.destroy) 
        
        self.root.config(menu = self.menubar)

        "-- gui variables "
        
        fontti = font.Font(size=14)
        fontti2 = font.Font(size=16)
    
        self.imagespath = StringVar()
        cf = os.getcwd()
        filename = cf+"\\"
        self.imagespath.set(filename)
        
        self.tileimage = StringVar()
        self.tileimage.set("")

        self.tile_stats_text = StringVar()
        self.tile_stats_text.set("")
        
        self.blend_original = IntVar()
        self.closer_to_clusters = IntVar()

        self.tile_image_is_set = False
         
        "-- first tab and sub widgets"
        
        #self.tabs = ttk.Notebook(root, width=self.fdx, height=self.fdy)
        self.main = Frame(self.root, highlightbackground="grey", highlightthickness=1)
        self.main.pack(fill="both", expand=True)
        
        f1 = Frame(self.main, highlightbackground="grey", highlightthickness=0)
        #self.tabs.add(f1, text="Create Mosaic Image")
        f1.pack(fill="both", expand=True)
        
        #f2 = Frame(self.main, highlightbackground="grey", highlightthickness=1)
        #f2.pack()
        
        f3 = Frame(f1, highlightbackground="grey", highlightthickness=1)
        f3.pack(fill="both")
        
        f3_1 = Frame(f3)
        f3_1.pack(side=TOP, fill="y", pady=5)
        
        self.foldername = Label(f3_1, textvariable=self.imagespath, width=100)
        self.foldername.pack(side=RIGHT)

        bbtn1 = Button(f3_1, width=20, text="Set picture tiles path...", command=self.set_images_folder)
        bbtn1.pack(side=LEFT)
        
        f3_2 = Frame(f3)
        f3_2.pack(side=BOTTOM, pady=5)
    
        self.filename = Label(f3_2, textvariable=self.tileimage, width=100)
        self.filename.pack(side=RIGHT,  fill="y")
        
        bbtn2 = Button(f3_2, width=20, text="Get mosaic picture", command=self.set_tile_image)
        bbtn2.pack(side=LEFT)

        ########

        f2 = Frame(f1, highlightbackground="grey")
        f2.pack(fill="both", expand=True)
        
        f4 = Frame(f2, highlightbackground="grey", highlightthickness=1)
        f4.pack(side=LEFT, fill="y", pady=1)

        "controls"
        
        self.msize = tk.Scale(f4, label="Decrease mosaic image size to-%", orient=tk.HORIZONTAL, length=200)
        self.msize.pack(anchor=W, padx=10)
        self.msize.set(80)
        
        self.nc = tk.Scale(f4, label="Number of Clusters", orient=tk.HORIZONTAL, length=200)
        self.nc.pack(anchor=W, padx=10)
        self.nc.set(25)
        
        self.pixel_n = tk.Scale(f4, label="Tile frame side n pixels", orient=tk.HORIZONTAL, length=200)
        self.pixel_n.pack(anchor=W, padx=10)
        self.pixel_n.set(32)

        self.cb1 = Checkbutton(f4, text="Blend with original", variable=self.blend_original, padx=20)
        self.cb1.pack(anchor=W)

        self.cb2 = Checkbutton(f4, text="Closer to cluster centers", variable=self.closer_to_clusters, padx=20)
        self.cb2.pack(anchor=W)

        bbtn4 = Button(f4, width=15, text="Get Preview...", font=fontti, height=1, command=self.set_preview, pady=5)
        bbtn4.pack(anchor=S, pady=5)
        
        bbtn3 = Button(f4, width=15, text="Create mosaic", font=fontti, height=1)
        bbtn3.pack(anchor=S, pady=5)
        
        self.f5 = Frame(f2, highlightbackground="grey", highlightthickness=1)
        self.f5.pack(side=LEFT, fill="both", padx=1, pady=1, expand=True)

        self.tile_stats = Label(self.f5, textvariable=self.tile_stats_text, width=100, font=fontti2)
        self.tile_stats.pack(side=TOP, fill="x")

        self.f5_1 = Frame(self.f5, highlightbackground="grey", highlightthickness=1)
        self.f5_1.pack(side=BOTTOM, fill="x")

        'buttons dict'
        # nappi dictionary controller rajapinta
        self.napit = {
            "getfolder": bbtn1,
            "getpicture": bbtn2,
            "create": bbtn3,
            "preview":bbtn4
        }

        'variables dict'
        self.variables = {
            "imagepath":self.imagespath,
            "tileimage":self.tileimage,
            "blend_original":self.blend_original, 
            "close_to_clusters":self.closer_to_clusters,
            "npixel":self.pixel_n, 
            "misize":self.msize,
            "nclust":self.nc
        }

    def set_button_callback(self, btn_key, callback_funktio):
        nappi_objekti = self.napit[btn_key]
        nappi_objekti.config(command=callback_funktio)

    def read_image(self):
        img = Image.open(self.path)

    def set_preview(self):
        if self.tile_image_is_set:
            
            self.image = Image.open(self.tileimage.get())
            w, h = self.image.size
            width = int(w*self.pixel_n.get()*(self.msize.get()/100))
            height = int(h*self.pixel_n.get()*(self.msize.get()/100))

            self.filesize = arvioi_jpeg_koko_mb(width, height)

            text = "Estmated tile mosaic image size: {} Mb \n".format(self.filesize)
            text += f"Width: {width}, Height: {height}"

            self.tile_stats_text.set(text)
        else:
            return

        self.tile_preview()
        
    def tile_preview(self):
        global img
        
        path = self.imagespath.get()
        
        self.clear_frame(self.f5_1)
        
        ns = self.pixel_n.get()
        tilesize = (ns, ns)
        
        with visit_dir(path):
            filenames = list(os.scandir())
            filenames = [e.name for e in filenames if ext_check(e.name, EXTS)]
            np.random.shuffle(filenames)
            random_names = filenames[:4]
            random_name = random_names[0]
            #for random_name in random_names:
                
            if True:
                
                img = Image.open(random_name)
                img = img.resize(tilesize)
                img = img.resize((250,250))
                
                tkimage = ImageTk.PhotoImage(img)
        
                image_label = tk.Label(self.f5_1)
                image_label.config(image=tkimage)
                image_label.photo = img
                image_label.pack(fill='both',expand='yes')
                
                #break
        
    def set_images_folder(self):
        # model
        folder_name = get_foldername_dialog(self.root, initdir=self.imagespath)
        if os.path.exists(folder_name):
            self.imagespath.set(folder_name)
            #self.tile_preview()
        else:
            print("wrong path...",folder_name)
        
    def set_tile_image(self, exts=["jpg","png","bmp","tif"]):
        # model
        file_name = get_filename_dialog(self.root, initdir=self.imagespath)
        if ext_check(file_name, exts):
            self.tile_image_is_set = True
            self.tileimage.set(file_name)
    
    def clear_frame(self, frame):
        'clear given frame (tk.Frame)'
        for widget in frame.winfo_children():
            widget.destroy()
        