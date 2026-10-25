   
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

from .model import visit_dir
from .model import ext_check
from .model import EXTS
from .model import Tiles
from .model import to_image
from .model import timestamp
from .model import get_filename_dialog
from .model import get_foldername_dialog

def get_filename_dialog(root, initdir="/"):
    root.filename =  filedialog.askopenfilename(initialdir=initdir,title="Select file",filetypes=(("jpeg files","*.jpg"),("all files","*.*")))
    return root.filename
    
def get_foldername_dialog(root, initdir="/"):
    root.foldername =  filedialog.askdirectory(initialdir=initdir,title="Select file")
    return root.foldername
    
class Gui:
    
    def __init__(self, root):
        self.root = root
        self.menubar = Menu(self.root)
        
        self.windowsize = root.size()
        self.fdx,self.fdy = self.windowsize
        
        "-- option menu widgets"
        
        self.menubar = Menu(self.root)
        self.filemenu = Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=self.filemenu)
        
        self.filemenu.add_command(label ='New File', command = None) 
        self.filemenu.add_command(label ='Open...', command = None) 
        self.filemenu.add_command(label ='Save', command = None) 
        self.filemenu.add_separator() 
        self.filemenu.add_command(label ='Exit', command = root.destroy) 
        
        self.root.config(menu = self.menubar)

        "-- gui variables "
        
        fontti = font.Font(size=15)
    
        self.imagespath = StringVar()
        cf = os.getcwd()
        filename = cf+"\\"
        self.imagespath.set(filename)
        
        self.tileimage = StringVar()
        self.tileimage.set("")
        
        self.blend_original = IntVar()
        self.closer_to_clusters = IntVar()
         
        "-- first tab and sub widgets"
        
        #self.tabs = ttk.Notebook(root, width=self.fdx, height=self.fdy)
        self.main = Frame(root, highlightbackground="grey", highlightthickness=1)
        self.main.pack(fill="both")
        
        f1 = Frame(self.main, highlightbackground="grey", highlightthickness=1)
        #self.tabs.add(f1, text="Create Mosaic Image")
        f1.pack()
        
        #f2 = Frame(self.main, highlightbackground="grey", highlightthickness=1)
        #f2.pack()
        
        f3 = Frame(f1, highlightbackground="grey", highlightthickness=1)
        f3.pack()
        
        f3_1 = Frame(f3)
        f3_1.pack(side=TOP)
        
        self.foldername = Label(f3_1, textvariable=self.imagespath, width=100)
        self.foldername.pack(side=RIGHT)

        bbtn1 = Button(f3_1, width=20, text="Set picture tiles path...", command=self.set_images_folder)
        bbtn1.pack(side=LEFT)
        
        f3_2 = Frame(f3)
        f3_2.pack(side=BOTTOM)
    
        self.filename = Label(f3_2, textvariable=self.tileimage, width=100)
        self.filename.pack(side=RIGHT)
        
        bbtn2 = Button(f3_2, width=20, text="Get mosaic picture", command=self.set_tile_image)
        bbtn2.pack(side=LEFT)
        
        f4 = Frame(f1, highlightbackground="grey", highlightthickness=1)
        f4.pack(side=LEFT)

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
        
        bbtn3 = Button(f4, width=15, text="Create mosaic", font=fontti, command=self.create_mosaic, height=1)
        bbtn3.pack(anchor=W)
        
        self.f5 = Frame(f1, highlightbackground="grey", highlightthickness=1)
        self.f5.pack(side=RIGHT, fill="both")

        'buttons dict'
        # nappi dictionary controller rajapinta
        self.napit = {
            "getfolder": bbtn1,
            "getpicture": bbtn2,
            "create": bbtn3
        }

        'variables dict'
        self.variables = {
            "imagepath":self.imagespath,
            "tileimage":self.tileimage,
            "blend_original":self.blend_original, 
            "close_to_clusters":self.closer_to_clusters,
            "npixel":self.pixel_n
        }

    def set_button_callback(self, btn_key, callback_funktio):
        nappi_objekti = self.napit[btn_key]
        nappi_objekti.config(command=callback_funktio)
        
    def tile_preview(self):
        global img
        
        path = self.imagespath.get()
        
        self.clear_frame(self.f5)
        
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
                img = img.resize((100,100))
                
                tkimage = ImageTk.PhotoImage(img)
        
                image_label = tk.Label(self.f5)
                image_label.config(image=tkimage)
                image_label.photo = img
                image_label.pack(fill='both',expand='yes')
                
                #break
        
    def set_images_folder(self):
        # model
        folder_name = get_foldername_dialog(self.root, initdir=self.imagespath)
        if os.path.exists(folder_name):
            self.imagespath.set(folder_name)
            self.tile_preview()
        else:
            print("wrong path...",folder_name)
        
    def set_tile_image(self, exts=["jpg","png","bmp","tif"]):
        # model
        file_name = get_filename_dialog(self.root, initdir=self.imagespath)
        if ext_check(file_name, exts):
            self.tileimage.set(file_name)
    
    def clear_frame(self, frame):
        'clear given frame (tk.Frame)'
        for widget in frame.winfo_children():
            widget.destroy()
        