# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 11:18:14 2021

@author: Mikko
"""

from tkinter import *
from tkinter import Frame, Canvas, Label, Button, Message, LEFT, RIGHT, BOTTOM, ALL, Tk, Entry, BOTH, S, HORIZONTAL
from tkinter import filedialog
from random import randint
import tkinter as tk
from tkinter import ttk

import time

from PIL import Image, ImageTk
import PIL

from sklearn.cluster import KMeans

import numpy as np
import os

def to_array(photo):
    I = np.asarray(photo)
    return(I)

def to_image(arr, path):
    im = PIL.Image.fromarray(np.uint8(arr))        
    #render = ImageTk.PhotoImage(im)
    im.save((path))
    
def row_to_image(row, check=True):
    l = len(row)/3
    l = int(l)
    if check:
        if not l==np.floor(l):
            raise ValueError("Column number has to be divisible by 3")
        if not np.sqrt(l)==np.floor(np.sqrt(l)):
            raise ValueError("Column number divided by 3 has to have a whole number squareroot, not {}".format(np.sqrt(l)))
    
    sl = np.sqrt(l).astype(int)   
        
    r = row[:l].reshape((sl,sl))
    g = row[l:(2*l)].reshape((sl,sl))
    b = row[2*l:].reshape((sl,sl))

    pic = np.array([color for color in (r,g,b)])
    pic = np.moveaxis(pic, 0, -1)
    
    return(pic)

def distance(vec1,vec2):
    return(np.sqrt(np.sum((vec1-vec2)**2)))
    
def get_filename_dialog(root, initdir="/"):
    root.filename =  filedialog.askopenfilename(initialdir=initdir,title="Select file",filetypes=(("jpeg files","*.jpg"),("all files","*.*")))
    return(root.filename)
    
def get_foldername_dialog(root, initdir="/"):
    root.foldername =  filedialog.askdirectory(initialdir=initdir,title="Select file")
    return(root.foldername)
    
def get_files_dir():
    files_path = [os.path.abspath(x) for x in os.listdir()]
    return(files_path)
    
def get_files_dir_path(path):
    files_path = [os.path.abspath(x) for x in os.listdir()]
    return(files_path)
    
def kmeans_model(arr,axis=2,n_clust=20):
    
    nx,ny = arr.shape[:2]

    vectors = arr.reshape(nx*ny,3)
    pkmeans = KMeans(n_clusters=n_clust, random_state=0).fit(vectors)
    return(pkmeans)
    

def array_stats_(arr):
    stats = []
    
    meanvec = [np.mean(arr[:,:,i]) for i in range(3)]
    stats.append(meanvec)
    
    return stats

def array_stats(arr):
    stats = []
    vec = [np.bincount(arr[:,:,i].flatten()).argmax() for i in range(3)]
    stats.append(vec)

    return(stats)
    
def towards_mean(arr):
    meancolor = np.array([np.mean(arr[:,:,i]) for i in range(3)])
    
    arr2 = arr.copy()
    arr2.setflags(write=1)
    arr2 = arr2.astype(float)
    
    for i in range(3):
        arr2 += meancolor[i]
        arr2 /= 2
        
    return(arr2)
    
class Gui1:
    
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

        self.imagespath = StringVar()
        cf = os.getcwd()
        filename = cf+"\\"
        self.imagespath.set(filename)
        
        self.tileimage = StringVar()
        self.tileimage.set("")
        
        self.blend_original = IntVar()
        self.closer_to_clusters = IntVar()
         
        "-- first tab and sub widgets"
        
        self.tabs = ttk.Notebook(root, width=self.fdx, height=self.fdy)
        
        f1 = Frame(self.tabs)
        self.tabs.add(f1, text="Create Mosaic Image")
        
        f2 = Frame(f1)
        f2.pack()
        
        self.filename = Label(f2, textvariable=self.imagespath, width=100)
        self.filename.pack(side=RIGHT)
        
        bbtn1 = Button(f2, width=15, text="Add Folder", command=self.add_folder_to_scheme)
        bbtn1.pack(side=LEFT)
        
        f3 = Frame(f1)
        f3.pack()
        
        self.filename = Label(f3, textvariable=self.tileimage, width=100)
        self.filename.pack(side=RIGHT)
        
        bbtn2 = Button(f3, width=15, text="Tile picture", command=self.add_file_to_scheme)
        bbtn2.pack(side=LEFT)
        
        f4 = Frame(f1)
        f4.pack(anchor=W)
        
        self.nc = tk.Scale(f4, label="Number of Clusters", orient=tk.HORIZONTAL, length=200)
        self.nc.pack(anchor=W, padx=10)
        self.nc.set(25)
        
        self.cb1 = Checkbutton(f4, text="Blend with original", variable=self.blend_original, padx=20)
        self.cb1.pack(anchor=W)
        
        self.cb2 = Checkbutton(f4, text="Closer to cluster centers", variable=self.closer_to_clusters)
        self.cb2.pack()
        
        bbtn3 = Button(f4, width=15, text="Create mosaic", command=self.create_mosaic)
        bbtn3.pack(anchor=W)
        
        self.tabs.grid(row=0, column=0)
        

    def add_folder_to_scheme(self):
        file_name = get_foldername_dialog(self.root, initdir=self.imagespath)
        self.imagespath.set(file_name)
        
    def add_file_to_scheme(self, exts=["jpg","png","bmp","tif"]):
        file_name = get_filename_dialog(self.root, initdir=self.imagespath)
        if file_name.split(".")[-1] in exts:
            self.tileimage.set(file_name)
            
    def mosaic_folder(self):
        pass
        
    def create_mosaic(self):
        t = Tiles()
        print("initializing images")
        path = self.imagespath.get()
        #pathends = ["one","two","three","four","five"]
        
        t.get_image_set(path)
            
        path = self.tileimage.get()   
        #path = "C://Users//Mikko//Documents//LiClipse Workspace//firebase//source2//data//xxx//meitsi.jpg"
        
        print("clustering images")
        t.tile_image(path)
        t.cluster_up(self.nc.get())
        
        print("mosaic creation bitches")
        
        t.create_mosaic_image(closer_to_clust=[False,True][self.closer_to_clusters.get()],
                              blend_original=[False,True][self.blend_original.get()],
                              blend_ratio=0.1)
        
        #path = os.path.abspath(__file__)
        path = os.getcwd()
        path = path+"//mosaic{}.jpg".format(int(time.time()))
        
        to_image(t.mosaic,path)
        print("done, check working directory")
        
class Tiles:

    def __init__(self, tilesize=(32,32)):
        
        self.arrays = []
        self.tilesize=tilesize
            
    def get_image_set(self, path):
        "folder iterating of self.get_pictures_from_folder"
        for e in os.scandir(path):
            if e.is_dir():
                self.get_pictures_from_folder(e.path,False)
                
    def get_pictures_from_array(self,array):
        "get mosaic image set from a np array color image set, for exmaple cifar-10"
        check = True
        for i in range(array.shape[0]):
            row = array[i,:]
            picarr = row_to_image(row, check=check)
            self.arrays.append(picarr)
            check = False
            
        n = len(self.arrays)
        self.n_arrays = n
    
        self.stats = np.zeros((n,6))
        self.stats[:,0] = [i for i in range(n)]
        self.set_means()
    
    def get_pictures_from_folder(self, path, clear_previous=True, exts=["jpg","jpeg","png","bmp","tif"]):
        "obtains mosaic images from a given folder"
        
        if clear_previous:
            del self.arrays
            self.arrays = []
        
        paths = [os.path.abspath(x.path) for x in os.scandir(path) if x.path.split(".")[-1] in exts]
        for path in paths:
            arr = Image.open(path).resize(self.tilesize)
            arr = to_array(arr)
            if arr.shape[2]==3:
                self.arrays.append(arr)    
    
        n = len(self.arrays)
        self.n_arrays = n
    
        self.stats = np.zeros((n,6))
        self.stats[:,0] = [i for i in range(n)]
        self.set_means()
        
    def get_picture(self, get_filename=True):
        "appends a single picture to mosaic images from path"
        if get_filename:
            photo_name = get_filename_dialog(self.root)
        else:
            photo_name = os.getcwd()+"\\"
            photo_name += self.folder_picture_names[self.picture_index]

        self.pictures.append(Image.open(photo_name))
    
    def set_means(self):
        "mosaic image array set statistics... used in tiling"
        i = 0
        for arr in self.arrays:
            self.stats[i,1:4] = array_stats(arr)[0]
            i += 1
            
    def set_clusters(self):
        "mosaic images assigned to clusters of the model"
        i = 0
        for arr in self.arrays:
            pred = self.km.predict([self.stats[i,1:4]])[0]
            self.stats[i,-2] = pred 
            self.stats[i,-1] = distance(self.km.cluster_centers_[pred,:],self.stats[i,1:4])
            i += 1
            
    def tile_image(self, path, resize=False, resize_ratio=0.50):
        "the image to be tiled is set"
        ti = Image.open(path)
        if resize:
            rx = int(resize_ratio*ti.size[0])
            ry = int(resize_ratio*ti.size[1])
            self.tile_image.resize((rx,ry))
        
        self.tile_image = ti
        
        self.tile_array = to_array(self.tile_image)
        self.shp = self.tile_array.shape
        
    def cluster_up(self, n_clusters=25):
        "tiles are assigned to self.cluster_array with clustering model predictions"
        
        self.km = kmeans_model(self.tile_array, n_clust=n_clusters)
        
        self.cluster_array = np.zeros((self.shp[0],self.shp[1]))
        
        for i in range(self.shp[0]):
            for j in range(self.shp[1]):
                self.cluster_array[i,j] = self.km.predict([self.tile_array[i,j,:]])[0]
                
        self.cluster_array = self.cluster_array.astype(int)
        self.set_clusters()
        
        
    def closer_to_clusters(self, i, ratio):
        "handicap #1 of the final product - tile array moved closed to cluster center"
        
        arr = self.arrays[i].copy()
        arr = arr.astype(float)
        center = self.km.cluster_centers_[int(self.stats[i,-2]),:]
        for z in range(3):
            arr[:,:,z]+=center[z]*ratio
            arr[:,:,z]/=2
        arr = arr.astype(int)
        return(arr)
        
    def create_mosaic_image(self, mode="distance", closer_to_clust=True, blend_original=True, blend_ratio=1.0):
        "mosaic image creation from tile images with cluster assignments"
        
        tilesize = self.tilesize
        self.mosaic = np.zeros((self.shp[0]*tilesize[0],self.shp[1]*tilesize[1],3))
        
        assignments = np.zeros(self.cluster_array.shape)
        ni,nj = self.cluster_array.shape
        
        br = max(0.0,blend_ratio)
        br = min(1.0,blend_ratio)
        
        for i in range(ni):
            for j in range(nj):
                assignments[i,j] = self.assign_tile(i,j,mode=mode)
        assignments = assignments.astype(int)
        
        ti, tj = self.tilesize
        
        for i in range(ni):
            for j in range(nj):
                im = int(ti*i)
                jm = int(tj*j)
                
                if closer_to_clust:
                    r = np.random.uniform(np.random.uniform(),1)
                    arr = self.closer_to_clusters(assignments[i,j],r)
                    #arr = towards_mean(arr)
                else:
                    arr  = self.arrays[assignments[i,j]]
                
                if blend_original:
                    "handicap #2: tile colors are moved closer to tile image pixels"
                    arr2 = arr.copy()
                    arr2.setflags(1)
                    arr2 = arr2.astype(float)
                    
                    orig = [self.tile_array[i,j,c].astype(float) for c in range(3)]
                    for c in range(3):
                        arr2[:,:,c] += orig[c]*br
                        arr2[:,:,c] /= (1+br)
            
                    self.mosaic[im:(im+ti),jm:(jm+tj),:] = arr2.astype(int)
                else:
                    self.mosaic[im:(im+ti),jm:(jm+tj),:] = arr
        
    def assign_tile(self, i, j, mode):
        """mosaic tile is given an index representing a single tile from self.arrays"
        mode is deals with how tiles are picked from clusters"""
        
         #global rows
         #global r
        
        w = np.where(self.stats[:,4]==self.cluster_array[i,j])[0]
        rows = self.stats[w,:]
         
        if mode=="random":
            if not rows.shape[0]==0:
                r = np.random.randint(0,rows.shape[0])   
                irow = rows[r,0]
            else:
                return(0)
            return(irow)
         
        if not rows.shape[0]==0:

            rows = rows[np.argsort(rows[:, -1])]  
            r = np.random.triangular(0,0,rows.shape[0])
            r = np.round(r)
            r = int(r)
            r = min(r, rows.shape[0]-1)
            irow = rows[r,0]
        else:
            return(0)
        return(irow)
             
    def tile_stats(self):
        pass

def init_gui():   
    w = Tk()
    lx,ly = 1000,500
    size = "{}x{}".format(lx,ly)
    w.geometry(size)
    w.resizable(1, 1) 
    g = Gui1(w)
    w.mainloop()
    
init_gui()


           























    
