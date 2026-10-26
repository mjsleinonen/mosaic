from sklearn.cluster import KMeans

import numpy as np
import os

#from tkinter import filedialog

import contextlib
from pathlib import Path

from PIL import Image, ImageTk
import PIL

import time

@contextlib.contextmanager
def visit_dir(dirname):
    curdir = os.getcwd()
    
    os.chdir(dirname)
    try:
        yield
    finally:
        os.chdir(curdir)

def to_array(photo):
    I = np.asarray(photo)
    return(I)

def to_image(arr, path):
    "saved an numpy array as an image..."
    im = PIL.Image.fromarray(np.uint8(arr))        
    #render = ImageTk.PhotoImage(im)
    im.save((path))

def kmeans_model(arr,axis=2,n_clust=20):
    "returns a k-means clustering model, with number of clusters as an argument"
    nx,ny = arr.shape[:2]

    vectors = arr.reshape(nx*ny, 3)
    pkmeans = KMeans(n_clusters=n_clust, random_state=0).fit(vectors)
    return pkmeans

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
    
    return pic

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
    
    return pic

def distance(vec1,vec2):
    return(np.sqrt(np.sum((vec1-vec2)**2)))
    
def get_files_dir():
    files_path = [os.path.abspath(x) for x in os.listdir()]
    return files_path
    
def get_files_dir_path(path):
    files_path = [os.path.abspath(x) for x in os.listdir()]
    return files_path

def array_stats(arr):
    stats = []
    vec = [np.bincount(arr[:,:,i].flatten()).argmax() for i in range(3)]
    stats.append(vec)

    return stats
    
def towards_mean(arr):
    meancolor = np.array([np.mean(arr[:,:,i]) for i in range(3)])
    
    arr2 = arr.copy()
    arr2.setflags(write=1)
    arr2 = arr2.astype(float)
    
    for i in range(3):
        arr2 += meancolor[i]
        arr2 /= 2
        
    return arr2

def ext_check(name, exts):
    return name.split(".")[-1] in exts

def timestamp():
    return int(time.time())

EXTS = ["jpg","jpeg","png","bmp","tif"]

class Tiles:
    def __init__(self):
        
        self.arrays = []

        "variables"
        
        self.imagespath = ""
        self.tileimage = ""
        self.pixel_n = 32
        self.tilesize=(self.pixel_n, self.pixel_n)
        self.msize = 0
        self.closer_to_clusters = True
        self.blend_original = True
        self.n_clusters = 25

        self.variables = {
            "imagespath":self.imagespath,
        }

    def set_tile_size(self):
        self.tilesize = (self.pixel_n, self.pixel_n)
        
    def get_image_set(self, path):
        "folder iterating of self.get_pictures_from_folder"
        self.get_pictures_from_folder(path, False)

    def get_pictures_from_array(self, array):
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
    
    def get_pictures_from_folder(self, path, clear_previous=True, exts=EXTS):
        "obtains mosaic images from a given folder"
        
        if clear_previous:
            del self.arrays
            self.arrays = []
        
        paths = [os.path.abspath(x.path) for x in os.scandir(path) if ext_check(x.name, EXTS)]
        
        for i,path in enumerate(paths):
            try:
                img = Image.open(path).resize(self.tilesize)
            except Exception as exp:
                print(exp)
                continue

            if i%1000==0:
                print(i/len(paths)," ", len(paths)," ", i)

            arr = to_array(img)
            
            if arr.shape[2]==3:
                self.arrays.append(arr)    
    
        n = len(self.arrays)
        self.n_arrays = n
    
        self.stats = np.zeros((n,6))
        self.stats[:,0] = [i for i in range(n)]
        self.set_means()
        
        print("ready...")
        
    def get_picture(self, photo_name, get_filename=True, resize=1.0):
        "appends a single picture to mosaic images from path"
        img = Image.open(photo_name)

        self.pictures.append(img)
    
    def set_means(self):
        "mosaic image array set statistics... used in tiling"
        i = 0
        for arr in self.arrays:
            self.stats[i, 1:4] = array_stats(arr)[0]
            i += 1
            
    def set_clusters(self):
        "mosaic images assigned to clusters of the model"
        i = 0
        for arr in self.arrays:
            pred = self.km.predict([self.stats[i,1:4]])[0]
            self.stats[i,-2] = pred 
            self.stats[i,-1] = distance(self.km.cluster_centers_[pred,:],self.stats[i,1:4])
            i += 1
            
    def get_tile_image(self, path, resize_ratio=1.0):
        "the image to be tiled is set"

        ti = Image.open(Path(path))
        
        if resize_ratio!=1.0:
            rx = int(resize_ratio*ti.size[0])
            ry = int(resize_ratio*ti.size[1])
            ti = ti.resize((rx,ry))
        
        self.tile_image = ti
        
        print("image size: ", self.tile_image.size)
        
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
        
    def move_closer_to_clusters(self, i, ratio):
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
                    arr = self.move_closer_to_clusters(assignments[i,j],r)
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

    ########

    def create_mosaic(self):
        try:
            print("initializing images...")

            path = self.imagespath
            self.get_image_set(path)
            
            print("clustering images...")
            
            rsratio = self.msize/100
            self.get_tile_image(self.tileimage, resize_ratio=rsratio)

            self.cluster_up(self.n_clusters)
            
            print("mosaic creation....")
            
            self.create_mosaic_image(
                                closer_to_clust=[False,True][self.closer_to_clusters],
                                blend_original=[False,True][self.blend_original],
                                blend_ratio=0.1
            )
            path = Path(os.getcwd())/Path(f"mosaic_{timestamp()}.jpg")
            to_image(self.mosaic, path)

            print("mosaic done... {}".format(path))
        except Exception as exp:
            raise exp
            pass