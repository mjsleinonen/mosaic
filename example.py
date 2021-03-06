from imagetiler import *

t = Tiles()
print("initializing images")
#pathends = ["one","two","three","four","five"]
        
t.get_pictures_from_array(dat[:1800,:])
            
path_to_tile_image = "..."
        
print("clustering images")
t.tile_image(path_to_tile_image)
t.cluster_up(20) #
        
print("mosaic creation...")
        
t.create_mosaic_image(closer_to_clust=True,
                              blend_original=True,
                              blend_ratio=0.30)

import os
path = os.path.abspath(a_module.__file__)

path = path+meitsimosaic{}.jpg".format(int(time.time()))
        
to_image(t.mosaic,path)
