
import contextlib
import os

@contextlib.contextmanager
def visit_dir(dirname):
    curdir = os.getcwd()
    
    os.chdir(dirname)
    try:
        yield
    finally:
        os.chdir(curdir)

class ImageTileController:
    "main controller class..."
    def __init__(self, model, view):

        self.model = model
        self.view = view

        self.TILE_IMAGE_KEY = "tileimage"
        self.IMAGE_PATH_KEY = "imagepath"
        self.SET_UP_COMMANDS_KEYS = ["create"]

        # connect callbacks
        self.set_up_commands()

    def set_up_commands(self):
        for napin_nimi, nappi_objekti in self.view.napit.items():
            # lambda_default-kikka (nimi=napin_nimi) tarvitaan, 
            # jotta Python muistaa oikean nimen silmukan pyöriessä
            if napin_nimi in self.SET_UP_COMMANDS_KEYS:
                nappi_objekti.config(command=lambda nimi=napin_nimi: self.handle_callback(nimi))

    def handle_callback(self, napin_nimi):
        print(f"Käyttäjä painoi nappia: {napin_nimi}")
        self.sync_variables()
        
        # Tässä voit kutsua Modelin metodeja napin nimen mukaan
        if napin_nimi == "getfolder":
            pass
        elif napin_nimi == "getpicture":
            pass
        elif napin_nimi == "create":
            self.model.create_mosaic()
        else:
            pass

    def get_tile_image(self):
        "tile image to model command"

        tile_image_path = self.view.variables[self.TILE_IMAGE_KEY].get()
        self.model.get_tile_image(tile_image_path)

    def sync_variables(self):
        "get variables from view to model"

        self.model.imagespath = self.view.imagespath.get()
        self.model.tileimage = self.view.tileimage.get()
        npixel = self.view.pixel_n.get()
        self.model.pixel_n = npixel
        self.model.tilesize=(npixel, npixel)
        self.model.msize = self.view.msize.get()
        self.model.closer_to_clusters = self.view.closer_to_clusters.get()
        self.model.blend_original = self.view.blend_original

    def run(self):
        self.view.root.mainloop()

if __name__ == "__main__":
    from .view_gui import Gui
    from .model import Tiles

    app_model = Tiles()
    app_view = Gui()

    app_controller = ImageTileController(app_model, app_view)

    app_controller.run()

