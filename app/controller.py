

class ImageTileController:
    def __init__(self, model, view):
        self.model = model
        self.view = view

        self.TILE_IMAGE_KEY = "tileimage"
        self.IMAGE_PATH_KEY = "imagepath"

        # connect callbacks
        self.set_up_commands()

    def set_up_commands(self):
        for napin_nimi, nappi_objekti in self.view.napit.items():
            # lambda_default-kikka (nimi=napin_nimi) tarvitaan, 
            # jotta Python muistaa oikean nimen silmukan pyöriessä
            nappi_objekti.config(command=lambda nimi=napin_nimi: self.handle_callback(nimi))

    def handle_callback(self, napin_nimi):
        print(f"Käyttäjä painoi nappia: {napin_nimi}")
        self.sync_variables()
        
        # Tässä voit kutsua Modelin metodeja napin nimen mukaan
        if napin_nimi == "getfolder":
            # self.model.tallenna()
            pass
        elif napin_nimi == "getpicture":
            pass
        elif napin_nimi ==" create":
            self.model.create_mosaic()
        else:
            pass

    def get_tile_image(self):
        tile_image_path = self.view.variables[self.TILE_IMAGE_KEY]
        self.model.get_tile_image(tile_image_path)

    def sync_variables(self):
        self.model.imagespath = self.view.imagespath.get()
        self.model.tileimage = self.view.tileimage.get()
        self.model.pixel_n = self.view.pixel_n.get()
        self.model.tilesize=(self.pixel_n, self.pixel_n)
        self.model.msize = self.view.msize.get()
        self.model.closer_to_clusters = self.view.closer_to_clusters.get()
        self.model.blend_original = self.view.blend_original

    def run(self):
        self.view.mainloop()


if __name__ == "__main__":
    from .view_gui import Gui
    from .model import Tiles

    app_model = Tiles()
    app_view = Gui()

    app_controller = ImageTileController(app_model, app_view)
    
    app_controller.run()