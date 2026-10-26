#import imagetiler_lumped

#from imagetiler_lumped import *
#imagetiler_lumped.init_gui()

if __name__ == "__main__":
    from app.view_gui import Gui
    from app.model import Tiles

    from app.controller import ImageTileController

    app_model = Tiles()
    app_view = Gui()

    app_controller = ImageTileController(app_model, app_view)

    app_controller.run()
