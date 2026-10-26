import pytest
import unittest

import sys
from pathlib import Path

from app.view_gui import Gui
from app.model import Tiles

# Import ylempää
nykyinen_kansio = Path(__file__).resolve().parent
ylempi_kansio = nykyinen_kansio.parent
sys.path.insert(0, str(ylempi_kansio))

class TestMosaicSovellus(unittest.TestCase):

    def setUp(self):
        """Suoritetaan AUTOMAATTISESTI ennen JOKAISTA testifunktiota."""
        # 1. View instannsi

        self.view = Gui()
        self.model = Tiles()
        
    def tearDown(self):
        """Suoritetaan AUTOMAATTISESTI jokaisen testifunktion jälkeen."""
        # Tyhjennetään tietokanta ja poistetaan konteksti, jotta seuraava testi alkaa puhtaalta pöydältä

        self.view.root.destroy()

    def test_initial_model_state(self):
        """Testataan, ."""
        self.assertEqual(self.model.pixel_n, 32)


if __name__ == '__main__':
    unittest.main()