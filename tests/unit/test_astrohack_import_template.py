class TestAstrohack:
    @classmethod
    def setup_class(cls):
        """setup any state specific to the execution of the given test class
        such as fetching test data"""
        pass

    @classmethod
    def teardown_class(cls):
        """teardown any state that was previously setup with a call to setup_class
        such as deleting test data"""
        pass

    def test_import_extract_holog(self):
        try:
            from astrohack.extract_holog import extract_holog
        except ImportError:
            assert False

    def test_import_holog(self):
        try:
            from astrohack.holog import holog
        except ImportError:
            assert False

    def test_import_panel(self):
        try:
            from astrohack.panel import panel
        except ImportError:
            assert False

    def test_import_dio_open_holog(self):
        try:
            from astrohack.io.dio import open_holog
        except ImportError:
            assert False

    def test_import_dio_open_image(self):
        try:
            from astrohack.io.dio import open_image
        except ImportError:
            assert False

    def test_import_dio_open_panel(self):
        try:
            from astrohack.io.dio import open_panel
        except ImportError:
            assert False

    def test_import_dio_open_pointing(self):
        try:
            from astrohack.io.dio import open_pointing
        except ImportError:
            assert False
