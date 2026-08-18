import importlib


def _test_import(object_name):
    astrohack = importlib.import_module("astrohack")
    try:
        object_ = getattr(astrohack, object_name)
        del object_
    except AttributeError:
        assert False, f"Could not import {object_name} from astrohack"


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

    def test_import_holography_functions(self):
        holog_functions = [
            "extract_pointing",
            "extract_holog",
            "holog",
            "combine",
            "panel",
        ]
        for function in holog_functions:
            _test_import(function)

    def test_import_holography_io_functions(self):
        holog_io_functions = ["open_holog", "open_image", "open_panel", "open_pointing"]
        for function in holog_io_functions:
            _test_import(function)

    def test_import_holography_classes(self):
        holog_classes = [
            "AstrohackPointFile",
            "AstrohackImageFile",
            "AstrohackPanelFile",
            "AstrohackHologFile",
        ]
        for class_name in holog_classes:
            _test_import(class_name)

    def test_import_baseline_functions(self):
        baseline_functions = ["extract_locit", "locit"]
        for function in baseline_functions:
            _test_import(function)

    def test_import_baseline_io_functions(self):
        baseline_io_functions = ["open_locit", "open_position"]
        for function in baseline_io_functions:
            _test_import(function)

    def test_import_baseline_classes(self):
        baseline_classes = ["AstrohackLocitFile", "AstrohackPositionFile"]
        for class_name in baseline_classes:
            _test_import(class_name)

    def test_import_beamcut(self):
        beamcut_objects = [
            "AstrohackBeamcutFile",
            "open_beamcut",
            "beamcut",
        ]
        for object_name in beamcut_objects:
            _test_import(object_name)
