from astrohack.utils.package_info import get_astrohack_path
import os
import shutil
import glob
import pytest
import subprocess


def set_tests_to_update():
    test_dict = {
        # "test_beamcut_mds.py": {
        #     "created_names": [],
        #     "destiny_names": [],
        # },
        "unit/mdses/test_image_mds.py": {
            "n_items": 1,
            "item_0": {
                "creation": "image_exports",
                "destiny": "image_data/ref_image_products",
                "description": "image export products",
                "type": "Holography",
                "telescope": "VLA",
            },
            "cleanup_names": ["image_data"],
        },
        # "test_locit_mds.py": 0,
        # "test_panel_mds.py": 0,
        # "test_position_mds.py": 0,
        # "test_beamcut.py": 0,
    }

    return test_dict


def execute_and_upload(test_file, test_params):
    uploader_exec = get_astrohack_path() / "scripts" / "file_uploader.py"
    pytest.main([test_file])
    print("Waited for pytest?")
    n_items = test_params["n_items"]
    print(glob.glob("*"))
    for i_item in range(n_items):
        item_pars = test_params[f"item_{i_item}"]
        shutil.rmtree(item_pars["destiny"])
        shutil.move(item_pars["creation"], item_pars["destiny"])
        # subprocess.call(
        #     [
        #         "python",
        #         uploader_exec,
        #         item_pars["destiny"],
        #         "-t",
        #         item_pars["telescope"],
        #         "-m",
        #         item_pars["type"],
        #         "-d",
        #         item_pars["description"],
        #     ]
        # )

    for name in test_params["cleanup_names"]:
        shutil.rmtree(name)


def main():
    distro_path = get_astrohack_path()
    os.chdir(distro_path / "../../tests")

    os.environ["SKIP_PYTEST_CLEANUP"] = "True"

    test_dict = set_tests_to_update()
    for test_file, test_params in test_dict.items():
        execute_and_upload(test_file, test_params)

    os.environ["SKIP_PYTEST_CLEANUP"] = "False"


main()
