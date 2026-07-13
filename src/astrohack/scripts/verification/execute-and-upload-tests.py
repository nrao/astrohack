from astrohack.utils.package_info import get_astrohack_path
import os
import shutil
import pytest
import subprocess

from astrohack.utils.text import format_duration


def set_tests_to_update():
    test_dict = {
        # "unit/mdses/test_beamcut_mds.py": {
        #     "n_items": 1,
        #     "item_0": {
        #         "creation": "beamcut_exports",
        #         "destiny": "ref_beamcut_products",
        #         "description": "beamcut reference export products",
        #         "type": "Beam cut",
        #         "telescope": "VLA",
        #         "update_manifest": False,
        #     },
        #     "cleanup_names": ["beamcut_data"],
        # },
        # "unit/mdses/test_image_mds.py": {
        #     "n_items": 1,
        #     "item_0": {
        #         "creation": "image_exports",
        #         "destiny": "ref_image_products",
        #         "description": "image export products",
        #         "type": "Holography",
        #         "telescope": "VLA",
        #         "update_manifest": False,
        #     },
        #     "cleanup_names": ["image_data"],
        # },
        # "unit/mdses/test_locit_mds.py": {
        #     "n_items": 1,
        #     "item_0": {
        #         "creation": "locit_exports",
        #         "destiny": "ref_locit_products",
        #         "description": "locit reference export products",
        #         "type": "Antenna position corrections",
        #         "telescope": "VLA",
        #         "update_manifest": False,
        #     },
        #     "cleanup_names": ["locit_data"],
        # },
        # "unit/mdses/test_panel_mds.py": {
        #     "n_items": 1,
        #     "item_0": {
        #         "creation": "panel_exports",
        #         "destiny": "ref_panel_products",
        #         "description": "panel export products",
        #         "type": "Holography",
        #         "telescope": "VLA",
        #         "update_manifest": False,
        #     },
        #     "cleanup_names": ["panel_data"],
        # },
        # "unit/mdses/test_position_mds.py": {
        #     "n_items": 1,
        #     "item_0": {
        #         "creation": "position_exports",
        #         "destiny": "ref_position_products",
        #         "description": "position export products",
        #         "type": "Antenna position corrections",
        #         "telescope": "VLA",
        #         "update_manifest": False,
        #     },
        #     "cleanup_names": ["position_data"],
        # },
        # "unit/user_facing_functions/test_beamcut.py": {
        #     "n_items": 1,
        #     "item_0": {
        #         "creation": "beamcut_data/kband_beamcut_small_local.beamcut.zarr",
        #         "destiny": "kband_beamcut_small.beamcut.zarr",
        #         "description": "beamcut reference Astrohack beamcut file",
        #         "type": "Beam cut",
        #         "telescope": "VLA",
        #         "update_manifest": False,
        #     },
        #     "cleanup_names": ["beamcut_data"],
        # },
        "unit/user_facing_functions/test_extract_holog.py": {
            "n_items": 1,
            "item_0": {
                "creation": "ext_holog_data/ea25_cal_small_before_fixed.split.holog.zarr",
                "destiny": "ea25_cal_small_before_reference.holog.zarr",
                "description": "Reference Astrohack holog file",
                "type": "Holography",
                "telescope": "VLA",
                "update_manifest": False,
            },
            "cleanup_names": ["ext_holog_data"],
        },
    }

    return test_dict


def execute_and_upload(test_file, test_params):
    subproc_exec_list = [
        "python",
        get_astrohack_path() / "scripts" / "verification" / "file_uploader.py",
    ]
    pytest.main([test_file])
    n_items = test_params["n_items"]
    for i_item in range(n_items):
        item_pars = test_params[f"item_{i_item}"]
        shutil.rmtree(item_pars["destiny"], ignore_errors=True)
        shutil.move(item_pars["creation"], item_pars["destiny"])
        subproc_exec_list.extend(
            [
                item_pars["destiny"],
                "-t",
                item_pars["telescope"],
                "-m",
                item_pars["type"],
                "-d",
                item_pars["description"],
            ]
        )
        if item_pars["update_manifest"]:
            subproc_exec_list.append("-u")
        subprocess.call(subproc_exec_list)
        shutil.rmtree(item_pars["destiny"], ignore_errors=True)
        if os.path.exists(item_pars["destiny"] + ".zip"):
            os.remove(item_pars["destiny"] + ".zip")

    for name in test_params["cleanup_names"]:
        shutil.rmtree(name, ignore_errors=True)


def main():
    import time

    wait_time = 3600

    distro_path = get_astrohack_path()
    os.chdir(distro_path / "../../tests")

    os.environ["SKIP_PYTEST_CLEANUP"] = "True"
    os.environ["PRODUCE_REFERENCE_PRODUCTS"] = "True"
    test_dict = set_tests_to_update()
    for test_file, test_params in test_dict.items():
        execute_and_upload(test_file, test_params)
        if len(test_dict.keys()) > 1:
            print(
                f"{test_file} done, waiting {format_duration(wait_time)} for cloudflare sync before executing next test."
            )
            time.sleep(wait_time)
    os.environ["SKIP_PYTEST_CLEANUP"] = "False"
    os.environ["PRODUCE_REFERENCE_PRODUCTS"] = "False"


main()
