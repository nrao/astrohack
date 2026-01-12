import shutil

from toolviper.utils import data


class Testlocit:
    data_folder = "locit_data"
    destination_folder = "locit_exports"
    ref_products_folder = f"{data_folder}/ref_locit_products"

    silly_name = "Anything"
    remote_locit_name = "kband_locit_small.locit.zarr"

    @classmethod
    def setup_class(cls):
        """setup any state specific to the execution of the given test class
        such as fetching test data"""
        data.download(file=cls.remote_locit_name, folder=cls.data_folder)
        data.download(file="ref_locit_products", folder=cls.data_folder)

        # Add datafolder to names for execution
        for varname, varvalue in cls.__dict__.items():
            if isinstance(varvalue, str):
                if varname.split("_")[-1] == "name":
                    setattr(cls, varname, f"{cls.data_folder}/{varvalue}")

    @classmethod
    def teardown_class(cls):
        """teardown any state that was previously setup with a call to setup_class
        such as deleting test data"""
        shutil.rmtree(cls.data_folder, ignore_errors=True)
        shutil.rmtree(cls.destination_folder, ignore_errors=True)
        return
