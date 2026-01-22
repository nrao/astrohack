import xarray as xr

import toolviper.utils.logger as logger

from astrohack.utils import (
    add_caller_and_version_to_dict_2,
    get_summary_header,
    get_property_string,
    get_data_content_string,
    get_method_list_string,
)


class AstrohackBaseFile:
    """Base Data class for astrohack.

    Data within an object of this class can be selected for further inspection, plotted or produce a report
    """

    def __init__(self, file: str):
        """Initialize an AstrohackBaseFile object.

        :param file: File to be linked to this object
        :type file: str

        :return: AstrohackBaseFile object
        :rtype: AstrohackBaseFile
        """
        self.file = file
        self._file_is_open = False
        self.root = None

    def __getitem__(self, key: str) -> xr.DataTree:
        """
        get item implementation that gets the xdtree at key.

        :param key: Key for which to fetch a subtree
        :type key: str

        :return: corresponding subtree
        :rtype: xr.DataTree
        """
        return self.root[key]

    def __setitem__(self, key: str, subtree: xr.DataTree) -> None:
        """
        Set item implementation that sets the xdtree at key.

        :param key: Key for which to set a subtree
        :type key: str

        :param subtree: Subtree to attach at key
        :type subtree: xr.DataTree

        :return: None
        :rtype: NoneType
        """
        self.root[key] = subtree
        return

    @property
    def is_open(self) -> bool:
        """
        Check whether the object has opened the corresponding hack file.

        :return: True if open, else False.
        :rtype: bool
        """
        return self._file_is_open

    def keys(self, *args, **kwargs):
        """
        Get children keys

        :param args: args to deliver to dict.keys() method
        :type args: list

        :param kwargs: Dict of keyword args to deliver to dict.keys() method
        :type kwargs: dict

        :return: dict keys iterable
        :rtype: dict_keys
        """
        return self.root.children.keys(*args, **kwargs)

    def items(self, *args, **kwargs):
        """
        Get children items

        :param args: args to deliver to dict.items() method
        :type args: list

        :param kwargs: Dict of keyword args to deliver to dict.items() method
        :type kwargs: dict

        :return: dict items iterable
        :rtype: dict_items
        """
        return self.root.children.items(*args, **kwargs)

    def values(self, *args, **kwargs):
        """
        Get children values

        :param args: args to deliver to dict.values() method
        :type args: list

        :param kwargs: Dict of keyword args to deliver to dict.values() method
        :type kwargs: dict

        :return: dict values iterable
        :rtype: dict_values
        """
        return self.root.children.values(*args, **kwargs)

    def open(self, file: str = None) -> bool:
        """
        Open Base file.

        :param file: File to be opened, if None defaults to the previously defined file
        :type file: str, optional

        :return: True if file is properly opened, else returns False
        :rtype: bool
        """

        if file is None:
            file = self.file

        try:
            # Chunks='auto' means lazy dask loading with automatic choice of chunk size
            # chunks=None is direct opening.
            self.root = xr.open_datatree(file, engine="zarr", chunks="auto")

            self._file_is_open = True
            self.file = file

        except Exception as error:
            logger.error(f"There was an exception opening the file: {error}")
            self._file_is_open = False

        return self._file_is_open

    def write(self, mode="w"):
        """
        Write mds to disk by saving the data tree to a file

        :param mode: File mode
        :type mode: str
        """
        self.root.to_zarr(self.file, mode=mode, consolidated=True)

    def summary(self) -> None:
        """
        Prints summary of the AstrohackBaseFile object, with available data, attributes and available methods

        :return: None
        :rtype: NoneType
        """
        outstr = get_summary_header(self.file)
        outstr += get_property_string(self.root.attrs)
        outstr += get_method_list_string(self)
        outstr += get_data_content_string(self.root)
        print(outstr)

    @classmethod
    def create_from_input_parameters(cls, file_name: str, input_parameters: dict):
        """
        Create an AstrohackBaseFile object from a filename and initializes xdtree root attributes.

        :param file_name: Name of the file in disk to be created
        :type file_name: str

        :param input_parameters: Input parameters for the calling function to be stored in root attributes.
        :type input_parameters: dict

        :return: Initiallized AstrohackBaseFile object
        :rtype: AstrohackBaseFile
        """
        data_obj = cls(file_name)
        data_obj.root = xr.DataTree(name="root")
        add_caller_and_version_to_dict_2(data_obj.root.attrs, direct_call=False)
        data_obj.root.attrs["input_parameters"] = input_parameters
        return data_obj

    def add_node_to_tree(self, new_node, dump_to_disk=True):
        """
        Add a node to root at a position determined by new_node's name

        :param new_node: Node to be included in root
        :type new_node: xarray.DataTree

        :param dump_to_disk: Dump root to disk to freeup RAM
        :type dump_to_disk: bool

        :return: None
        :rtype: NoneType
        """
        assert isinstance(new_node, xr.DataTree)
        lvls = new_node.name.split("-")
        n_lvls = len(lvls)
        if n_lvls == 1:
            lvl_0 = lvls[0]
            self.root.update({lvl_0: new_node})
        elif n_lvls == 2:
            lvl_0, lvl_1 = lvls
            if lvl_0 in self.keys():
                self[lvl_0].update({lvl_1: new_node})
            else:
                self[lvl_0] = xr.DataTree(name=lvl_0, children={lvl_1: new_node})
        elif n_lvls == 3:
            lvl_0, lvl_1, lvl_2 = lvls
            if lvl_0 in self.keys():
                if lvl_1 in self[lvl_0].keys():
                    self[lvl_0][lvl_1].update({lvl_2: new_node})
                else:
                    self[lvl_0][lvl_1] = xr.DataTree(
                        name=lvl_1, children={lvl_2: new_node}
                    )
            else:
                self[lvl_0] = xr.DataTree(
                    name=lvl_0,
                    children={
                        lvl_1: xr.DataTree(name=lvl_1, children={lvl_2: new_node})
                    },
                )
        else:
            raise NotImplementedError("Cannot handle a case of more than three levels")

        if dump_to_disk:
            self.write(mode="a")
            del self.root
            self.open()

        return
