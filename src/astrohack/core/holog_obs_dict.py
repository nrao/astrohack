import json
import pathlib
import copy
import numpy as np

import toolviper.utils.logger as logger

from typing import Union, List, NewType, Dict, Any

from astrohack.io.dio import inspect_holog_obs_dict

JSON = NewType("JSON", Dict[str, Any])
KWARGS = NewType("KWARGS", Union[Dict[str, str], Dict[str, int]])


class HologObsDict(dict):
    """
    ddi --> map --> ant, scan
                    |
                    o--> map: [reference, ...]
    """

    def __init__(self, obj: JSON = None):
        if obj is None:
            super().__init__()
        else:
            super().__init__(obj)

    def __getitem__(self, key: str):
        return super().__getitem__(key)

    def __setitem__(self, key: str, value: Any):
        return super().__setitem__(key, value)

    @classmethod
    def from_file(cls, filepath):
        if filepath.endswith(".holog.zarr"):
            filepath = str(
                pathlib.Path(filepath).resolve().joinpath("holog_obs_dict.json")
            )

        try:
            with open(filepath, "r") as file:
                obj = json.load(file)

                return HologObsDict(obj)

        except FileNotFoundError:
            logger.error(f"File {filepath} not found")

    def print(self, style: str = "static"):
        if style == "dynamic":
            return inspect_holog_obs_dict(self, style="dynamic")

        else:
            return inspect_holog_obs_dict(self, style="static")

    def select(
        self, key: str, value: any, inplace: bool = False, **kwargs: KWARGS
    ) -> object:

        if inplace:
            obs_dict = self

        else:
            obs_dict = HologObsDict(copy.deepcopy(self))

        if key == "ddi":
            return self._select_ddi(value, obs_dict=obs_dict)

        elif key == "map":
            return self._select_map(value, obs_dict=obs_dict)

        elif key == "antenna":
            return self._select_antenna(value, obs_dict=obs_dict)

        elif key == "scan":
            return self._select_scan(value, obs_dict=obs_dict)

        elif key == "baseline":
            if "reference" in kwargs.keys():
                return self._select_baseline(
                    value,
                    n_baselines=None,
                    reference=kwargs["reference"],
                    obs_dict=obs_dict,
                )

            elif "n_baselines" in kwargs.keys():
                return self._select_baseline(
                    value,
                    n_baselines=kwargs["n_baselines"],
                    reference=None,
                    obs_dict=obs_dict,
                )

            else:
                logger.error(
                    "Must specify a list of reference antennas for this option."
                )
                return {}
        else:
            logger.error("Valid key not found: {key}".format(key=key))
            return {}

    @staticmethod
    def get_nearest_baselines(
        antenna: str, n_baselines: int = None, path_to_matrix: str = None
    ) -> object:
        import pandas as pd

        if path_to_matrix is None:
            path_to_matrix = str(
                pathlib.Path.cwd().joinpath(".baseline_distance_matrix.csv")
            )

        if not pathlib.Path(path_to_matrix).exists():
            logger.error(
                "Unable to find baseline distance matrix in: {path}".format(
                    path=path_to_matrix
                )
            )

        df_matrix = pd.read_csv(path_to_matrix, sep="\t", index_col=0)

        # Skip the first index because it is a self distance
        if n_baselines is None:
            return (
                df_matrix[antenna].sort_values(ascending=True).index[1:].values.tolist()
            )

        return (
            df_matrix[antenna]
            .sort_values(ascending=True)
            .index[1:n_baselines]
            .values.tolist()
        )

    @staticmethod
    def _select_ddi(value: Union[int, List[int]], obs_dict: object) -> object:
        convert = lambda x: "ddi_" + str(x)

        if not isinstance(value, list):
            value = [value]

        value = list(map(convert, value))
        ddi_list = list(obs_dict.keys())

        for ddi in ddi_list:
            if ddi not in value:
                obs_dict.pop(ddi)

        return obs_dict

    @staticmethod
    def _select_map(value: Union[int, List[int]], obs_dict: object) -> object:
        convert = lambda x: "map_" + str(x)

        if not isinstance(value, list):
            value = [value]

        value = list(map(convert, value))
        ddi_list = list(obs_dict.keys())

        for ddi in ddi_list:
            map_list = list(obs_dict[ddi].keys())
            for mp in map_list:
                if mp not in value:
                    obs_dict[ddi].pop(mp)

        return obs_dict

    @staticmethod
    def _select_antenna(value: Union[str, List[str]], obs_dict: object) -> object:
        if not isinstance(value, list):
            value = [value]

        ddi_list = list(obs_dict.keys())

        for ddi in ddi_list:
            map_list = list(obs_dict[ddi].keys())
            for mp in map_list:
                ant_list = list(obs_dict[ddi][mp]["ant"].keys())
                for ant in ant_list:
                    if ant not in value:
                        obs_dict[ddi][mp]["ant"].pop(ant)

        return obs_dict

    @staticmethod
    def _select_scan(value: Union[int, List[int]], obs_dict: object) -> object:
        if not isinstance(value, list):
            value = [value]

        ddi_list = list(obs_dict.keys())

        for ddi in ddi_list:
            map_list = list(obs_dict[ddi].keys())
            for mp in map_list:
                obs_dict[ddi][mp]["scans"] = value

        return obs_dict

    @staticmethod
    def _select_baseline(
        value: str,
        n_baselines: int,
        obs_dict: object,
        reference: Union[str, List[int]] = None,
    ) -> object:
        if reference is not None:
            if not isinstance(reference, list):
                reference = [reference]

        ddi_list = list(obs_dict.keys())

        for ddi in ddi_list:
            map_list = list(obs_dict[ddi].keys())
            for mp in map_list:
                ant_list = list(obs_dict[ddi][mp]["ant"].keys())
                for ant in ant_list:
                    if ant not in value:
                        obs_dict[ddi][mp]["ant"].pop(ant)
                        continue

                    if reference is None and n_baselines is not None:
                        reference_antennas = obs_dict[ddi][mp]["ant"][ant]

                        if n_baselines > len(reference_antennas):
                            n_baselines = len(reference_antennas)

                        sorted_antennas = np.array(
                            obs_dict.get_nearest_baselines(antenna=ant)
                        )

                        values, i, j = np.intersect1d(
                            reference_antennas, sorted_antennas, return_indices=True
                        )
                        index = np.sort(j)

                        obs_dict[ddi][mp]["ant"][ant] = sorted_antennas[index][
                            :n_baselines
                        ]

                    else:
                        obs_dict[ddi][mp]["ant"][ant] = reference

        return obs_dict
