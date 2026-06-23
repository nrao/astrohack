from importlib.metadata import version


def get_astrohack_version():
    return version("astrohack")


def get_astrohack_path():
    from importlib.resources import files

    package_dir = files("astrohack")
    return package_dir
