from astrohack.io.base_mds import AstrohackBaseFile


class AstrohackPositionFile(AstrohackBaseFile):

    def __init__(self, file: str):
        """Initialize an AstrohackPositionFile object.

        :param file: File to be linked to this object
        :type file: str

        :return: AstrohackPositionFile object
        :rtype: AstrohackPositionFile
        """
        super().__init__(file=file)
