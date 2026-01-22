from astrohack.io.base_mds import AstrohackBaseFile


class AstrohackHologFile(AstrohackBaseFile):

    def __init__(self, file: str):
        """Initialize an AstrohackHologFile object.

        :param file: File to be linked to this object
        :type file: str

        :return: AstrohackHologFile object
        :rtype: AstrohackHologFile
        """
        super().__init__(file=file)
