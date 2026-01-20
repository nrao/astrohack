from astrohack.io.base_mds import AstrohackBaseFile


class AstrohackPointFile(AstrohackBaseFile):

    def __init__(self, file: str):
        """Initialize an AstrohackPointFile object.

        :param file: File to be linked to this object
        :type file: str

        :return: AstrohackPointFile object
        :rtype: AstrohackPointFile
        """
        super().__init__(file=file)
