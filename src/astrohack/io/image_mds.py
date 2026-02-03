from astrohack.io.base_mds import AstrohackBaseFile


class AstrohackImageFile(AstrohackBaseFile):
    """Data class for image data.

    Data within an object of this class can be selected for further inspection, plotted or produce a report
    """

    def __init__(self, file: str):
        """Initialize an AstrohackImageFile object.

        :param file: File to be linked to this object
        :type file: str

        :return: AstrohackImageFile object
        :rtype: AstrohackImageFile
        """
        super().__init__(file=file)
