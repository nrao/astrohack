from astrohack.io.base_mds import AstrohackBaseFile


class AstrohackPanelFile(AstrohackBaseFile):
    """Data class for panel data.

    Data within an object of this class can be selected for further inspection, plotted or produce a report
    """

    def __init__(self, file: str):
        """Initialize an AstrohackPanelFile object.

        :param file: File to be linked to this object
        :type file: str

        :return: AstrohackPanelFile object
        :rtype: AstrohackPanelFile
        """
        super().__init__(file=file)
