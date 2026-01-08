from astrohack.io.base_mds import AstrohackBaseFile


class AstrohackLocitFile2(AstrohackBaseFile):
    """Data class for locit data.

    Data within an object of this class can be selected for further inspection, plotted or produce a report
    """

    def __init__(self, file: str):
        """Initialize an AstrohackLocitFile object.

        :param file: File to be linked to this object
        :type file: str

        :return: AstrohackLocitFile object
        :rtype: AstrohackLocitFile
        """
        super().__init__(file=file)
