# Author: Tuan Chien, James Diprose

class Mfcc:
    """
    Class containing the MFCC data as well as associated meta data.
    """

    def __init__(self, timestamp, nmfccs, mean, stddev, length, stride, data):
        """
        Parameters
        ----------
        timestamp : float
            Timestamp for the start of this data.
        nmfccs : int
            Number of coefficients per time sample.
        mean : float
            Mean value used in normalization.
        stddev : float
            Standard deviation used in normalization.
        length : float
            Length of the window in seconds.
        stride : float
            Size of the window to translate between time samples in seconds.
        data : 2d numpy array
            The MFCC data.
        """
        self.timestamp = timestamp
        self.nmfccs = nmfccs
        self.mean = mean
        self.stddev = stddev
        self.length = length
        self.stride = stride
        self.data = data
