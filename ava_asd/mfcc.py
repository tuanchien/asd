# Copyright 2020 Tuan Chien, James Diprose
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
