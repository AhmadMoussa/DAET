"""
Helper Class that allows the encoding and decoding of the dataset into a json object

"""

import json
from json import JSONEncoder
import librosa

class Class_No_JSONSerialization:
    pass

class DataPointEncoder(JSONEncoder):

    def default(self, object):

        if isinstance(object, DataPoint):
            return object.__dict__

        else:
            return json.JSONEncoder.default(self, object)