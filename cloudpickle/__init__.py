from __future__ import absolute_import

import sys
import pickle


if hasattr(pickle.Pickler, 'reducer_override'):
    from cloudpickle.cloudpickle_fast import *
else:
    from cloudpickle.cloudpickle import *

__version__ = '0.9.0.dev0'
