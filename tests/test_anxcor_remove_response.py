
import unittest
# travis execution
try:
    from tests.synthetic_trace_factory import create_sinsoidal_trace
except:
    from synthetic_trace_factory import create_sinsoidal_trace
#ide testing
#from synthetic_trace_factory import create_sinsoidal_trace
import numpy as np
from anxcor.xarray_routines import XArrayResample, XArrayConverter
import anxcor.anxcor_utils as anxcor_utils
from obspy.clients.fdsn import Client
from obspy.core import UTCDateTime
converter =XArrayConverter()
import matplotlib.pyplot as plt

def test_remove_response_not_identical_to_returned_response():
    client = Client("IRIS")
    t = UTCDateTime("2018-12-25 12:00:00").timestamp
    st = client.get_waveforms("UU", "SPU", "*", "H*", t, t + 10 * 60, attach_response=True)

    target_trace = st[0].copy()
    pre_filt = (0.01, 0.03, 40.0, 45.0)
    source_trace = anxcor_utils.remove_response(st[0], output='DISP', pre_filt=pre_filt,
                                                zero_mean=True, taper=True)
    target_trace.normalize()
    source_trace.normalize()
    assert not np.allclose(source_trace.data,target_trace.data)