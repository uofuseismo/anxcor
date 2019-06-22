import  unittest
from obspy.core import UTCDateTime
from anxcor import IRISBank

christmas_time = UTCDateTime('2018-12-25T00:00:00.0')

class TestWindowManager(unittest.TestCase):

    def test_irisbank_single_channel(self):
        iris_bank = IRISBank(longitude=-111,latitude=39,maxradius=0.5,minradius=0)
        iris_bank.set_default_components(['..Z'])
        timestamp = christmas_time.timestamp
        streams = iris_bank.get_waveforms(starttime=timestamp,endtime=timestamp+100)
        not_z = []
        for trace in streams:
            if 'Z' not in trace.stats.channel:
                not_z.append(1)
        self.assertEqual(0,len(not_z), 'component filter did not work')


    def test_irisbank_multi_channel(self):
        iris_bank = IRISBank(longitude=-111,latitude=39,maxradius=0.5,minradius=0)
        iris_bank.set_default_components(['..Z','..E'])
        timestamp = christmas_time.timestamp
        streams = iris_bank.get_waveforms(starttime=timestamp,endtime=timestamp+100)
        not_z = []
        for trace in streams:
            if 'Z' not in trace.stats.channel and 'E' not in trace.stats.channel:
                not_z.append(1)
        self.assertEqual(0,len(not_z), 'multi component filter did not work')


    def test_consistently_returns_same_streams(self):
        iris_bank = IRISBank(longitude=-111, latitude=39, maxradius=0.5, minradius=0)
        window = 60*15.0
        starttime = christmas_time.timestamp
        endtime   = window + starttime
        streams = iris_bank.get_waveforms(starttime=starttime,endtime=endtime)

        source = len(streams)
        target = 23

        self.assertEqual(target,source,'incorrect number of streams have been returned')