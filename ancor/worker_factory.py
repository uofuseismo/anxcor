import worker_processes as process
from typing import List
from obspy.core import Trace

class Worker:

    def __init__(self,**kwargs):
        self.pz_file=None
        self.zero_step = None
        self.steps=[]


    def __call__(self,trace_list: List[Trace]):
        trace_list=self._sanitize_traces(trace_list)

        for step in self.steps:
            trace_list = step(trace_list)
        return trace_list

    def _sanitize_traces(self, trace_list):
        new_trace_list = []
        for trace in trace_list:
            if trace.stats['npts'] > 1:
                new_trace_list.append(trace)
        return new_trace_list

    def append_step(self,step):
        self.steps.append(step)

    def add_kwarg(self,**kwarg):
        for step in self.steps:
            step.add_kwarg(**kwarg)



def _shapiro(**kwargs):
    worker=Worker()
    worker.append_step(process.RemoveMeanTrend())
    worker.append_step(process.RemoveMeanTrend())
    worker.append_step(process.Taper())
    worker.append_step(process.Downsample(target_rate=4.0))
    worker.append_step(process.OneBit())
    return worker


def _bensen(**kwargs):
    worker=Worker()
    worker.append_step(process.RemoveMeanTrend())
    worker.append_step(process.Taper())
    worker.append_step(process.Downsample(target_rate=4.0))
    worker.append_step(process.RemoveMeanTrend())
    worker.append_step(process.Taper())
    worker.append_step(process.BandPass(freqmin=50.0, freqmax=1.0 / 200))
    worker.append_step(process.Taper())
    worker.append_step(process.RunningAbsoluteMeanNorm(time_window=100.0))
    worker.append_step(process.Taper())
    worker.append_step(process.SpectralWhiten())
    return worker

def _berg(**kwargs):
    worker = Worker()
    worker.append_step(process.RemoveMeanTrend())
    worker.append_step(process.Taper())
    worker.append_step(process.Downsample(target_rate=4.0))
    worker.append_step(process.RemoveMeanTrend())
    worker.append_step(process.Taper())
    worker.append_step(process.BandPass(freqmin=1 / 5.0, freqmax=1.0 / 150))
    worker.append_step(process.Taper())
    worker.append_step(process.MaxMeanComponentNorm(freqmin=1 / 50.0, freqmax=1 / 15.0, time_window=128))
    worker.append_step(process.Taper())
    worker.append_step(process.SpectralWhiten())
    return worker


_worker_approaches = {
    'shapiro': _shapiro,
    'bensen' : _bensen,
    'berg'  : _berg,
}

def build_worker(reference=None,**kwargs):
    if reference is None:
        return Worker()
    else:
        if reference.lower() in _worker_approaches.keys():
            return _worker_approaches[reference](**kwargs)



