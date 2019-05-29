import ancor_processor as processors
from typing import List
from obspy.core import Trace

class Worker:

    def __init__(self,**kwargs):
        self.pz_file=None
        self.zero_step = None
        self.steps=[]

    def add_zero_step(self,step):
        self.zero_step = step


    def __call__(self,trace_list: List[Trace], response_file):

        #TODO: remove response before working on things
        if self.zero_step is not None:
            trace_list = self.zero_step(trace_list)
        for step in self.steps:
            trace_list = step(trace_list,response_file)
        return trace_list

    def append_step(self,step):
        self.steps.append(step)

    def add_kwarg(self,**kwarg):
        for step in self.steps:
            step.add_kwarg(**kwarg)



def _shapiro(**kwargs):
    worker=Worker()
    worker.append_step(processors.RemoveMeanTrend())
    worker.append_step(processors.Taper())
    worker.append_step(processors.Resample(target=4.0))
    worker.append_step(processors.OneBit())
    return worker


def _bensen(**kwargs):
    worker=Worker()
    worker.append_step(processors.RemoveMeanTrend())
    worker.append_step(processors.Taper())
    worker.append_step(processors.Resample(target=4.0))
    worker.append_step(processors.Taper())
    worker.append_step(processors.BandPass(freqmin=50.0,freqmax=1.0/200))
    worker.append_step(processors.Taper())
    worker.append_step(processors.RunningAbsoluteMeanNorm(time_window=100.0))
    worker.append_step(processors.Taper())
    worker.append_step(processors.SpectralWhiten())
    return worker

def _berg(**kwargs):
    worker = Worker()
    worker.append_step(processors.Taper())
    worker.append_step(processors.Resample(target=4.0))
    worker.append_step(processors.RemoveMeanTrend())
    worker.append_step(processors.Taper())
    worker.append_step(processors.BandPass(freqmin=1/5.0, freqmax=1.0 / 150))
    worker.append_step(processors.Taper())
    worker.append_step(processors.MaxMeanComponentNorm(freqmin=1 / 50.0, freqmax=1 / 15.0, time_window=128))
    worker.append_step(processors.Taper())
    worker.append_step(processors.SpectralWhiten())
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



