from worker_processes import AncorProcessorBase
from typing import List
from obspy.core import Trace



class SetTrimVerify(AncorProcessorBase):
    window = 300
    def __init__(self):
        super().__init__()


    def __call__(self,trace_list: List[Trace])->List[Trace]:
        # set samples
        new_list = []
        for trace in trace_list:
            trace.stats['delta']= 1.0/250.0
            new_list.append(trace)
        # trim
        for trace in trace_list:
            starttime = trace.stats['starttime']
            trace.trim(starttime=starttime,endtime=starttime+self.window)

        #n sample target:
        target = 250*self.window
        for trace in trace_list:
            if trace.stats['npts']==target:
                new_list.append(trace)

        return new_list