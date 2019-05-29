from obspy.core.utcdatetime import UTCDateTime
from multiprocessing import Process
import psutil
import copy

def execute(traces,worker):
    output = worker(traces)
    return output

class WindowManager:
    SAFE_DELTA = 1.0

    def __init__(self, window_length=300, overlap_percent=0, retained_window=40):
        self.window_length   = 300
        self.overlap_percent = 0
        self.retained_window = retained_window
        self.databases=[]

    def add_database(self, database, worker):
        self.databases.append((database,worker))

    def correlate_windows(self, max_stacks=None,starttime=None,endtime=None):
        self._check_params(max_stacks,starttime,endtime)
        max_windows_in_ram = self._ram_check(starttime)

    def _check_params(self,max_stacks,starttime,endtime):
        if max_stacks is None and starttime is None and endtime is None:
            raise SyntaxError('Too many Nones! Please specify a max_stacks integer'+\
                              'value and a valid starttime or a valid starttime and endtime')

    def _ram_check(self,starttime: UTCDateTime):
        initial = psutil.virtual_memory()['used']
        endtime = starttime + self.window_length + self.SAFE_DELTA
        worker_arguments = []
        for worker, database in self.databases:
            waveforms            = database.get_waveforms(starttime=starttime,endtime=endtime)
            collected_components = self._collect_components(waveforms)
            for key, value in collected_components.items():
                worker_arguments.append((value, copy.deepcopy(worker)))

        p = Process(target=execute, args=worker_arguments)
        p.start()
        #not sure how to get objects back
        final = psutil.virtual_memory()['used']
        p.join()



        delta_ram = final - initial

        ram_available = psutil.virtual_memory()['total']  - psutil.virtual_memory()['used'] - psutil.swap_memory()['used']

        possible_instances = int(ram_available/delta_ram)

        return possible_instances


    def _collect_components(self,stream):
        station_dict = {}
        for trace in stream.traces:

            key = trace.stats.station

            if key not in station_dict:
                station_dict[key]=[]

            station_dict[key].append(trace)

        return station_dict