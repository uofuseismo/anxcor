from ancor   import WindowManager, IRISBank
from ancor   import worker_factory
from obsplus import WaveBank

def set_delta(trace_list):
    new_list = []
    for trace in trace_list:
        trace.stats.delta = 1.0/250.0
        new_list.append(trace)
    return new_list


database_1     = WaveBank('test_data/test_ancor_bank/test_waveforms')
hv_worker_1    = worker_factory.build_worker('berg et. al. 2018')
hv_worker_1.step['downsample'].set_sampling_rate(10.0)
hv_worker_1.prepend_step(set_delta)

database_2      = IRISBank(database_1, max_distance=40000)
hv_worker_2     = worker_factory.build_worker('berg et. al. 2018')
hv_worker_2.step['downsample'].set_sampling_rate(10.0)

window_manager = WindowManager(window_length=60*5, overlap_percent=75)
window_manager.add_database(database_1, hv_worker_1)
window_manager.add_database(database_2, hv_worker_2)


correlations = window_manager.correlate_windows(window_retain=40)
