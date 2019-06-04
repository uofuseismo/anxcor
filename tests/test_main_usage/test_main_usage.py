from ancor   import WindowManager, IRISBank
from ancor   import worker_factory




database_2      = IRISBank(latitude=38.44, longitude=-112.7, minradius=0, maxradius=0.5)
hv_worker_2     = worker_factory.build_worker('berg et. al. 2018')
hv_worker_2.step['downsample'].set_sampling_rate(10.0)

window_manager = WindowManager(window_length=60*5)
window_manager.add_database(database_2, hv_worker_2)



"""
ways I want to use window_manager


process_windows([given list of window starting times],window_length=something, mode='cpu_limited' or mode='ram_limited')
# will output windows to worker directories


correlate_windows('directory of windowed files',max tau shift,save_to_file=True)
# returns a dict of {'station source' : {'station receiver' : { 'component 1' : {'component 2' : np.ndarray } } } } 

process_and_correlate_windows(all combined arguments


"""