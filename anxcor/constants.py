"""
defaults used in processing routine initialization
"""
TAPER_DEFAULT   =0.05
RESAMPLE_DEFAULT=10.0
UPPER_CUTOFF_FREQ=5.0
LOWER_CUTOFF_FREQ=0.01
MAX_TAU_DEFAULT=100.0
FILTER_ORDER_BANDPASS=4
FILTER_ORDER_WHITEN=6
SECONDS_2_NANOSECONDS = 1e9
T_NORM_WINDOW=10.0
T_NORM_LOWER_FREQ=0.001
T_NORM_UPPER_FREQ=0.05
WHITEN_WINDOW_RATIO=10.0

OPERATIONS_SEPARATION_CHARACTER = '\n'
WHITEN_TYPE='whiten_type'
T_NORM_TYPE='reduce_metric'
ROLLING_METRIC= 'mean'
REDUCE_METRIC= 'max'