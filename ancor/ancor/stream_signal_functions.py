
def demean(stream,**kwargs):
    return stream

def resample(stream,**kwargs):
    return stream

def detrend(stream,**kwargs):
    return stream

def taper(stream,**kwargs):
    return stream

def whiten(stream,**kwawrgs):
    return stream

def t_normalize(stream,**kwargs):
    return stream

def bandpass(stream,**kwargs):
    return stream

def cross_correlate(stream1,stream2,**kwargs):
    return None

def process_all(arr,**kwargs):
    arr  = resample(arr,**kwargs)
    arr  = demean(arr,**kwargs)
    arr  = detrend(arr,**kwargs)
    arr  = whiten(arr,**kwargs)
    arr  = t_normalize(arr,**kwargs)
    return arr