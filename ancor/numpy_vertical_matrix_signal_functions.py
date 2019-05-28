
def demean(arr,**kwargs):
    return arr

def resample(arr,**kwargs):
    return arr

def detrend(arr,**kwargs):
    return arr

def taper(arr,**kwargs):
    return arr

def whiten(arr,**kwawrgs):
    return arr

def t_normalize(arr,**kwargs):
    return arr

def bandpass(arr,**kwargs):
    return arr

def cross_correlate(arr1,arr2,**kwargs):
    return None


def process_all(arr,**kwargs):
    arr  = resample(arr,**kwargs)
    arr  = demean(arr,**kwargs)
    arr  = detrend(arr,**kwargs)
    arr  = whiten(arr,**kwargs)
    arr  = t_normalize(arr,**kwargs)
    return arr

