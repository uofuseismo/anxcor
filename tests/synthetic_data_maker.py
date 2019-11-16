{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "import obspy\n",
    "from obspy.core import UTCDateTime, Trace, Stream\n",
    "import numpy as np\n",
    "\n",
    "stations   = ['1','2','3']\n",
    "components = ['HHN', 'HHZ','HHE']\n",
    "starttime  = UTCDateTime(0)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "\"\"\"\n",
    "traces should be:\n",
    "\n",
    "10 min total all traces, aimed at \n",
    "starting from 0 UTCDateTime\n",
    "50 hz sampling\n",
    "network = 'test'\n",
    "\n",
    "Station    component:\n",
    "            \n",
    "                2 min         2 min         2 min         2 min         2 min\n",
    "                perfect data  one channel   one station   one channel   two stations  \n",
    "                              gone          gone          incomplete    incomplete\n",
    "\n",
    "(1)         HHE ------------- ------------- ------------- ------------- ------------- \n",
    "            HHN ------------- ------------- ------------- ------------- ------------- \n",
    "            HHZ ------------- ------------- ------------- ------------- -------------\n",
    "            \n",
    "(2)         HHE ------------- -------------               ------------- ----------\n",
    "            HHN -------------                             ------------- ----------\n",
    "            HHZ ------------- -------------               ----------    ---------- \n",
    "            \n",
    "(3)         HHE ------------- ------------- ------------- ------------- ----------\n",
    "            HHN ------------- ------------- ------------- ------------- ----------\n",
    "            HHZ ------------- ------------- ------------- ------------- ----------\n",
    "\"\"\"\n",
    "sample_rate =50.0\n",
    "duration    = 2 * 60\n",
    "channels    = ['HHE','HHN','HHZ']\n",
    "network     = 'AX'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "# station 1 data\n",
    "station = '1'\n",
    "total_time = 0\n",
    "traces = []\n",
    "for window in range(0,5):\n",
    "    for channel in channels:\n",
    "        ch_data   = np.random.uniform(-1.0,1.0,int(sample_rate*duration))\n",
    "        header = {\n",
    "            'starttime' : UTCDateTime(total_time),\n",
    "            'delta'     : 1/sample_rate,\n",
    "            'station'   : station,\n",
    "            'network'   : network,\n",
    "            'channel'   : channel\n",
    "            \n",
    "        }\n",
    "        trace = Trace(ch_data,header=header)\n",
    "        traces.append(trace)\n",
    "    total_time+= duration "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "# station 2 data\n",
    "station = '2'\n",
    "total_time = 0\n",
    "for channel in channels:\n",
    "    ch_data   = np.random.uniform(-1.0,1.0,int(sample_rate*duration))\n",
    "    header = {\n",
    "            'starttime' : UTCDateTime(total_time),\n",
    "            'delta'     : 1/sample_rate,\n",
    "            'station'   : station,\n",
    "            'network'   : network,\n",
    "            'channel'   : channel\n",
    "            \n",
    "    }\n",
    "    trace = Trace(ch_data,header=header)\n",
    "    traces.append(trace)\n",
    "total_time+= duration \n",
    "station = '2'\n",
    "for channel in channels:\n",
    "    if channel!='HHN':\n",
    "        ch_data   = np.random.uniform(-1.0,1.0,int(sample_rate*duration))\n",
    "        header = {\n",
    "            'starttime' : UTCDateTime(total_time),\n",
    "            'delta'     : 1/sample_rate,\n",
    "            'station'   : station,\n",
    "            'network'   : network,\n",
    "            'channel'   : channel\n",
    "            \n",
    "        }\n",
    "        trace = Trace(ch_data,header=header)\n",
    "        traces.append(trace)\n",
    "total_time+= duration*2\n",
    "for channel in channels:\n",
    "    header = {\n",
    "            'starttime' : UTCDateTime(total_time),\n",
    "            'delta'     : 1/sample_rate,\n",
    "            'station'   : station,\n",
    "            'network'   : network,\n",
    "            'channel'   : channel\n",
    "        }\n",
    "    if channel!='HHZ':\n",
    "        ch_data   = np.random.uniform(-1.0,1.0,int(sample_rate*duration))\n",
    "    else:\n",
    "        ch_data   = np.random.uniform(-1.0,1.0,int(sample_rate*duration*3/4))\n",
    "        \n",
    "    trace = Trace(ch_data,header=header)\n",
    "    traces.append(trace)\n",
    "    \n",
    "total_time+= duration\n",
    "for channel in channels:\n",
    "    header = {\n",
    "            'starttime' : UTCDateTime(total_time),\n",
    "            'delta'     : 1/sample_rate,\n",
    "            'station'   : station,\n",
    "            'network'   : network,\n",
    "            'channel'   : channel\n",
    "        }\n",
    "    ch_data   = np.random.uniform(-1.0,1.0,int(sample_rate*duration*3/4))    \n",
    "    trace = Trace(ch_data,header=header)\n",
    "    traces.append(trace)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "station = '3'\n",
    "total_time = 0\n",
    "for window in range(0,4):\n",
    "    for channel in channels:\n",
    "        ch_data   = np.random.uniform(-1.0,1.0,int(sample_rate*duration))\n",
    "        header = {\n",
    "            'starttime' : UTCDateTime(total_time),\n",
    "            'delta'     : 1/sample_rate,\n",
    "            'station'   : station,\n",
    "            'network'   : network,\n",
    "            'channel'   : channel\n",
    "        }\n",
    "        trace = Trace(ch_data,header=header)\n",
    "        traces.append(trace)\n",
    "    total_time+= duration\n",
    "for channel in channels:\n",
    "    ch_data   = np.random.uniform(-1.0,1.0,int(sample_rate*duration*3/5))\n",
    "    header = {\n",
    "            'starttime' : UTCDateTime(total_time),\n",
    "            'delta'     : 1.0/sample_rate,\n",
    "            'station'   : station,\n",
    "            'network'   : network,\n",
    "            'channel'   : channel\n",
    "    }\n",
    "    trace = Trace(ch_data,header=header)\n",
    "    traces.append(trace)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "stream = Stream(traces=traces)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "41"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(stream)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "6000\n",
      "6000\n",
      "6000\n",
      "6000\n",
      "6000\n",
      "6000\n",
      "6000\n",
      "6000\n",
      "6000\n",
      "6000\n",
      "6000\n",
      "6000\n",
      "6000\n",
      "6000\n",
      "6000\n",
      "6000\n",
      "6000\n",
      "6000\n",
      "6000\n",
      "6000\n",
      "6000\n",
      "6000\n",
      "4500\n",
      "4500\n",
      "4500\n",
      "4500\n",
      "6000\n",
      "6000\n",
      "6000\n",
      "6000\n",
      "6000\n",
      "6000\n",
      "6000\n",
      "6000\n",
      "6000\n",
      "6000\n",
      "6000\n",
      "6000\n",
      "3600\n",
      "3600\n",
      "3600\n"
     ]
    }
   ],
   "source": [
    "save_dir = '../tests/test_data/test_anxcor_database/test_waveforms_multi_station/'\n",
    "import time\n",
    "format = 'mseed'\n",
    "for trace in stream:\n",
    "    station = trace.stats.station\n",
    "    network = trace.stats.network\n",
    "    channel = trace.stats.channel\n",
    "    starttime    = trace.stats.starttime.isoformat()\n",
    "    print(trace.stats.npts)\n",
    "    file_name = '{}{}.{}.{}.{}.{}'.format(save_dir,network,station,channel,starttime,'mseed')\n",
    "    trace.write(file_name,format='mseed')\n",
    "    #time.sleep(0.1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:ancor] *",
   "language": "python",
   "name": "conda-env-ancor-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
