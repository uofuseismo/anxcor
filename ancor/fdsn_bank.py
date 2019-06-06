from obspy.clients.fdsn import RoutingClient, Client
from obspy.core import UTCDateTime
import re

class IRISBank:


    def __init__(self,**query_kwargs):
        """
        Uses the iris-federator or the eida-routing client to retrieve waveforms

        Parameters
        ----------
        query_kwargs:
            For the \'iris-federator\' client , the following kwarg values can be passed:
            "includeoverlaps", "level", "network", "station", "channel",
            "location", "minlatitude", "maxlatitude",
            "minlongitude", "maxlongitude", "latitude", "longitude",
            "minradius", "maxradius"

        """
        self._default_components = ['.H.', '.P.', '.L.']
        self.client = Client('IRIS')
        self.routing_client = RoutingClient('iris-federator')
        self.query_kwargs = query_kwargs


    def get_waveforms(self, starttime=None, endtime=None):
        startutc= UTCDateTime(starttime)
        endutc = UTCDateTime(endtime)
        bulk_request = self._create_bulk_request(startutc,endutc)
        delta_t = endtime - starttime
        streams = self.client.get_waveforms_bulk(bulk_request,minimumlength=delta_t, attach_response=True)
        return streams


    def get_default_components(self):
        return self._default_components

    def set_default_components(self,default_components):
        self._default_components=default_components


    def _create_bulk_request(self, starttime, endtime):

        streams = self.routing_client.get_waveforms(starttime=starttime, endtime=endtime,**self.query_kwargs)
        bulk_request = []
        for trace in streams.traces:
            network = trace.stats['network']
            station = trace.stats['station']
            location= trace.stats['location']
            channel = trace.stats['channel']
            for acceptable_channel in self._default_components:
                if re.search(acceptable_channel,channel):
                    bulk_request.append([network, station, location, channel, starttime, endtime])
                    break
        return bulk_request
