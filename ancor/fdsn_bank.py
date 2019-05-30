from obspy.clients.fdsn import RoutingClient, Client
from obspy.core import UTCDateTime

class IRISClient:


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
        self.client = Client('IRIS')
        self.routing_client = RoutingClient('iris-federator')
        self.query_kwargs = query_kwargs


    def get_waveforms(self, starttime=None, endtime=None):

        bulk_request = self._create_bulk_request(UTCDateTime(starttime), UTCDateTime(endtime))
        delta_t = float(int(endtime - starttime))
        streams = self.client.get_waveforms_bulk(bulk_request,minimumlength=delta_t, attach_response=True)
        return streams


    def _create_bulk_request(self, starttime, endtime):
        streams = self.routing_client.get_waveforms(starttime=starttime,
                                                    endtime=endtime, **self.query_kwargs)
        bulk_request = []
        for trace in streams.traces:
            network = trace.stats['network']
            station = trace.stats['station']
            location= trace.stats['location']
            channel = trace.stats['channel']
            bulk_request.append([network, station, location, channel, starttime, endtime])
        return bulk_request
