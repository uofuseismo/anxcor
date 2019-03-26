import os_utils

import ancor_processor
import obsplus
import obspy

time_format = '"D%d_M%m_Y%Y__H%H:M%M:S%S"'

class AncorBank:

    def __init__(self, job, working_directory,
                 processor_data=None,
                 clean_bank_when_finished=True,
                 clean_output_when_finished=True):
        extension = job['min time'].strftime(time_format)
        self._directories = {
            'root' : working_directory,
            'bank' : os_utils.create_workingdir(working_directory, extension),
            'output'  : os_utils.create_workingdir(working_directory, 'output')
        }
        self._job_attributes        = job
        self._processor             = ancor_processor.build_processor(self._directories['output'],
                                                                      processor_data)
        self._clean_instructions = {
            'bank'   : clean_bank_when_finished,
            'output' : clean_output_when_finished
        }

    def create_bank(self):
        self._bank = obsplus.bank.WaveBank(self._directories['bank'])
        for source_file in self._job_attributes['files']:
            self._bank.put_waveforms(obspy.read(source_file))
        self._bank.update_index()

    def execute(self):
        self._delete()

    def _delete(self):
        if self._clean_instructions['bank']:
            os_utils.delete_dirs(self._directories['bank'])
        if self._clean_instructions['output']:
            os_utils.delete_dirs(self._directories['output'])