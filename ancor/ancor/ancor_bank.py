import os_utils

import ancor_processor
import obsplus
import obspy
import regex_utils
from datetime import datetime
time_format = '"D%d_M%m_Y%Y__H%H:M%M:S%S"'


class AncorBank:

    def __init__(self, job, working_directory,
                 processor_data=None,
                 clean_bank_when_finished=True):
        """

        Parameters
        ----------
        job: list or dict
            list of jobs to assign to this bank

        working_directory: str
            the working directory to use

        processor_data: dict or str

        clean_bank_when_finished: bool, True
        clean_output_when_finished
        """

        self._directories = {
            'root' : working_directory,
            'bank directories': [],
            'output'  : []
        }

        self._jobs        = job
        self._processor             = ancor_processor.build_processor(self._directories['output'],
                                                                      processor_data)
        self._clean_instructions = clean_bank_when_finished

    def _create_bank(self,job):
        bank_dir = os_utils.create_workingdir(self._directories['root'],'banks',job['min time'])
        output   = os_utils.create_workingdir(self._directories['root'], 'output', job['min time'])
        self._directories['bank directories'].append(bank_dir)
        self._directories['output'].append(output)
        bank = obsplus.bank.WaveBank(bank_dir)
        for source_file in job['files']:
            bank.put_waveforms(obspy.read(source_file))
        bank.update_index()
        return bank,output

    def execute(self):
        if isinstance(self._jobs,list):
            self._execute_job_list()
        else:
            self._process_job(self._jobs)

    def _delete(self):
        if self._clean_instructions:
            self._delete_dirs('bank directories')

    def _delete_dirs(self,key):
        for directory in self._directories[key]:
            os_utils.delete_dirs(directory)

    def _execute_job_list(self):
        for job in self._jobs:
            self._process_job(job,delete=False)
        self._delete()

    def _process_job(self, job,delete=True):
        bank,output = self._create_bank(job)
        job['bank'] = bank
        job['output dir']= output
        self._processor.process(job)
        if delete:
            self._delete()
