

def build_processor(working_directory, settings):
    if settings=="single test processor":
        return SingleTestProcessor()
    elif settings=="multi test processor":
        return MultiTestProcessor()

class Processor:

    def __init__(self,*args,**kwargs):
        pass

    def process(self,job):
        """

        Parameters
        ----------
        job: dict
            a job dict with the following keys:
                'windows':
                    window start and stop times as tuples in list
                'max time':
                    max timestamp in utc
                'min time':
                    min timestamp in utc

        """
        pass


class SingleTestProcessor(Processor):

    def __init__(self):
        super().__init__()
        self.success = False

    def process(self, job):
        if 'windows' in job.keys():
            self.success = True

class MultiTestProcessor(Processor):

    def __init__(self):
        super().__init__()
        self.successes=[]

    def process(self, job):
        self.successes.append('windows' in job.keys())