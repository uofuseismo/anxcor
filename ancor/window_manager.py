

class WindowManager:


    def __init__(self, window_length=300, overlap_percent=0):
        self.window_length  =300
        self.overlap_percent=0
        self.databases=[]

    def add_database(self, database, worker):
        self.databases.append((database,worker))

    def correlate_windows(self, window_retain,stacks=None,):
        pass