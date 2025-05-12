
import pandas
import datetime

class attendence:
    def __init__(self):
        self.attendence = {}  # Correct dictionary initialization
        self.count = 0

    def take_attendence(self, name):
        current_time = datetime.datetime.now().time()  # Correct way to get current time
        if name in self.attendence:
            self.attendence[name].append(current_time)
        else:
            self.attendence[name] = [current_time]

    def get_attendence(self):
        print(self.count)
        print(self.attendence)
