

import utils.tensor
import utils.la4
import utils.la3np
import utils.la1
import utils.la2
import utils.exr
import time

class Timer:
    def __init__(self, name):
        self.name = name
        self.elapsed = None

    def __enter__(self):
        self.start = time.clock()
        return self

    def calc_diff(self):
        self.end = time.clock()
        self.elapsed = self.end - self.start

    def print_elapsed(self):
        self.calc_diff()
        print("Timer {}:\t{}".format(self.name, self.elapsed))

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.print_elapsed()

    def __del__(self):
        if self.elapsed is None:
            self.print_elapsed()

def apply(func,*args):
    return [func(arg) for arg in args]