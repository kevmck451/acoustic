# File to characterize a sound source called "Target"

from sample_library import *
from utils import CSVFile

import numpy as np
import math
import ast


class Target:

    def __init__(self):
        self.name = str
        self.type = str


    def __str__(self):
        return f'---------Target Object---------\n' \
               f'Name: {self.name}\n' \
               f'Type: {self.type}\n' \







if __name__ == '__main__':

    target = Target()