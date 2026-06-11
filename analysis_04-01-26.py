import pickle as pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm

import matplotlib.patches as mp
from matplotlib import cm

import struct as struct
import pandas as pd 
from astropy.io import fits
import sys
import pickle as pickle
import re
import os



filepath = r"C:\Users\roadr\OneDrive\Desktop\LBNL Projects\FCGT\data\FCGT_data_12-15-24\TimePix3\250K_1800s.t3pa" 
file = open(filepath, "r")
for i in range(15):
    line = file.readline()
    if not line:
        break
    print(line.strip())

