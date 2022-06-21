import pandas as pd
import numpy as np

def toRecency(data, upper_):
    enc = [upper_]
    enc[0] = 1
    for x in range(1, upper_):
        if data[x] == data[x-1]:
            enc.append(enc[x-1])
        else:
            enc.append(enc[x-1]+1)
    for x in range(0, upper_):
        enc[x] = enc[upper_ - 1] - enc[x] + 1
    
    return enc