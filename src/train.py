import logging
import sys
import pathlib

import numpy as np
import pandas as pd
import feather

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger(__name__)

log.info("Load data")
train = feather.read_dataframe('/data/train')
test = feather.read_dataframe('/data/test')
with open('/data/final_num_cols.txt', 'r') as f:
    num_cols = [c.strip() for c in f.readlines()]

print(num_cols)
print(train.shape)
print(test.shape)
