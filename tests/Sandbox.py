import pandas as pd
from functions import process_dataframe

x = [
    1.00002,
    1.00003,
    1.00258,
    1.02589,
    100059,
    100056,
    100560
]
y = [
    0,
    1e-9,
    0,
    0,
    1e-9,
    0,
    0
]
data = {'Sol_r':x,'Sol_i':y}

process_dataframe(pd.DataFrame(data),4)