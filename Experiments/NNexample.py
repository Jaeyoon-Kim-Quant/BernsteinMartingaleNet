import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from zoneinfo import ZoneInfo

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.utils import get_sequence_data

folder_path = r"../MarketData/historical_data"
xs, ys = get_sequence_data(folder_path, 10)