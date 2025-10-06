import pandas as pd
import numpy as np

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    df_close = [np.array(group["close"])[:-1] for _, group in df.groupby('date') if len(group) == 391]
    minutely_returns = np.concatenate([np.diff(np.log(arr)) for arr in df_close])
    std = np.sqrt(np.mean(minutely_returns**2))
    xs = minutely_returns / std * (np.pi / np.sqrt(3))
    return xs