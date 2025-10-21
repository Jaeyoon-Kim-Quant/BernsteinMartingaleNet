import sys
import os
import numpy as np
import pandas as pd
import glob
import datetime
import exchange_calendars as ec
from zoneinfo import ZoneInfo


def load_data(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    df_close = [np.array(group["close"])[:-1] for _, group in df.groupby('date') if len(group) == 391]
    minutely_returns = np.concatenate([np.diff(np.log(arr)) for arr in df_close])
    std = np.sqrt(np.mean(minutely_returns**2))
    xs = minutely_returns / std * (np.pi / np.sqrt(3))
    return xs

def is_full_nyse_day(date):
    """
    Returns True if `date` is a full NYSE trading day (09:30–16:00 ET),
    False if it's a holiday, weekend, or early-close (half-day).
    
    Parameters
    ----------
    date : str | datetime-like
        The date to check (e.g., '2025-07-03' or pd.Timestamp('2025-07-03')).

    Returns
    -------
    bool
    """
    nyse = ec.get_calendar("XNYS")
    date = pd.Timestamp(date).normalize()

    # Check if it's a trading session at all
    if not nyse.is_session(date):
        return False

    open_time = nyse.session_open(date)
    close_time = nyse.session_close(date)

    # A "full" NYSE day runs 6.5 hours (09:30–16:00 ET)
    full_length = pd.Timedelta(hours=6.5)
    return (close_time - open_time) >= full_length

def get_sequence_data(folder_path, context_window, force_recompute=False):
    output_file_name = os.path.join(folder_path, f"spy_1min_data_context_{context_window}.npz")
    if os.path.exists(output_file_name) and not force_recompute:
        print("using cached data from", output_file_name)
        return np.load(output_file_name)["xs"], np.load(output_file_name)["ys"]

    files = glob.glob(os.path.join(folder_path, "spy_1min_*.csv"))
    print(folder_path)
    print(len(files))

    xs = []
    ys = []
    for file in files:
        date_str = file.split("_")[-1].split(".")[0]
        date_str = date_str[:4] + "-" + date_str[4:6] + "-" + date_str[6:]
        if not is_full_nyse_day(date_str):
            print("skipping", date_str)
            continue
        df = pd.read_csv(file)

        open_time = datetime.datetime.strptime(date_str + " 09:30:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo= ZoneInfo("America/New_York"))
        close_time = datetime.datetime.strptime(date_str + " 16:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo= ZoneInfo("America/New_York"))
        df["ts_recv"] = pd.to_datetime(df["ts_recv"])
        df['ts_recv_est'] = df['ts_recv'].dt.tz_convert('America/New_York')
        df = df[(df["ts_recv_est"] >= open_time) & (df["ts_recv_est"] <= close_time)]
        df["seconds_since_open"] = (df["ts_recv_est"] - open_time).dt.total_seconds()

        data = df[["ret_60s", "rv_60s", "seconds_since_open"]].values
        # Create rolling windows for xs
        if data.shape[0] <= context_window:
            continue
        strides = np.array([data[i-context_window:i, :] for i in range(context_window, data.shape[0])])
        y_vec = data[context_window:, 0]
        if np.isnan(strides).any():
            print("nan in strides", date_str)
            continue
        xs.extend(strides)
        ys.extend(y_vec)
    np.savez(output_file_name, xs=np.array(xs), ys=np.array(ys))
    return np.array(xs), np.array(ys)