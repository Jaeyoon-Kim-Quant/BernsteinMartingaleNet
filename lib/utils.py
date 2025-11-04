import sys
import os
import numpy as np
import pandas as pd
import glob
import datetime
import exchange_calendars as ec
from zoneinfo import ZoneInfo
import torch
import torch.nn as nn
import torch.optim as optim


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

def train_model(model, train_X, train_Y, dev_X, dev_Y, lr, weight_decay=1e-4, num_steps=1000, batch_size=None, device: torch.device = None, verbose=True):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if batch_size is None:
        batch_size = train_X.shape[0]
    if verbose:
        print("num batches", train_X.shape[0] // batch_size)
    train_losses = []
    dev_losses = []
    
    # Helper function to compute loss in batches
    def compute_loss_batched(X, Y, batch_size):
        model.eval()
        with torch.no_grad():
            total_loss = 0.0
            num_samples = 0
            for i in range(0, X.shape[0], batch_size):
                batch_X = X[i:i+batch_size, :]
                batch_Y = Y[i:i+batch_size, :]
                batch_loss = model(batch_X, batch_Y)
                batch_size_actual = batch_X.shape[0]
                total_loss += batch_loss.item() * batch_size_actual
                num_samples += batch_size_actual
            return total_loss / num_samples if num_samples > 0 else 0.0
    
    # Helper function to compute individual losses and their std
    def compute_loss_std(X, Y, batch_size):
        model.eval()
        with torch.no_grad():
            all_losses = []
            use_naive_pdf = getattr(model, 'use_naive_pdf', False)
            for i in range(0, X.shape[0], batch_size):
                batch_X = X[i:i+batch_size, :]
                batch_Y = Y[i:i+batch_size, :]
                # Get parameters for the batch
                params = model.get_params(batch_X)
                # Compute individual losses (negative logpdf) without taking mean
                if use_naive_pdf:
                    individual_losses = -model.blogistic.naive_logpdf(batch_Y, params[:, :-1], params[:, -1])
                else:
                    individual_losses = -model.blogistic.logpdf(batch_Y, params[:, :-1], params[:, -1])
                all_losses.append(individual_losses.cpu())
            # Concatenate all individual losses
            all_losses = torch.cat(all_losses, dim=0)
            # Calculate standard deviation
            loss_std = torch.std(all_losses).item()
            return loss_std
    
    train_loss = compute_loss_batched(train_X, train_Y, batch_size)
    dev_loss = compute_loss_batched(dev_X, dev_Y, batch_size)
    dev_loss_std = compute_loss_std(dev_X, dev_Y, batch_size) / np.sqrt(dev_X.shape[0])
    if verbose:
        print(f"Init, Train Loss: {train_loss:.4f}, Dev Loss: {dev_loss:.4f}, Dev Loss confidence interval: {dev_loss - 2 * dev_loss_std:.4f}, {dev_loss + 2 * dev_loss_std:.4f}")
    train_losses.append(train_loss)
    dev_losses.append(dev_loss)

    for step in range(num_steps):
        model.train()
        # run mini-batch training
        total_loss = 0.0
        for i in range(0, train_X.shape[0], batch_size):
            batch_X = train_X[i:i+batch_size, :]
            batch_Y = train_Y[i:i+batch_size, :]
            loss = model(batch_X, batch_Y)
            total_loss += loss.item() * batch_X.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        #print(f"Step {step}, Total Loss: {total_loss / train_X.shape[0]:.4f}")

        if step % 10 == 0:
            train_loss = compute_loss_batched(train_X, train_Y, batch_size)
            dev_loss = compute_loss_batched(dev_X, dev_Y, batch_size)
            dev_loss_std = compute_loss_std(dev_X, dev_Y, batch_size) / np.sqrt(dev_X.shape[0])
            train_losses.append(train_loss)
            dev_losses.append(dev_loss)

            if verbose:
                print(f"Step {step}, Train Loss: {train_loss:.4f}, Dev Loss: {dev_loss:.4f}, Dev Loss confidence interval: {dev_loss - 2 * dev_loss_std:.4f}, {dev_loss + 2 * dev_loss_std:.4f}")
    return model, train_losses, dev_losses