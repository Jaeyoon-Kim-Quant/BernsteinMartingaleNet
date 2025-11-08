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
import json

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

def get_sequence_data_by_month(folder_path, context_window, num_dev_months, num_test_months, force_recompute=False):
    output_file_name = os.path.join(folder_path, f"spy_1min_data_context_{context_window}_by_month.npz")
    if os.path.exists(output_file_name) and not force_recompute:
        print("using cached data from", output_file_name)
        f = np.load(output_file_name)
        train_xs = f["train_xs"]
        train_ys = f["train_ys"]
        dev_xs = f["dev_xs"]
        dev_ys = f["dev_ys"]
        test_xs = f["test_xs"]
        test_ys = f["test_ys"]
        return train_xs, train_ys, dev_xs, dev_ys, test_xs, test_ys
    files = glob.glob(os.path.join(folder_path, "spy_1min_*.csv"))
    xs = {}
    ys = {}
    months = set()
    for file in files:
        date_str = file.split("_")[-1].split(".")[0]
        date_str = date_str[:4] + "-" + date_str[4:6] + "-" + date_str[6:]
        if not is_full_nyse_day(date_str):
            print("skipping", date_str)
            continue

        month = date_str[:7]
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
        if month not in xs:
            xs[month] = []
            ys[month] = []
            months.add(month)
        xs[month].extend(strides)
        ys[month].extend(y_vec)
    
    np.random.seed(0)
    months = np.random.permutation(list(months))
    dev_months = months[:num_dev_months]
    test_months = months[num_dev_months:num_dev_months + num_test_months]
    train_months = months[num_dev_months + num_test_months:]
    print("train months", train_months)
    print("dev months", dev_months)
    print("test months", test_months)
    print("num months", len(train_months), len(dev_months), len(test_months))

    def combine_data(months):
        xs_combined = []
        ys_combined = []
        for month in months:
            xs_combined.extend(xs[month])
            ys_combined.extend(ys[month])
        return np.array(xs_combined), np.array(ys_combined)

    train_xs, train_ys = combine_data(train_months)
    dev_xs, dev_ys = combine_data(dev_months)
    test_xs, test_ys = combine_data(test_months)

    np.savez(output_file_name, train_xs=train_xs, train_ys=train_ys, dev_xs=dev_xs, dev_ys=dev_ys, test_xs=test_xs, test_ys=test_ys)
    return train_xs, train_ys, dev_xs, dev_ys, test_xs, test_ys


def train_model(
    model, 
    train_X, 
    train_Y, 
    dev_X, 
    dev_Y, 
    test_X, 
    test_Y, 
    lr, 
    weight_decay=1e-4, 
    num_steps=1000, 
    batch_size=None, 
    device: torch.device = None, 
    verbose=True, 
    output_folder=None,
    lr_decay_step=200,    # every N steps to decay learning rate
    lr_decay_gamma=0.5,   # decay factor
):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)

    if batch_size is None:
        batch_size = train_X.shape[0]
    if verbose:
        print("num batches", train_X.shape[0] // batch_size)
    train_losses = []
    dev_losses = []
    loss_steps = []
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
    def eval_loss(X, Y, batch_size):
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
                individual_losses = -model.dist_head.logpdf(batch_Y, params)
                all_losses.append(individual_losses.cpu())
            # Concatenate all individual losses
            all_losses = torch.cat(all_losses, dim=0)
            # Calculate standard deviation
            loss_std = torch.std(all_losses).item()
            loss_mean = torch.mean(all_losses).item()
            return loss_mean, loss_std * np.sqrt(1 / X.shape[0])

    train_loss, train_loss_std = eval_loss(train_X, train_Y, batch_size)
    dev_loss, dev_loss_std = eval_loss(dev_X, dev_Y, batch_size)
    if verbose:
        print(f"Init, Train Loss: {train_loss:.4f}, Dev Loss: {dev_loss:.4f}, Dev Loss confidence interval: {dev_loss - 2 * dev_loss_std:.4f}, {dev_loss + 2 * dev_loss_std:.4f}")

    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)

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

        # Step learning rate decay after each step
        scheduler.step()

        #print(f"Step {step}, Total Loss: {total_loss / train_X.shape[0]:.4f}")

        if step % 10 == 0:
            train_loss, train_loss_std = eval_loss(train_X, train_Y, batch_size)
            dev_loss, dev_loss_std = eval_loss(dev_X, dev_Y, batch_size)
            train_losses.append(train_loss)
            dev_losses.append(dev_loss)
            loss_steps.append(step)
            if output_folder is not None:
                # save model
                torch.save(model.state_dict(), os.path.join(output_folder, f"model_{step}.pth"))

            if verbose:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Step {step}, Train Loss: {train_loss:.4f}, Dev Loss: {dev_loss:.4f}, Dev Loss confidence interval: {dev_loss - 2 * dev_loss_std:.4f}, {dev_loss + 2 * dev_loss_std:.4f}, LR: {current_lr:.6f}")
    # save losses to csv
    df = pd.DataFrame({"epoch": loss_steps, "train_loss": train_losses, "dev_loss": dev_losses})
    df.to_csv(os.path.join(output_folder, "losses.csv"), index=False)
    # save hyperparameters to json
    with open(os.path.join(output_folder, "hyperparameters.json"), "w") as f:
        json.dump({"lr": lr, "weight_decay": weight_decay, "num_steps": num_steps, "batch_size": batch_size, "lr_decay_step": lr_decay_step, "lr_decay_gamma": lr_decay_gamma}, f)
    
    # evaluate model on test data
    final_train_loss, final_train_loss_std = eval_loss(train_X, train_Y, batch_size)
    final_dev_loss, final_dev_loss_std = eval_loss(dev_X, dev_Y, batch_size)
    test_loss, test_loss_std = eval_loss(test_X, test_Y, batch_size)
    print(f"Final Train Loss: {final_train_loss:.4f}, Final Train Loss confidence interval: {final_train_loss - 2 * final_train_loss_std:.4f}, {final_train_loss + 2 * final_train_loss_std:.4f}")
    print(f"Final Dev Loss: {final_dev_loss:.4f}, Final Dev Loss confidence interval: {final_dev_loss - 2 * final_dev_loss_std:.4f}, {final_dev_loss + 2 * final_dev_loss_std:.4f}")
    print(f"Test Loss: {test_loss:.4f}, Test Loss confidence interval: {test_loss - 2 * test_loss_std:.4f}, {test_loss + 2 * test_loss_std:.4f}")
    # save final losses to csv
    final_df = pd.DataFrame({"data_type": ["train", "dev", "test"], "loss": [final_train_loss, final_dev_loss, test_loss], "loss_std": [final_train_loss_std, final_dev_loss_std, test_loss_std]})
    final_df.to_csv(os.path.join(output_folder, "final_losses.csv"), index=False)

    if output_folder is not None:
        # save model
        torch.save(model.state_dict(), os.path.join(output_folder, f"final_model.pth"))

    return model, train_losses, dev_losses