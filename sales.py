import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# -----------------------------
# Load and prepare data
# -----------------------------
def load_data():
    try:
        train = pd.read_csv("train.csv")
        stores = pd.read_csv("stores.csv")
        features = pd.read_csv("features.csv")
    except FileNotFoundError as e:
        messagebox.showerror("File error", f"Could not find file: {e}")
        raise

    # Convert dates
    train["Date"] = pd.to_datetime(train["Date"])
    features["Date"] = pd.to_datetime(features["Date"])

    # Merge
    data = train.merge(features, on=["Store", "Date", "IsHoliday"], how="left")
    data = data.merge(stores, on="Store", how="left")

    # Sort
    data = data.sort_values(["Store", "Dept", "Date"])
    return data


data = load_data()

# all unique stores
store_ids = sorted(data["Store"].unique())


# -----------------------------
# Forecasting function
# -----------------------------
def run_forecast():
    try:
        store_val = store_combo.get()
        dept_val = dept_combo.get()
        weeks_val = weeks_entry.get()
    except Exception:
        messagebox.showerror("Input error", "Please select store, dept and weeks.")
        return

    if not store_val or not dept_val:
        messagebox.showerror("Input error", "Please select both Store and Department.")
        return

    try:
        store_id = int(store_val)
        dept_id = int(dept_val)
    except ValueError:
        messagebox.showerror("Input error", "Store and Dept must be numbers.")
        return

    try:
        future_periods = int(weeks_val)
        if future_periods <= 0:
            raise ValueError
    except ValueError:
        messagebox.showerror("Input error", "Weeks to forecast must be a positive integer.")
        return

    # Filter data for selected store & dept
    sd = data[(data["Store"] == store_id) & (data["Dept"] == dept_id)].copy()
    if sd.empty:
        messagebox.showerror("No data", "No rows for this Store/Dept combination.")
        return

    # Create time series
    ts = sd.groupby("Date")["Weekly_Sales"].sum().sort_index()

    # Resample weekly (sum within each week)
    ts = ts.resample("W").sum()

    if len(ts) < 20:
        messagebox.showerror("Not enough data", "Need at least 20 data points for forecasting.")
        return

    seasonal_periods = 52  # yearly seasonality with weekly data

    try:
        model = ExponentialSmoothing(
            ts,
            trend="add",
            damped_trend=True,
            seasonal="add",
            seasonal_periods=seasonal_periods,
        ).fit(
            smoothing_level=0.2,
            smoothing_trend=0.2,
            smoothing_seasonal=0.2,
            optimized=False,
        )
    except Exception as e:
        messagebox.showerror("Model error", f"Error fitting model:\n{e}")
        return

    # Forecast future_periods weeks
    future_forecast = model.forecast(future_periods)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts.index, ts.values, label="Historical")
    ax.plot(future_forecast.index, future_forecast.values, "--", label="Forecast")
    ax.set_title(f"Store {store_id}, Dept {dept_id} — {future_periods} Week Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Weekly Sales")
    ax.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------
# UI helpers
# -----------------------------
def on_store_change(event=None):
    """Update Dept dropdown when Store changes."""
    store_val = store_combo.get()
    if not store_val:
        return
    try:
        sid = int(store_val)
    except ValueError:
        return

    depts = sorted(data[data["Store"] == sid]["Dept"].unique())
    dept_combo["values"] = depts
    if depts:
        dept_combo.current(0)


# -----------------------------
# Build Tkinter window
# -----------------------------
root = tk.Tk()
root.title("Sales Forecasting (Holt-Winters)")
root.geometry("400x250")

# Store selection
tk.Label(root, text="Select Store:").pack(pady=(10, 0))
store_combo = ttk.Combobox(root, values=store_ids, state="readonly")
store_combo.pack()
store_combo.bind("<<ComboboxSelected>>", on_store_change)

# Dept selection
tk.Label(root, text="Select Department:").pack(pady=(10, 0))
dept_combo = ttk.Combobox(root, values=[], state="readonly")
dept_combo.pack()

# Weeks to forecast
tk.Label(root, text="Weeks to forecast:").pack(pady=(10, 0))
weeks_entry = tk.Entry(root)
weeks_entry.insert(0, "30")  # default 30 weeks
weeks_entry.pack()

# Run button
run_button = tk.Button(root, text="Run Forecast", command=run_forecast)
run_button.pack(pady=20)

# Pre-select first store & depts
if store_ids:
    store_combo.current(0)
    on_store_change()

root.mainloop()