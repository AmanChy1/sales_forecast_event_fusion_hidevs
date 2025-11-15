import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- 1. Set up the UI Title and Description ---
st.set_page_config(layout="wide")
st.title("Weekly Sales Forecasting App")
st.markdown("Use the controls below to select a Store and Department to generate a Holt-Winters sales forecast.")

# --- 2. Data Loading (Cached for Performance) ---
@st.cache_data
def load_data():
    try:
        train = pd.read_csv("train.csv")
        stores = pd.read_csv("stores.csv")
        features = pd.read_csv("features.csv")
    except FileNotFoundError:
        st.error("Missing data files. Please ensure 'train.csv', 'stores.csv', and 'features.csv' are in the same directory.")
        return None, None, None, None

    # parse dates
    train["Date"] = pd.to_datetime(train["Date"])
    features["Date"] = pd.to_datetime(features["Date"])

    # merge (keep as in original)
    data = train.merge(features, on=["Store", "Date", "IsHoliday"], how="left")
    data = data.merge(stores, on="Store", how="left")
    data = data.sort_values(["Store", "Dept", "Date"])

    return data, train, stores, features

data, train_data, stores_data, features_data = load_data()

if data is not None:

    # --- 3. Sidebar Widgets for User Input ---
    with st.sidebar:
        st.header("Model Configuration")

        all_stores = sorted(data["Store"].unique())
        all_depts  = sorted(data["Dept"].unique())

        selected_store = st.selectbox("Select Store ID", all_stores, index=0)
        selected_dept  = st.selectbox("Select Department ID", all_depts, index=0)

        future_periods = st.slider("Future Weeks to Forecast", min_value=1, max_value=104, value=30, step=1)
        seasonal_periods = st.number_input("Seasonal Periods (Weeks)", min_value=1, value=52)

        run_button = st.button("Generate Forecast")


    # --- 4. Forecasting Function ---
    @st.cache_data
    def generate_forecast(data, store_id, dept_id, future_periods, seasonal_periods):
        """
        Returns: (ts, future_forecast_series, forecast_df, message)
        If an error happens, ts and future_forecast_series and forecast_df will be None and message explains the issue.
        """
        # Filter
        sd = data[(data["Store"] == store_id) & (data["Dept"] == dept_id)].copy()
        if sd.empty:
            return None, None, None, f"No data for Store {store_id}, Dept {dept_id}."

        sd = sd.sort_values("Date")

        # Resample to weekly (ensures regular freq for forecasting)
        # Use the raw Weekly_Sales column (summing in case duplicates per date)
        ts = sd.set_index("Date")["Weekly_Sales"].resample("W").sum()

        # Drop any leading/trailing NaNs if present
        ts = ts.dropna()

        # Validate enough data
        if len(ts) < max(2 * seasonal_periods, seasonal_periods + 5):
            msg = (
                f"Insufficient data after resampling. "
                f"Have {len(ts)} weekly points; need at least {2 * seasonal_periods} for the chosen seasonal_periods={seasonal_periods}."
            )
            return None, None, None, msg

        # Fit model
        try:
            hw_full = ExponentialSmoothing(
                ts,
                trend="add",
                seasonal="add",
                seasonal_periods=int(seasonal_periods)
            ).fit()

            # Forecast: this returns a Series with a DatetimeIndex if ts has a freq
            future_forecast = hw_full.forecast(future_periods)

            # Prepare forecast dataframe for display/download
            forecast_df = future_forecast.reset_index()
            forecast_df.columns = ["Date", "Forecast_Weekly_Sales"]
            forecast_df["Date"] = forecast_df["Date"].dt.strftime("%Y-%m-%d")

            msg = f"Successfully generated a {future_periods}-week forecast for Store {store_id}, Dept {dept_id}."
            return ts, future_forecast, forecast_df, msg

        except Exception as e:
            return None, None, None, f"An error occurred during model fitting: {e}"


    # --- 5. Main App Logic ---
    if run_button:
        st.subheader(f"Results for Store {selected_store}, Department {selected_dept}")

        ts_historical, future_forecast_series, forecast_table_df, status_message = generate_forecast(
            data, selected_store, selected_dept, future_periods, seasonal_periods
        )

        if ts_historical is None:
            st.error(status_message)
        else:
            st.success(status_message)

            # --- Visualization ---
            st.subheader("Forecast Visualization")
            fig, ax = plt.subplots(figsize=(10, 5))

            # historical
            ax.plot(ts_historical.index, ts_historical.values, label="Historical Sales")
            # forecast
            ax.plot(future_forecast_series.index, future_forecast_series.values, linestyle="--", label="Forecasted Sales")

            ax.set_title(f"Historical and Future Sales Forecast — Store {selected_store}, Dept {selected_dept}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Weekly Sales")
            ax.legend()
            st.pyplot(fig)

            # --- Data Table and Download ---
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Forecast Data (Table)")
                st.dataframe(forecast_table_df)

            with col2:
                st.subheader("Download Forecast")
                csv_file = forecast_table_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Forecast as CSV",
                    data=csv_file,
                    file_name=f"forecast_store{selected_store}_dept{selected_dept}.csv",
                    mime="text/csv",
                )
