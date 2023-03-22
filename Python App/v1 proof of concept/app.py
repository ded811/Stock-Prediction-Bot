import pandas as pd
import yfinance as yf
from prophet import Prophet
import tkinter as tk

# Create a tkinter window
window = tk.Tk()

# Create labels and entry boxes for the input parameters
tk.Label(window, text="Ticker: ").grid(row=0)
ticker_entry = tk.Entry(window)
ticker_entry.grid(row=0, column=1)

tk.Label(window, text="Timeframe: ").grid(row=1)
timeframe_entry = tk.Entry(window)
timeframe_entry.grid(row=1, column=1)

tk.Label(window, text="Frequency: ").grid(row=2)
freq_entry = tk.Entry(window)
freq_entry.grid(row=2, column=1)

tk.Label(window, text="Length: ").grid(row=3)
length_entry = tk.Entry(window)
length_entry.grid(row=3, column=1)

# Define a function to get the prediction and display the results
def get_prediction():
    # Get the input parameters
    ticker = ticker_entry.get()
    timeframe = timeframe_entry.get()
    freq = freq_entry.get()
    length = int(length_entry.get())

    # Get the stock data from Yahoo Finance
    data = yf.download(ticker, period=timeframe)

    # Check for missing values
    if data.isnull().values.any():
        data = data.dropna()

    # Prepare the data
    data = data[['Close']]
    data = data.reset_index()
    data = data.rename(columns={'Date': 'ds', 'Close': 'y'})

    # Create the model
    model = Prophet()
    model.fit(data)

    # Make a prediction for the next 'length' hours
    future = model.make_future_dataframe(periods=length, freq=freq)
    forecast = model.predict(future)

    # Display the prediction in a new window
    results_window = tk.Toplevel(window)
    results_window.title("Prediction Results")
    tk.Label(results_window, text=forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(length)).grid(row=0)

# Create a button to get the prediction
tk.Button(window, text="Get Prediction", command=get_prediction).grid(row=4, column=1)

# Run the tkinter event loop
window.mainloop()
