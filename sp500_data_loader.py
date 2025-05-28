import pandas as pd
import yfinance as yf
from io import StringIO
import requests
from datetime import datetime, timedelta
import os # Added import
import time # Ensure time is imported
from tvDatafeed import TvDatafeed, Interval # Added import
import numpy as np # Added for np.busday_count

# Fallback list of S&P 500 tickers
FALLBACK_SP500_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'BRK-B', 'JPM', 'JNJ', 'V',
    'PG', 'UNH', 'HD', 'MA', 'BAC', 'DIS', 'ADBE', 'PYPL', 'NFLX', 'CRM'
]

# Directory for saving CSV data
CSV_DATA_DIR = "sp500_csv_data"

def get_sp500_tickers():
    """
    Fetches the list of S&P 500 tickers from Wikipedia.
    Falls back to a hardcoded list if fetching fails.
    """
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        response = requests.get(url, headers={'User-agent': 'Mozilla/5.0'})
        response.raise_for_status() # Will raise an HTTPError for bad responses (4XX or 5XX)
        html = pd.read_html(StringIO(response.text))
        table = html[0]
        tickers = table['Symbol'].tolist()
        # Clean tickers: Some symbols on Wikipedia might have suffixes like '.B' or '.BF.B'
        # yfinance usually prefers them without such suffixes for common stocks, or with '-' e.g. 'BRK-B', 'BF.B'
        tickers = [ticker.replace('.', '-') for ticker in tickers] # Replace . with - first
        tickers = [t.replace('BF-B', 'BF.B') if t == 'BF-B' else t for t in tickers] # Specific fix for BF.B if needed by yfinance
        # Add other specific replacements if found necessary for yfinance/TvDatafeed
        print(f"Successfully fetched {len(tickers)} S&P 500 tickers from Wikipedia.")
        if not tickers: # Fallback if the list is empty for some reason
            print("Fetched ticker list from Wikipedia is empty. Using fallback list.")
            tickers = FALLBACK_SP500_TICKERS
        return tickers
    except requests.exceptions.RequestException as e_req:
        print(f"Could not fetch S&P 500 tickers from Wikipedia (requests error): {e_req}. Using fallback list.")
        return FALLBACK_SP500_TICKERS
    except Exception as e:
        print(f"Could not fetch S&P 500 tickers from Wikipedia (other error): {e}. Using fallback list.")
        return FALLBACK_SP500_TICKERS

def fetch_stock_data(ticker, start_date=None, end_date=None, n_bars=1200):
    """
    Fetches daily historical stock data for a given ticker, trying yfinance first,
    then TvDatafeed as a fallback.
    """
    print(f"Fetching data for {ticker}...")
    
    # Attempt 1: yfinance
    print(f"Attempting to fetch data for {ticker} using yfinance...")
    data_yf = None
    try:
        stock_yf = yf.Ticker(ticker)
        if start_date:
            # Ensure end_date is set if start_date is
            actual_end_date_yf = pd.to_datetime(end_date).strftime('%Y-%m-%d') if end_date else datetime.today().strftime('%Y-%m-%d')
            data_yf = stock_yf.history(start=pd.to_datetime(start_date).strftime('%Y-%m-%d'), end=actual_end_date_yf, interval='1d')
        else:
            actual_end_date_yf = pd.to_datetime(end_date) if end_date else datetime.today()
            estimated_calendar_days = int(n_bars * (365.25 / 252.0) + 45) # Increased buffer
            actual_start_date_yf = actual_end_date_yf - timedelta(days=estimated_calendar_days)
            data_yf = stock_yf.history(start=actual_start_date_yf.strftime('%Y-%m-%d'), 
                                       end=actual_end_date_yf.strftime('%Y-%m-%d'), 
                                       interval='1d')
            if not data_yf.empty: # Ensure trimming only if data is not empty
                 # Trim to n_bars only if more rows are fetched than n_bars
                if len(data_yf) > n_bars:
                    data_yf = data_yf.iloc[-n_bars:]
        
        if data_yf is not None and not data_yf.empty:
            print(f"Successfully fetched data for {ticker} using yfinance. Shape: {data_yf.shape}")
            data_yf.columns = [col.lower() for col in data_yf.columns]
            common_cols = ['open', 'high', 'low', 'close', 'volume']
            # Ensure all common_cols exist, fill with NaN or 0 if not (though yfinance usually has them)
            for col in common_cols:
                if col not in data_yf.columns:
                    data_yf[col] = 0 if col == 'volume' else np.nan 
            return data_yf[common_cols]
        else:
            print(f"No data found for {ticker} using yfinance with given parameters.")
    except Exception as e:
        # Catch more specific errors if possible, or check error message content
        print(f"yfinance failed for {ticker}: {e}")
        if "No timezone found" in str(e) or "Failed to get ticker" in str(e) or "symbol may be delisted" in str(e).lower():
             print(f"yfinance indicated potential delisting or symbol issue for {ticker}.")
        # Do not return here, proceed to TvDatafeed

    # Attempt 2: TvDatafeed
    print(f"Attempting to fetch data for {ticker} using TvDatafeed...")
    tv_user = os.getenv('TV_USERNAME')
    tv_pass = os.getenv('TV_PASSWORD')
    
    tv_n_bars_final = n_bars 
    if start_date:
        s_dt = pd.to_datetime(start_date)
        e_dt = pd.to_datetime(end_date) if end_date else datetime.today()
        # Calculate approximate number of trading days. np.busday_count is good.
        # It needs dates, not datetimes for simpler usage here.
        num_days = np.busday_count(s_dt.date(), e_dt.date())
        if num_days > 0 : 
            tv_n_bars_final = num_days
        else: # if date range is too small or inverted, fall back to default n_bars
            print(f"Warning: Calculated n_bars for TvDatafeed is {num_days} for {ticker}. Using default {n_bars}.")
            tv_n_bars_final = n_bars


    try:
        tv = TvDatafeed(username=tv_user, password=tv_pass) if tv_user and tv_pass else TvDatafeed()
        df_tv = None
        exchanges_to_try = ['NASDAQ', 'NYSE', 'AMEX', 'OTC', 'ARCA'] # Added more exchanges
        for exch in exchanges_to_try:
            try:
                print(f"Trying {ticker} on {exch} with TvDatafeed (n_bars={tv_n_bars_final})...")
                df_tv_temp = tv.get_hist(symbol=ticker, exchange=exch, interval=Interval.in_daily, n_bars=tv_n_bars_final)
                if df_tv_temp is not None and not df_tv_temp.empty:
                    print(f"Successfully fetched data for {ticker} from {exch} using TvDatafeed. Shape: {df_tv_temp.shape}")
                    df_tv = df_tv_temp
                    break 
            except Exception as e_tv_exch:
                msg = str(e_tv_exch).lower()
                if "not found" in msg or "unknown symbol" in msg or "timeout" in msg or "empty" in msg:
                    print(f"TvDatafeed: {ticker} not found on {exch} or other issue. Trying next exchange.")
                    continue
                else:
                    print(f"TvDatafeed non-recoverable error for {ticker} on {exch}: {e_tv_exch}")
                    break 
        
        if df_tv is not None and not df_tv.empty:
            df_tv.columns = [col.lower() for col in df_tv.columns]
            common_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in common_cols: # Ensure columns exist
                if col not in df_tv.columns:
                    df_tv[col] = 0 if col == 'volume' else np.nan
            df_tv = df_tv[common_cols]
            
            if not isinstance(df_tv.index, pd.DatetimeIndex):
                df_tv.index = pd.to_datetime(df_tv.index)
            df_tv = df_tv.sort_index() # Ensure chronological order

            # Filter by date range if start_date was provided, as TvDatafeed's n_bars is from present
            if start_date:
                df_tv = df_tv[df_tv.index >= pd.to_datetime(start_date)]
            if end_date:
                 df_tv = df_tv[df_tv.index <= pd.to_datetime(end_date)]
            # If using n_bars primarily and dates were for yfinance, ensure TvDatafeed result is also trimmed if needed
            # This logic can get complex if mixing n_bars and date ranges across APIs strictly
            # For now, if start_date was provided, the above filter is primary for TvDatafeed.
            # If only n_bars was goal, TvDatafeed's n_bars parameter is used.

            if not df_tv.empty:
                print(f"Data for {ticker} from TvDatafeed after processing. Shape: {df_tv.shape}")
                return df_tv
            else:
                print(f"TvDatafeed data for {ticker} became empty after date filtering.")
                return None
        else:
            print(f"TvDatafeed could not find data for {ticker} on tried exchanges.")
            return None
    except Exception as e_tv:
        print(f"TvDatafeed overall failed for {ticker}: {e_tv}") # This can include login errors
        return None

    print(f"All data fetching attempts failed for {ticker}.")
    return None

def download_and_save_ticker_data(ticker, start_date=None, end_date=None, n_bars=1200):
    """
    Fetches stock data using the hybrid strategy and saves it to a CSV file.
    """
    df = fetch_stock_data(ticker, start_date, end_date, n_bars)
    if df is not None and not df.empty:
        try:
            os.makedirs(CSV_DATA_DIR, exist_ok=True)
            safe_ticker_fname = ticker.replace('/', '_').replace('.', '_') # Further sanitize for filesystem
            csv_path = os.path.join(CSV_DATA_DIR, f"{safe_ticker_fname}.csv")
            df.to_csv(csv_path, index=True)
            print(f"Successfully saved data for {ticker} to {csv_path}")
            return True
        except Exception as e:
            print(f"Error saving CSV for {ticker}: {e}")
            return False
    else:
        print(f"No data fetched for {ticker}, so not saving CSV.")
        return False

def download_all_sp500_sequential(n_bars_data=1200, start_date_all=None, end_date_all=None, max_tickers=None):
    """
    Downloads data for S&P 500 tickers sequentially and saves to CSV.
    """
    tickers = get_sp500_tickers()
    if max_tickers is not None and max_tickers > 0 : # ensure max_tickers is positive
        tickers = tickers[:max_tickers]
        print(f"Processing a subset of {len(tickers)} tickers (max_tickers={max_tickers}).")
    
    print(f"Starting download for {len(tickers)} S&P 500 tickers...")
    if start_date_all and end_date_all:
        print(f"Using date range: {start_date_all} to {end_date_all}")
    else:
        print(f"Using n_bars: {n_bars_data}")

    success_count = 0
    failure_count = 0
    
    for i, ticker_symbol in enumerate(tickers):
        print(f"\nProcessing ticker {i+1}/{len(tickers)}: {ticker_symbol}")
        if download_and_save_ticker_data(ticker_symbol, 
                                         start_date=start_date_all, 
                                         end_date=end_date_all, 
                                         n_bars=n_bars_data):
            success_count += 1
        else:
            failure_count += 1
        
        import time # Recommended for long runs; uncommented.
        time.sleep(0.2) # Small delay to be polite to APIs; uncommented. Consider increasing if issues arise.

    print(f"\n--- Download Process Complete ---")
    print(f"Successfully downloaded and saved data for {success_count} tickers.")
    print(f"Failed to download or save data for {failure_count} tickers.")
    print(f"Data saved in directory: '{CSV_DATA_DIR}'")
    print("---------------------------------")

if __name__ == '__main__':
    print("--- Starting S&P 500 Data Download Script ---")
    
    # --- Option 1: Fetch last N bars (e.g., ~5 years) for a subset of tickers ---
    # download_all_sp500_sequential(n_bars_data=1260, max_tickers=10) 

    # --- Option 2: Fetch data for a specific date range for a subset of tickers ---
    # download_all_sp500_sequential(start_date_all='2022-01-01', end_date_all='2023-12-31', max_tickers=10)

    # --- Option 3: Fetch last N bars for ALL tickers (potentially long running!) ---
    # download_all_sp500_sequential(n_bars_data=1260) 

    # Default test: Fetch approx 1 year for 5 tickers for quick testing.
    # --- Default behavior: Fetch data for ALL S&P 500 tickers ---
    # This will download approximately 5 years of daily data (n_bars_data=1260)
    # for all tickers fetched from Wikipedia (or the fallback list).
    # This can be a long-running process (potentially hours).
    #
    # To be polite to APIs and reduce the risk of being rate-limited,
    # it's recommended to uncomment the 'import time' statement at the top of the script (if not already present)
    # and uncomment the 'time.sleep(0.2)' line within the 'download_all_sp500_sequential' function.
    # You might consider increasing the sleep duration (e.g., to 0.5 or 1 second) if you encounter issues.
    download_all_sp500_sequential(n_bars_data=1260, max_tickers=None)

    print("\n--- S&P 500 Data Download Script Finished ---")