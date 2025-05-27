import pandas as pd
import yfinance as yf
from io import StringIO
import requests
from datetime import datetime, timedelta
import os # Added import
from tvDatafeed import TvDatafeed, Interval # Added import

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
        response.raise_for_status()
        html = pd.read_html(StringIO(response.text))
        table = html[0]
        tickers = table['Symbol'].tolist()
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        tickers = [t.replace('BF-B', 'BF.B') if t == 'BF-B' else t for t in tickers]
        print(f"Successfully fetched {len(tickers)} S&P 500 tickers from Wikipedia.")
        if not tickers:
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

    Args:
        ticker (str): The stock ticker symbol.
        start_date (str, optional): Start date in 'YYYY-MM-DD' format.
        end_date (str, optional): End date in 'YYYY-MM-DD' format.
        n_bars (int, optional): Number of trading bars (days) to fetch.
                                  For yfinance, if start_date is not specified, n_bars from end_date is used.
                                  For TvDatafeed, n_bars is used directly if start_date is not specified.
                                  If start_date is specified for TvDatafeed, n_bars is estimated.

    Returns:
        pandas.DataFrame: DataFrame with historical data, or None if fetching fails from all sources.
    """
    print(f"Fetching data for {ticker}...")
    
    # Attempt 1: yfinance
    print(f"Attempting to fetch data for {ticker} using yfinance...")
    try:
        stock_yf = yf.Ticker(ticker)
        data_yf = None
        if start_date:
            data_yf = stock_yf.history(start=start_date, end=end_date, interval='1d')
        else:
            # yfinance history needs start/end. If only n_bars, calculate start from end.
            actual_end_date = pd.to_datetime(end_date) if end_date else datetime.today()
            # Estimate start date to get roughly n_bars
            # This logic is similar to what was in the original file.
            estimated_calendar_days = int(n_bars * (365.25 / 252.0) + 30) # Added buffer
            actual_start_date = actual_end_date - timedelta(days=estimated_calendar_days)
            data_yf = stock_yf.history(start=actual_start_date.strftime('%Y-%m-%d'), 
                                       end=actual_end_date.strftime('%Y-%m-%d'), 
                                       interval='1d')
            if not data_yf.empty and len(data_yf) > n_bars:
                data_yf = data_yf.iloc[-n_bars:]
        
        if data_yf is not None and not data_yf.empty:
            print(f"Successfully fetched data for {ticker} using yfinance. Shape: {data_yf.shape}")
            data_yf.columns = [col.lower() for col in data_yf.columns]
            # Select common columns to match TvDatafeed potential output and simplify
            common_cols = ['open', 'high', 'low', 'close', 'volume']
            data_yf = data_yf[[col for col in common_cols if col in data_yf.columns]]
            return data_yf
        else:
            print(f"No data found for {ticker} using yfinance with given parameters.")
    except Exception as e:
        print(f"yfinance failed for {ticker}: {e}")

    # Attempt 2: TvDatafeed
    print(f"Attempting to fetch data for {ticker} using TvDatafeed...")
    tv_user = os.getenv('TV_USERNAME') # Assuming these are set if TvDatafeed login is needed
    tv_pass = os.getenv('TV_PASSWORD')
    
    tv_n_bars = n_bars # Default for TvDatafeed
    if start_date:
        # If start_date is given, estimate n_bars for TvDatafeed
        # This is a rough estimation, as TvDatafeed's get_hist primarily uses n_bars from present.
        s_dt = pd.to_datetime(start_date)
        e_dt = pd.to_datetime(end_date) if end_date else datetime.today()
        # Calculate business days, roughly. This doesn't account for holidays.
        # A more robust way would be to use pandas_market_calendars if precision is critical.
        tv_n_bars = np.busday_count(s_dt.date(), e_dt.date())
        if tv_n_bars <=0 : tv_n_bars = n_bars # Fallback if calculation is off
        print(f"Calculated n_bars for TvDatafeed based on date range: {tv_n_bars}")


    try:
        if tv_user and tv_pass:
            tv = TvDatafeed(username=tv_user, password=tv_pass)
        else:
            tv = TvDatafeed() # Guest session

        df_tv = None
        # S&P 500 stocks are typically on NASDAQ or NYSE
        exchanges_to_try = ['NASDAQ', 'NYSE', 'AMEX'] 
        for exch in exchanges_to_try:
            try:
                print(f"Trying {ticker} on {exch} with TvDatafeed (n_bars={tv_n_bars})...")
                df_tv_temp = tv.get_hist(symbol=ticker, exchange=exch, interval=Interval.in_daily, n_bars=tv_n_bars)
                if df_tv_temp is not None and not df_tv_temp.empty:
                    print(f"Successfully fetched data for {ticker} from {exch} using TvDatafeed. Shape: {df_tv_temp.shape}")
                    df_tv = df_tv_temp
                    break 
            except Exception as e_tv_exch:
                # More specific error checking might be needed based on TvDatafeed library's exceptions
                msg = str(e_tv_exch).lower()
                if "not found" in msg or "unknown symbol" in msg or "timeout" in msg: # Added timeout
                    print(f"TvDatafeed: {ticker} not found on {exch} or timeout. Trying next exchange.")
                    continue
                else:
                    print(f"TvDatafeed error for {ticker} on {exch}: {e_tv_exch}. Stopping TvDatafeed attempts for this ticker.")
                    break # Non-recoverable error for this ticker with TvDatafeed
        
        if df_tv is not None and not df_tv.empty:
            # TvDatafeed columns might include 'symbol'. We need 'open', 'high', 'low', 'close', 'volume'.
            # Standardize to lowercase and select common columns.
            df_tv.columns = [col.lower() for col in df_tv.columns]
            common_cols = ['open', 'high', 'low', 'close', 'volume']
            df_tv = df_tv[[col for col in common_cols if col in df_tv.columns]]
            
            # Ensure index is datetime
            if not isinstance(df_tv.index, pd.DatetimeIndex):
                df_tv.index = pd.to_datetime(df_tv.index)

            # If start_date was specified, TvDatafeed might return more bars than up to start_date. Filter it.
            if start_date:
                df_tv = df_tv[df_tv.index >= pd.to_datetime(start_date)]
            if end_date: # Also filter by end_date if provided
                 df_tv = df_tv[df_tv.index <= pd.to_datetime(end_date)]

            if not df_tv.empty:
                print(f"Data for {ticker} from TvDatafeed after processing. Shape: {df_tv.shape}")
                return df_tv
            else:
                print(f"TvDatafeed data for {ticker} became empty after date filtering.")
                return None
        else:
            print(f"TvDatafeed could not find data for {ticker} on tried exchanges with n_bars={tv_n_bars}.")
            return None
    except Exception as e_tv:
        print(f"TvDatafeed overall failed for {ticker}: {e_tv}")
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
            # Sanitize ticker for filename if it contains characters like '/' (e.g. 'BRK.B' vs 'BRK-B')
            # yfinance usually handles this, but good practice if tickers could be arbitrary
            safe_ticker_fname = ticker.replace('/', '_') 
            csv_path = os.path.join(CSV_DATA_DIR, f"{safe_ticker_fname}.csv")
            df.to_csv(csv_path, index=True) # index=True to save the DatetimeIndex
            print(f"Successfully saved data for {ticker} to {csv_path}")
            return True
        except Exception as e:
            print(f"Error saving CSV for {ticker}: {e}")
            return False
    else:
        # fetch_stock_data would have printed the reason
        print(f"No data fetched for {ticker}, so not saving CSV.")
        return False

def download_all_sp500_sequential(n_bars_data=1200, start_date_all=None, end_date_all=None, max_tickers=None):
    """
    Downloads data for all S&P 500 tickers sequentially and saves to CSV.
    Args:
        n_bars_data: Number of bars if start_date_all is not specified.
        start_date_all: Global start date for all tickers.
        end_date_all: Global end date for all tickers.
        max_tickers: Max number of tickers to process (for testing).
    """
    tickers = get_sp500_tickers()
    if max_tickers is not None:
        tickers = tickers[:max_tickers]
        print(f"Processing a subset of {max_tickers} tickers.")
    
    print(f"Starting download for {len(tickers)} S&P 500 tickers...")
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
        
        # Optional: add a small delay to be polite to APIs
        # import time
        # time.sleep(0.5) # 0.5 second delay

    print(f"\n--- Download Process Complete ---")
    print(f"Successfully downloaded and saved data for {success_count} tickers.")
    print(f"Failed to download or save data for {failure_count} tickers.")
    print(f"Data saved in directory: '{CSV_DATA_DIR}'")
    print("---------------------------------")

if __name__ == '__main__':
    print("--- Starting S&P 500 Data Download Script ---")
    
    # Example Usage:
    # 1. Fetch last N bars (e.g., ~5 years)
    # download_all_sp500_sequential(n_bars_data=1260) 

    # 2. Fetch data for a specific date range
    # download_all_sp500_sequential(start_date_all='2020-01-01', end_date_all='2023-12-31')

    # 3. Fetch last N bars for a limited number of tickers for testing
    download_all_sp500_sequential(n_bars_data=252, max_tickers=10) # Approx 1 year for 10 tickers

    print("\n--- S&P 500 Data Download Script Finished ---")
