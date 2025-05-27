import pandas as pd
import yfinance as yf
from io import StringIO # Required for fallback if requests fail for Wikipedia
import requests # For fetching HTML from Wikipedia
from datetime import datetime, timedelta # For date calculations in fetch_stock_data

# Fallback list of S&P 500 tickers (subset for easier testing, can be expanded)
# Real implementation should ideally fetch this dynamically.
FALLBACK_SP500_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'BRK-B', 'JPM', 'JNJ', 'V',
    'PG', 'UNH', 'HD', 'MA', 'BAC', 'DIS', 'ADBE', 'PYPL', 'NFLX', 'CRM' # Approx 20 for testing
]

def get_sp500_tickers():
    """
    Fetches the list of S&P 500 tickers from Wikipedia.
    Falls back to a hardcoded list if fetching fails.
    """
    try:
        # Attempt to fetch from Wikipedia
        # Using a known reliable source for S&P 500 list
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        # Use pandas to read HTML tables. The S&P 500 tickers are usually in the first table.
        # Adding a User-Agent to avoid potential HTTP 403 Forbidden errors
        response = requests.get(url, headers={'User-agent': 'Mozilla/5.0'})
        response.raise_for_status() # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
        html = pd.read_html(StringIO(response.text))
        table = html[0]
        tickers = table['Symbol'].tolist()
        # Clean tickers: Some symbols on Wikipedia might have suffixes like '.B' or '.BF.B'
        # yfinance usually prefers them without such suffixes for common stocks, or with '-' e.g. 'BRK-B'
        # For now, we'll do a basic replacement, this might need refinement.
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        # Specific known replacements for yfinance compatibility
        tickers = [t.replace('BF-B', 'BF.B') if t == 'BF-B' else t for t in tickers] # e.g. Brown-Forman

        print(f"Successfully fetched {len(tickers)} S&P 500 tickers from Wikipedia.")
        if not tickers: # Fallback if the list is empty for some reason
            print("Fetched ticker list from Wikipedia is empty. Using fallback list.")
            tickers = FALLBACK_SP500_TICKERS
        return tickers
    except requests.exceptions.RequestException as e_req:
        print(f"Could not fetch S&P 500 tickers from Wikipedia due to a requests error: {e_req}. Using fallback list.")
        return FALLBACK_SP500_TICKERS
    except Exception as e:
        print(f"Could not fetch S&P 500 tickers from Wikipedia due to: {e}. Using fallback list.")
        return FALLBACK_SP500_TICKERS

def fetch_stock_data(ticker, start_date=None, end_date=None, n_bars=1200):
    """
    Fetches daily historical stock data for a given ticker using yfinance.

    Args:
        ticker (str): The stock ticker symbol.
        start_date (str, optional): Start date in 'YYYY-MM-DD' format.
        end_date (str, optional): End date in 'YYYY-MM-DD' format.
        n_bars (int, optional): Number of trading bars (days) to fetch if start_date is not specified.
                                  Ignored if start_date is provided.

    Returns:
        pandas.DataFrame: DataFrame with historical data, or None if fetching fails.
    """
    try:
        stock = yf.Ticker(ticker)
        if start_date:
            data = stock.history(start=start_date, end=end_date, interval='1d')
        else:
            # If no start_date, use n_bars.
            if end_date:
                end_dt = pd.to_datetime(end_date)
            else:
                end_dt = datetime.today()
            
            # Estimate start date based on n_bars (approx. 252 trading days a year)
            # Add some buffer to account for non-trading days (weekends, holidays)
            # Roughly 365.25/252 ratio for calendar days per trading day.
            estimated_calendar_days = int(n_bars * (365.25 / 252.0) + 15) # Add a small buffer of 15 days
            start_dt = end_dt - timedelta(days=estimated_calendar_days)
            
            data = stock.history(start=start_dt.strftime('%Y-%m-%d'), end=end_dt.strftime('%Y-%m-%d'), interval='1d')
            
            if not data.empty and len(data) > n_bars: # Trim if we got more than n_bars
                data = data.iloc[-n_bars:]
            elif not data.empty and len(data) < n_bars:
                 print(f"Warning: Fetched {len(data)} bars for {ticker}, requested {n_bars}. Data might be shorter than expected due to listing date or data availability.")


        if data.empty:
            print(f"No data found for {ticker} for the given parameters.")
            return None
        
        # Standardize column names to lowercase for easier access
        data.columns = [col.lower() for col in data.columns]
        print(f"Successfully fetched data for {ticker}. Shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Could not fetch data for {ticker} due to: {e}")
        return None

if __name__ == '__main__':
    # Test functions
    print("Testing S&P 500 ticker fetching...")
    tickers = get_sp500_tickers()
    print(f"First 5 tickers: {tickers[:5]}")
    print(f"Total tickers: {len(tickers)}")

    if tickers:
        print(f"\nTesting data fetching for a few tickers (e.g., {tickers[:3]}):")
        for ticker_symbol in tickers[:3]: # Test with first 3 tickers from the fetched list
            stock_df = fetch_stock_data(ticker_symbol, n_bars=252) # Approx 1 year
            if stock_df is not None:
                print(f"Data for {ticker_symbol}:")
                print(stock_df.head())
            else:
                print(f"Failed to get data for {ticker_symbol}")
    
    print("\nTesting with a specific ticker known to have issues with '.' -> '-' (e.g. BRK.B vs BRK-B)")
    # yfinance uses BRK-B for Berkshire Hathaway Class B
    brk_data = fetch_stock_data('BRK-B', n_bars=100)
    if brk_data is not None:
        print("BRK-B data fetched successfully.")
        print(brk_data.head())
    else:
        print("Failed to fetch BRK-B data.")

    print("\nTesting with 'BF.B' (Brown-Forman Corp Class B)")
    bfb_data = fetch_stock_data('BF.B', n_bars=100)
    if bfb_data is not None:
        print("BF.B data fetched successfully.")
        print(bfb_data.head())
    else:
        print("Failed to fetch BF.B data.")


    print("\nTesting with a non-existent ticker:")
    non_existent_data = fetch_stock_data('NONEXISTENTTICKERXYZ', n_bars=100)
    if non_existent_data is None:
        print("Correctly handled non-existent ticker (returned None).")
    else:
        print("Error: Non-existent ticker did not return None.")
    
    print("\nTesting fetch_stock_data with start_date and end_date:")
    aapl_data_range = fetch_stock_data('AAPL', start_date='2023-01-01', end_date='2023-01-31')
    if aapl_data_range is not None:
        print("AAPL data for Jan 2023 fetched successfully.")
        print(aapl_data_range.head())
        print(aapl_data_range.tail())
    else:
        print("Failed to fetch AAPL data for Jan 2023.")

    print("\nTesting fetch_stock_data with n_bars and end_date:")
    msft_data_nbars_end = fetch_stock_data('MSFT', end_date='2023-12-31', n_bars=60)
    if msft_data_nbars_end is not None:
        print(f"MSFT data for 60 bars ending 2023-12-31 fetched successfully. Shape: {msft_data_nbars_end.shape}")
        print(msft_data_nbars_end.head())
        print(msft_data_nbars_end.tail())
        # Verify the end date is close to specified
        if not msft_data_nbars_end.empty:
             print(f"Last date in data: {msft_data_nbars_end.index[-1]}")
    else:
        print("Failed to fetch MSFT data with n_bars and end_date.")

    print("\nAll tests in sp500_data_loader.py completed.")
