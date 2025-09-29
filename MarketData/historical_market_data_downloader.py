#!/usr/bin/env python3
"""
Historical Data Downloader using Polygon API

This script downloads historical minute-level price data for SPY (SPDR S&P 500 ETF Trust)
for the past 2 years using the Polygon API.

Requirements:
- Polygon API key (sign up at https://polygon.io/)
- Python 3.8+
- Required packages: polygon-api-client, pandas, python-dotenv

Usage:
1. Set your POLYGON_API_KEY environment variable or create a .env file
2. Run: python spy_historical_data_downloader.py
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from typing import List, Optional
import pandas as pd
from polygon import RESTClient
from tqdm import tqdm

# Load API key from file
def load_api_key(filename: str = "api_key.txt") -> str:
    """Load API key from text file."""
    try:
        with open(filename, "r") as f:
            api_key = f.read().strip()
        if not api_key:
            raise ValueError("API key file is empty")
        return api_key
    except FileNotFoundError:
        raise FileNotFoundError(f"API key file '{filename}' not found")
    except Exception as e:
        raise Exception(f"Error reading API key file: {str(e)}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('spy_data_download.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class SPYDataDownloader:
    """Downloads historical SPY data from Polygon API with rate limiting and error handling."""
    
    def __init__(self, api_key: str):
        """
        Initialize the downloader with API key.
        
        Args:
            api_key: Polygon API key
        """
        self.client = RESTClient(api_key)
        self.ticker = 'SPY'
        self.rate_limit_delay = 0.1  # 100ms delay between requests
        self.max_retries = 3
        
    def calculate_date_range(self, years_back: int = 2) -> tuple:
        """
        Calculate start and end dates for data download, ensuring only full trading days.
        
        Args:
            years_back: Number of years to go back from today
            
        Returns:
            Tuple of (start_date_str, end_date_str) in YYYY-MM-DD format
        """
        # Get today's date
        today = datetime.now()
        
        # Calculate start date (years_back years ago)
        start_date = today - timedelta(years_back * 365)
        
        # Ensure we start from a Monday (beginning of trading week)
        # Monday is weekday 0, so we adjust to get the Monday of that week
        days_since_monday = start_date.weekday()
        start_date = start_date - timedelta(days=days_since_monday)
        
        # End date should be the last complete trading day (yesterday or earlier)
        # This ensures we don't try to download incomplete data for today
        end_date = today - timedelta(days=1)
        
        # If today is Monday, go back to Friday of previous week
        if today.weekday() == 0:  # Monday
            end_date = today - timedelta(days=3)  # Friday
        elif today.weekday() == 6:  # Sunday
            end_date = today - timedelta(days=2)  # Friday
        elif today.weekday() == 5:  # Saturday
            end_date = today - timedelta(days=1)  # Friday
        
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        logger.info(f"Date range: {start_date_str} to {end_date_str}")
        logger.info(f"Only downloading complete trading days (Monday-Friday)")
        return start_date_str, end_date_str
    
    def fetch_data_with_retry(self, start_date: str, end_date: str) -> List[dict]:
        """
        Fetch data with retry logic and rate limiting.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            List of aggregated data points
        """
        aggs = []
        retry_count = 0
        
        while retry_count < self.max_retries:
            try:
                logger.info(f"Fetching data for {self.ticker} from {start_date} to {end_date}")
                logger.info(f"Attempt {retry_count + 1}/{self.max_retries}")
                
                # Add rate limiting delay
                if retry_count > 0:
                    time.sleep(self.rate_limit_delay * (2 ** retry_count))  # Exponential backoff
                
                # Calculate total days for progress tracking
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                total_days = (end_dt - start_dt).days
                
                # Initialize progress bar
                with tqdm(total=total_days, desc="Downloading data", unit="days", 
                         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} days [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
                    
                    # Fetch data using Polygon API
                    for agg in self.client.list_aggs(
                        ticker=self.ticker,
                        multiplier=1,
                        timespan='minute',
                        from_=start_date,
                        to=end_date,
                        limit=50000,  # Maximum limit per request
                        adjusted=True,  # Adjusted for splits and dividends
                        sort='asc'  # Ascending order by timestamp
                    ):
                        aggs.append({
                            'timestamp': agg.timestamp,
                            'open': agg.open,
                            'high': agg.high,
                            'low': agg.low,
                            'close': agg.close,
                            'volume': agg.volume,
                            'vwap': getattr(agg, 'vwap', None),  # Volume weighted average price
                            'transactions': getattr(agg, 'transactions', None)
                        })
                        
                        # Update progress based on data points (approximate)
                        if len(aggs) % 1000 == 0:  # Update every 1000 data points
                            # Estimate days processed based on data points
                            # Assuming ~390 minutes per trading day (6.5 hours * 60 minutes)
                            estimated_days = min(len(aggs) / 390, total_days)
                            pbar.n = int(estimated_days)
                            pbar.refresh()
                        
                        # Add small delay between individual data points to respect rate limits
                        time.sleep(0.001)
                    
                    # Complete the progress bar
                    pbar.n = total_days
                    pbar.refresh()
                
                logger.info(f"Successfully fetched {len(aggs)} data points")
                break
                
            except Exception as e:
                retry_count += 1
                logger.error(f"Error fetching data (attempt {retry_count}): {str(e)}")
                
                if retry_count >= self.max_retries:
                    logger.error(f"Failed to fetch data after {self.max_retries} attempts")
                    raise
                else:
                    logger.info(f"Retrying in {self.rate_limit_delay * (2 ** retry_count)} seconds...")
                    time.sleep(self.rate_limit_delay * (2 ** retry_count))
        
        return aggs
    
    def filter_market_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter DataFrame to only include market hours (9:30 AM - 4:00 PM ET).
        
        Args:
            df: DataFrame with timestamp index
            
        Returns:
            Filtered DataFrame with only market hours data
        """
        if df.empty:
            return df
        
        # Convert to Eastern Time if needed (Polygon data is in UTC)
        # For simplicity, we'll assume the data is already in ET or convert accordingly
        df_et = df.copy()
        
        # If the data is in UTC, convert to ET
        if df_et.index.tz is None:
            # Assume UTC and convert to ET
            df_et.index = df_et.index.tz_localize('UTC').tz_convert('US/Eastern')
        elif df_et.index.tz.zone != 'US/Eastern':
            # Convert from other timezone to ET
            df_et.index = df_et.index.tz_convert('US/Eastern')
        
        # Filter to market hours (9:30 AM - 4:00 PM ET)
        market_hours_mask = (
            (df_et.index.time >= pd.Timestamp('09:30:00').time()) &
            (df_et.index.time <= pd.Timestamp('16:00:00').time()) &
            (df_et.index.weekday < 5)  # Monday=0, Friday=4
        )
        
        df_filtered = df_et[market_hours_mask].copy()
        
        # Convert back to naive datetime (remove timezone info)
        df_filtered.index = df_filtered.index.tz_localize(None)
        
        logger.info(f"Filtered to market hours: {len(df_filtered)} records remaining out of {len(df)}")
        
        # Log market hours statistics
        if not df_filtered.empty:
            trading_days = df_filtered.index.date
            unique_trading_days = len(set(trading_days))
            logger.info(f"Data covers {unique_trading_days} unique trading days")
            
            # Check for any non-weekday data (should be none after filtering)
            non_weekdays = df_filtered[df_filtered.index.weekday >= 5]
            if len(non_weekdays) > 0:
                logger.warning(f"Found {len(non_weekdays)} records outside weekdays - this shouldn't happen!")
        
        return df_filtered
    
    def process_data(self, raw_data: List[dict]) -> pd.DataFrame:
        """
        Process raw data into a clean DataFrame.
        
        Args:
            raw_data: List of raw data dictionaries
            
        Returns:
            Processed pandas DataFrame
        """
        if not raw_data:
            logger.warning("No data to process")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(raw_data)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        # Sort by timestamp
        df.sort_index(inplace=True)
        
        # Add additional calculated columns
        df['price_change'] = df['close'] - df['open']
        df['price_change_pct'] = (df['close'] - df['open']) / df['open'] * 100
        df['high_low_spread'] = df['high'] - df['low']
        df['high_low_spread_pct'] = (df['high'] - df['low']) / df['low'] * 100
        
        # Remove any duplicate timestamps
        df = df[~df.index.duplicated(keep='first')]
        
        # Filter to market hours only (9:30 AM - 4:00 PM ET, Monday-Friday)
        df = self.filter_market_hours(df)
        
        logger.info(f"Processed {len(df)} data points")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        
        return df
    
    def save_data(self, df: pd.DataFrame, filename: Optional[str] = None) -> str:
        """
        Save DataFrame to CSV file.
        
        Args:
            df: DataFrame to save
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d')
            filename = f'spy_historical_data_{timestamp}.csv'
        
        filepath = os.path.join(os.getcwd(), filename)
        df.to_csv(filepath)
        
        logger.info(f"Data saved to: {filepath}")
        logger.info(f"File size: {os.path.getsize(filepath) / (1024*1024):.2f} MB")
        
        return filepath
    
    def download_historical_data(self, years_back: int = 2, save_to_file: bool = True) -> pd.DataFrame:
        """
        Main method to download and process historical SPY data.
        
        Args:
            years_back: Number of years to go back from today
            save_to_file: Whether to save data to CSV file
            
        Returns:
            Processed DataFrame with historical data
        """
        try:
            # Calculate date range
            start_date, end_date = self.calculate_date_range(years_back)
            
            # Calculate total days for overall progress
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            total_days = (end_dt - start_dt).days
            
            logger.info(f"Starting download of {total_days} days of data...")
            logger.info("Only downloading complete trading days (Monday-Friday, 9:30 AM - 4:00 PM ET)")
            
            # Fetch raw data with progress tracking
            raw_data = self.fetch_data_with_retry(start_date, end_date)
            
            if not raw_data:
                logger.error("No data retrieved")
                return pd.DataFrame()
            
            # Process data with progress bar
            logger.info("Processing downloaded data...")
            with tqdm(total=len(raw_data), desc="Processing data", unit="records") as pbar:
                df = self.process_data(raw_data)
                pbar.update(len(raw_data))
            
            # Save to file if requested
            if save_to_file and not df.empty:
                logger.info("Saving data to file...")
                with tqdm(total=1, desc="Saving data", unit="file") as pbar:
                    self.save_data(df)
                    pbar.update(1)
            
            return df
            
        except Exception as e:
            logger.error(f"Error in download_historical_data: {str(e)}")
            raise


def main():
    """Main function to run the data downloader."""
    # Get API key from file
    try:
        api_key = load_api_key()
    except FileNotFoundError as e:
        logger.error(f"API key file not found: {str(e)}")
        logger.info("Please create an 'api_key.txt' file with your Polygon API key")
        logger.info("1. Create a file named 'api_key.txt' in the same directory")
        logger.info("2. Add your API key as the first line of the file")
        logger.info("3. Get your API key from: https://polygon.io/")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading API key: {str(e)}")
        sys.exit(1)
    
    try:
        # Initialize downloader
        downloader = SPYDataDownloader(api_key)
        
        # Download data
        logger.info("Starting SPY historical data download...")
        df = downloader.download_historical_data(years_back=2, save_to_file=True)
        
        if not df.empty:
            logger.info("Download completed successfully!")
        else:
            logger.error("No data was downloaded")
            
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
