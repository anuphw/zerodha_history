"""Analyze Zerodha account history and generate a comprehensive report."""

import json
import argparse
import random
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import requests
import numpy as np
import pandas as pd
import yfinance as yf

# Benchmark indices configuration
# Yahoo Finance tickers for Indian and global indices
BENCHMARKS = {
    "Nifty 50": "^NSEI",
    "Bank Nifty": "^NSEBANK",
    "Nifty IT": "^CNXIT",
    "Gold (INR)": "GC=F",  # Gold futures, will convert to INR
    "S&P 500 (INR)": "^GSPC",  # S&P 500, will convert to INR
}

# Risk-free rate (approximate Indian 10-year G-Sec yield)
RISK_FREE_RATE = 0.07  # 7% annual

# Global API call counter for rate limiting
_api_call_count = 0


def get_console_cookies_manual():
    """Open browser for manual login and extract Console cookies.

    Returns: (cookies_dict, user_id, browser, playwright)
    """
    from playwright.sync_api import sync_playwright

    print("Opening browser for manual login...")
    print("Please log in to Kite, then we'll navigate to Console.\n")

    p = sync_playwright().start()
    browser = p.chromium.launch(
        headless=False,
        args=["--start-maximized"]
    )
    context = browser.new_context(no_viewport=True)
    page = context.new_page()

    # Go to Kite login page
    page.goto("https://kite.zerodha.com/")

    # Wait for user to complete login - detect when we leave the login page
    print("Waiting for login to complete (5 min timeout)...")
    try:
        # Wait until URL no longer contains 'login' or we reach dashboard/positions
        page.wait_for_function(
            """() => {
                const url = window.location.href;
                return url.includes('/dashboard') ||
                       url.includes('/positions') ||
                       url.includes('/holdings') ||
                       url.includes('/orders');
            }""",
            timeout=300000
        )
    except Exception as e:
        print(f"Warning: Login detection timed out or failed: {e}")
        print("Continuing anyway - please make sure you're logged in...")
        time.sleep(2)

    print(f"Current URL: {page.url}")

    # Dismiss "I understand" modal if it appears
    try:
        page.get_by_text("I understand").click(timeout=3000)
        print("Dismissed modal")
    except Exception:
        pass

    # Extract user_id from cookies
    kite_cookies = context.cookies("https://kite.zerodha.com")
    user_id = None
    for c in kite_cookies:
        if c["name"] == "user_id":
            user_id = c["value"]
            break

    if user_id:
        print(f"Logged in as: {user_id}")
    else:
        print("WARNING: Could not extract user_id from cookies!")

    print("Navigating to Console...")

    # Click on user nav (the profile dropdown)
    try:
        # Wait a moment for page to stabilize
        time.sleep(1)

        page.locator("nav.user-nav.perspective").click(timeout=10000)
        print("Clicked user nav")

        # Click on Console link and capture the new tab
        with context.expect_page() as new_page_info:
            page.locator("a[href*='console.zerodha.com']").first.click(timeout=5000)

        # Get the new tab/page
        console_page = new_page_info.value
        console_page.wait_for_load_state("networkidle")
        print(f"Console page loaded: {console_page.url}")

        # Navigate to tradebook to ensure all cookies are set
        console_page.goto("https://console.zerodha.com/reports/tradebook")
        console_page.wait_for_load_state("networkidle")
        print("Tradebook page loaded")

        # Extract cookies from the console page context
        cookies = context.cookies("https://console.zerodha.com")
        cookie_dict = {c["name"]: c["value"] for c in cookies}

        print(f"Extracted {len(cookie_dict)} cookies")
        if "public_token" in cookie_dict:
            print("public_token cookie found")
        else:
            print("WARNING: public_token cookie NOT found!")

        return cookie_dict, user_id, browser, p

    except Exception as e:
        browser.close()
        p.stop()
        raise Exception(f"Failed to navigate to Console: {e}")


def fetch_with_retry(url, headers, max_retries=5):
    """Fetch URL with exponential backoff retry on any failure."""
    global _api_call_count

    # Add random delay after every 10 API calls to avoid rate limiting
    _api_call_count += 1
    if _api_call_count % 10 == 0:
        delay = random.uniform(5, 15)
        print(f"  [Throttle] {_api_call_count} API calls made, pausing {delay:.1f}s...")
        time.sleep(delay)

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 429:
                wait_time = 10 * (2 ** attempt)  # Longer waits: 10, 20, 40, 80, 160s
                print(f"  Rate limited (429), waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as e:
            wait_time = 5 * (2 ** attempt)
            print(f"  Request failed: {e}, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait_time)
    raise Exception(f"Max retries ({max_retries}) exceeded for {url}")


def fetch_tradebook(cookies, segment, from_date, to_date):
    """Fetch tradebook data for a given segment and date range."""
    all_transactions = []
    page_num = 1

    public_token = cookies.get("public_token", "")
    cookie_str = "; ".join(f"{k}={v}" for k, v in cookies.items())

    headers = {
        "accept": "application/json, text/plain, */*",
        "cookie": cookie_str,
        "referer": "https://console.zerodha.com/",
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "x-csrftoken": public_token,
    }

    while True:
        url = (
            f"https://console.zerodha.com/api/reports/tradebook"
            f"?segment={segment}&from_date={from_date}&to_date={to_date}"
            f"&page={page_num}&sort_by=order_execution_time&sort_desc=false"
        )

        resp = fetch_with_retry(url, headers)
        data = resp.json()
        results = data.get("data", {}).get("result", [])
        if not results:
            break

        all_transactions.extend(results)
        print(f"  Page {page_num}: {len(results)} transactions")
        page_num += 1
        time.sleep(0.5)

    return all_transactions


def fetch_account_values(cookies):
    """Fetch account values from Console API."""
    public_token = cookies.get("public_token", "")
    cookie_str = "; ".join(f"{k}={v}" for k, v in cookies.items())

    headers = {
        "accept": "application/json, text/plain, */*",
        "cookie": cookie_str,
        "referer": "https://console.zerodha.com/",
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "x-csrftoken": public_token,
    }

    url = "https://console.zerodha.com/api/dashboard/account_values"
    resp = fetch_with_retry(url, headers)
    return resp.json()


def fetch_profile(cookies):
    """Fetch user profile from Kite API."""
    cookie_str = "; ".join(f"{k}={v}" for k, v in cookies.items())

    headers = {
        "accept": "application/json, text/plain, */*",
        "cookie": cookie_str,
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    }

    # Try Kite API first (more reliable)
    try:
        url = "https://kite.zerodha.com/oms/user/profile"
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 200:
            data = resp.json().get("data", {})
            if data:
                return data
    except Exception as e:
        print(f"Warning: Could not fetch from Kite API: {e}")

    # Fallback to Console API
    try:
        headers["x-csrftoken"] = cookies.get("public_token", "")
        headers["referer"] = "https://console.zerodha.com/"
        url = "https://console.zerodha.com/api/user/profile"
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 200:
            return resp.json().get("data", {})
    except Exception as e:
        print(f"Warning: Could not fetch from Console API: {e}")

    return {}


def get_user_data_dir(user):
    """Get the data directory for a user, creating if needed."""
    data_dir = Path("tmp") / user
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def download_all_data(start_year=2020):
    """Download EQ, FO transactions and account values with manual login.

    Returns (user_id, user_name) extracted from the login session.
    """
    cookies, user_id, browser, p = get_console_cookies_manual()

    if not user_id:
        browser.close()
        p.stop()
        raise Exception("Could not determine user ID from login session")

    current_year = datetime.now().year
    data_dir = get_user_data_dir(user_id)

    # Fetch and save profile
    print("Fetching user profile...")
    profile = fetch_profile(cookies)
    # Try multiple field names for user name
    user_name = profile.get("user_name") or profile.get("name") or profile.get("user_shortname") or "Unknown"
    print(f"  User: {user_name} ({user_id})")

    profile_path = data_dir / "profile.json"
    with open(profile_path, "w") as f:
        json.dump(profile, f, indent=2)
    print(f"  Saved profile to {profile_path}")

    try:
        # Download EQ and FO transactions
        for segment in ["EQ", "FO"]:
            all_transactions = []

            for year in range(start_year, current_year + 1):
                from_date = f"{year}-01-01"
                to_date = f"{year}-12-31"
                print(f"Fetching {segment} transactions for {year}...")

                try:
                    transactions = fetch_tradebook(cookies, segment, from_date, to_date)
                    all_transactions.extend(transactions)
                    print(f"  Total for {year}: {len(transactions)} transactions")
                except Exception as e:
                    print(f"  ERROR fetching {segment} for {year}: {e}")

            filepath = data_dir / f"{segment}.jsonl"
            if all_transactions:
                with open(filepath, "w") as f:
                    for tx in all_transactions:
                        f.write(json.dumps(tx) + "\n")
                print(f"Saved {len(all_transactions)} {segment} transactions to {filepath}\n")
            else:
                print(f"WARNING: No {segment} transactions fetched! Keeping existing {filepath}\n")

        # Download account values
        print("Fetching account values...")
        try:
            data = fetch_account_values(cookies)
            results = data.get("data", {}).get("result", [])
            results.sort(key=lambda x: x.get("trade_date", ""))

            filepath = data_dir / "value.jsonl"
            if results:
                with open(filepath, "w") as f:
                    for item in results:
                        f.write(json.dumps(item) + "\n")
                print(f"Saved {len(results)} account values to {filepath}\n")
            else:
                print(f"WARNING: No account values fetched! Keeping existing {filepath}\n")
        except Exception as e:
            print(f"ERROR fetching account values: {e}\n")

    finally:
        browser.close()
        p.stop()

    return user_id, user_name


def load_data(user):
    """Load all data files for a user."""
    data_dir = get_user_data_dir(user)
    data = {}

    for filename in ['EQ.jsonl', 'FO.jsonl', 'value.jsonl']:
        filepath = data_dir / filename
        if filepath.exists():
            with open(filepath) as f:
                data[filename.replace('.jsonl', '')] = [json.loads(line) for line in f]
        else:
            data[filename.replace('.jsonl', '')] = []

    # Load profile
    profile_path = data_dir / "profile.json"
    if profile_path.exists():
        with open(profile_path) as f:
            data['profile'] = json.load(f)
    else:
        data['profile'] = {}

    return data


def parse_value(v):
    """Parse a value from the account data."""
    if v == '' or v is None or v == 'None':
        return 0
    return float(v)


def get_benchmark_cache_path(user):
    """Get path for cached benchmark data."""
    return get_user_data_dir(user) / "benchmarks.json"


def save_benchmark_cache(user, benchmarks_data, start_date, end_date):
    """Save benchmark data to cache."""
    cache_path = get_benchmark_cache_path(user)
    cache = {
        'start_date': start_date,
        'end_date': end_date,
        'fetched_at': datetime.now().isoformat(),
        'data': {}
    }
    for name, series in benchmarks_data.items():
        cache['data'][name] = {
            'dates': series.index.strftime('%Y-%m-%d').tolist(),
            'values': series.tolist()
        }
    with open(cache_path, 'w') as f:
        json.dump(cache, f)


def load_benchmark_cache(user, start_date, end_date):
    """Load benchmark data from cache if valid.

    Returns None if cache is missing, stale (>24h), or date range doesn't match.
    """
    cache_path = get_benchmark_cache_path(user)
    if not cache_path.exists():
        return None

    try:
        with open(cache_path) as f:
            cache = json.load(f)

        # Check if cache is recent (within 24 hours)
        fetched_at = datetime.fromisoformat(cache['fetched_at'])
        if datetime.now() - fetched_at > timedelta(hours=24):
            print("  Benchmark cache expired (>24h old)")
            return None

        # Check date range covers what we need
        if cache['start_date'] > start_date or cache['end_date'] < end_date:
            print("  Benchmark cache date range doesn't match")
            return None

        # Reconstruct Series objects
        benchmarks_data = {}
        for name, data in cache['data'].items():
            dates = pd.to_datetime(data['dates'])
            benchmarks_data[name] = pd.Series(data['values'], index=dates)

        print(f"  Loaded {len(benchmarks_data)} benchmarks from cache")
        return benchmarks_data

    except Exception as e:
        print(f"  Could not load benchmark cache: {e}")
        return None


def fetch_benchmark_data(start_date, end_date, user=None):
    """Fetch benchmark data from Yahoo Finance.

    Returns a dict of DataFrames with daily closing prices.
    For USD-based assets (Gold, S&P 500), converts to INR.
    Uses cache if available and valid.
    """
    # Try loading from cache first
    if user:
        cached = load_benchmark_cache(user, start_date, end_date)
        if cached:
            return cached

    print("Fetching benchmark data from Yahoo Finance...")
    benchmarks_data = {}

    # Fetch USD/INR for conversion
    usdinr_rate = 83.0  # Default fallback
    try:
        usdinr = yf.download("USDINR=X", start=start_date, end=end_date, progress=False)
        if not usdinr.empty:
            # Handle multi-index columns from yfinance
            close_col = usdinr['Close']
            if isinstance(close_col, pd.DataFrame):
                close_col = close_col.iloc[:, 0]  # Get first column if multi-index
            usdinr_rate = close_col.ffill()
    except Exception as e:
        print(f"  Warning: Could not fetch USD/INR rate: {e}")

    for name, ticker in BENCHMARKS.items():
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not data.empty:
                # Handle multi-index columns from yfinance
                close_col = data['Close']
                if isinstance(close_col, pd.DataFrame):
                    close_col = close_col.iloc[:, 0]  # Get first column if multi-index

                # Convert USD-based assets to INR
                if name in ["Gold (INR)", "S&P 500 (INR)"]:
                    if isinstance(usdinr_rate, pd.Series):
                        # Align dates and multiply
                        aligned_rate = usdinr_rate.reindex(close_col.index, method='ffill')
                        close_col = close_col * aligned_rate
                    else:
                        close_col = close_col * usdinr_rate

                benchmarks_data[name] = close_col
                print(f"  Fetched {name}: {len(close_col)} data points")
            else:
                print(f"  Warning: No data for {name}")
        except Exception as e:
            print(f"  Warning: Could not fetch {name}: {e}")

    # Save to cache
    if user and benchmarks_data:
        save_benchmark_cache(user, benchmarks_data, start_date, end_date)
        print("  Saved benchmark data to cache")

    return benchmarks_data


def build_portfolio_timeseries(value_data, deposits_by_date, withdrawals_by_date):
    """Build a time series of portfolio values and calculate returns.

    Uses Time-Weighted Return (TWR) methodology to account for cash flows.
    Returns a DataFrame with dates, values, and daily returns.
    """
    eq_values = [r for r in value_data if r['segment'] == 'EQ']
    if not eq_values:
        return pd.DataFrame()

    # Build daily portfolio values
    records = []
    for r in sorted(eq_values, key=lambda x: x['trade_date']):
        date = r['trade_date']
        cash = parse_value(r['values'][3])
        holdings = parse_value(r['values'][5])
        mf = parse_value(r['values'][6])
        total = cash + holdings + mf

        deposit = deposits_by_date.get(date, 0)
        withdrawal = withdrawals_by_date.get(date, 0)

        records.append({
            'date': pd.to_datetime(date),
            'value': total,
            'deposit': deposit,
            'withdrawal': withdrawal,
            'cash_flow': deposit - withdrawal
        })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df = df.set_index('date').sort_index()

    # Remove duplicate dates, keeping last value
    df = df[~df.index.duplicated(keep='last')]

    # Calculate daily returns adjusted for cash flows (TWR approximation)
    # Return = (End Value - Cash Flow) / Start Value - 1
    df['prev_value'] = df['value'].shift(1)
    df['daily_return'] = (df['value'] - df['cash_flow']) / df['prev_value'] - 1
    df.loc[df['prev_value'] == 0, 'daily_return'] = 0
    df.loc[df['prev_value'].isna(), 'daily_return'] = 0
    df['daily_return'] = df['daily_return'].clip(-0.5, 0.5)  # Cap extreme values

    return df


def calculate_returns_for_period(portfolio_df, benchmark_series, years):
    """Calculate portfolio and benchmark returns for a given period.

    Returns tuple: (portfolio_return, benchmark_return, num_days)
    """
    if portfolio_df.empty:
        return None, None, 0

    end_date = portfolio_df.index.max()
    start_date = end_date - pd.DateOffset(years=years)

    # Filter to period
    mask = portfolio_df.index >= start_date
    period_df = portfolio_df[mask]

    if len(period_df) < 20:  # Need at least ~1 month of data
        return None, None, 0

    # Portfolio return using TWR (compounding daily returns)
    portfolio_return = (1 + period_df['daily_return']).prod() - 1

    # Benchmark return
    benchmark_return = None
    if benchmark_series is not None and not benchmark_series.empty:
        bench_period = benchmark_series[benchmark_series.index >= start_date]
        if len(bench_period) >= 2:
            benchmark_return = (bench_period.iloc[-1] / bench_period.iloc[0]) - 1

    return portfolio_return, benchmark_return, len(period_df)


def calculate_alpha_beta(portfolio_df, benchmark_series, years=None):
    """Calculate Alpha and Beta relative to a benchmark.

    Alpha: Excess return over benchmark (risk-adjusted)
    Beta: Portfolio sensitivity to market movements

    Returns tuple: (alpha, beta, r_squared)
    """
    if portfolio_df.empty or benchmark_series is None or benchmark_series.empty:
        return None, None, None

    # Filter by time period if specified
    if years:
        end_date = portfolio_df.index.max()
        start_date = end_date - pd.DateOffset(years=years)
        portfolio_df = portfolio_df[portfolio_df.index >= start_date]

    # Align dates
    common_dates = portfolio_df.index.intersection(benchmark_series.index)
    if len(common_dates) < 30:
        return None, None, None

    # Get aligned returns
    port_returns = portfolio_df.loc[common_dates, 'daily_return'].dropna()

    # Calculate benchmark daily returns
    bench_values = benchmark_series.loc[common_dates]
    bench_returns = bench_values.pct_change().dropna()

    # Align again after pct_change
    common = port_returns.index.intersection(bench_returns.index)
    if len(common) < 30:
        return None, None, None

    port_ret = port_returns.loc[common].values.flatten()
    bench_ret = bench_returns.loc[common].values.flatten()

    # Remove any NaN or inf values
    mask = np.isfinite(port_ret) & np.isfinite(bench_ret)
    port_ret = port_ret[mask]
    bench_ret = bench_ret[mask]

    if len(port_ret) < 30:
        return None, None, None

    # Calculate Beta using covariance method
    cov_matrix = np.cov(port_ret, bench_ret)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else None

    # Calculate Alpha (annualized)
    # Alpha = Portfolio Return - (Risk-free + Beta * (Market Return - Risk-free))
    port_annual = (1 + np.mean(port_ret)) ** 252 - 1
    bench_annual = (1 + np.mean(bench_ret)) ** 252 - 1

    if beta is not None:
        alpha = port_annual - (RISK_FREE_RATE + beta * (bench_annual - RISK_FREE_RATE))
    else:
        alpha = None

    # R-squared
    if len(port_ret) > 2:
        correlation = np.corrcoef(port_ret, bench_ret)[0, 1]
        r_squared = correlation ** 2 if np.isfinite(correlation) else None
    else:
        r_squared = None

    return alpha, beta, r_squared


def calculate_sharpe_ratio(portfolio_df, years=None):
    """Calculate Sharpe Ratio.

    Sharpe = (Portfolio Return - Risk-free Rate) / Portfolio Std Dev
    Measures risk-adjusted return per unit of volatility.
    """
    if portfolio_df.empty:
        return None

    if years:
        end_date = portfolio_df.index.max()
        start_date = end_date - pd.DateOffset(years=years)
        portfolio_df = portfolio_df[portfolio_df.index >= start_date]

    returns = portfolio_df['daily_return'].dropna()
    if len(returns) < 30:
        return None

    # Annualize
    annual_return = (1 + returns.mean()) ** 252 - 1
    annual_std = returns.std() * np.sqrt(252)

    if annual_std == 0:
        return None

    sharpe = (annual_return - RISK_FREE_RATE) / annual_std
    return sharpe


def calculate_sortino_ratio(portfolio_df, years=None):
    """Calculate Sortino Ratio.

    Sortino = (Portfolio Return - Risk-free Rate) / Downside Deviation
    Like Sharpe but only penalizes downside volatility.
    """
    if portfolio_df.empty:
        return None

    if years:
        end_date = portfolio_df.index.max()
        start_date = end_date - pd.DateOffset(years=years)
        portfolio_df = portfolio_df[portfolio_df.index >= start_date]

    returns = portfolio_df['daily_return'].dropna()
    if len(returns) < 30:
        return None

    # Annualize return
    annual_return = (1 + returns.mean()) ** 252 - 1

    # Downside deviation (only negative returns)
    negative_returns = returns[returns < 0]
    if len(negative_returns) == 0:
        return None  # No downside = undefined

    downside_std = np.sqrt(np.mean(negative_returns ** 2)) * np.sqrt(252)

    if downside_std == 0:
        return None

    sortino = (annual_return - RISK_FREE_RATE) / downside_std
    return sortino


def calculate_max_drawdown(portfolio_df, years=None):
    """Calculate Maximum Drawdown.

    Max Drawdown = Largest peak-to-trough decline
    Measures worst-case loss from any high point.
    """
    if portfolio_df.empty:
        return None, None, None

    if years:
        end_date = portfolio_df.index.max()
        start_date = end_date - pd.DateOffset(years=years)
        portfolio_df = portfolio_df[portfolio_df.index >= start_date]

    values = portfolio_df['value']
    if len(values) < 2:
        return None, None, None

    # Calculate running maximum
    running_max = values.expanding().max()
    drawdown = (values - running_max) / running_max

    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()

    # Find the peak before the max drawdown
    peak_date = values[:max_dd_date].idxmax() if max_dd_date else None

    return max_dd, peak_date, max_dd_date


def calculate_volatility(portfolio_df, years=None):
    """Calculate annualized volatility (standard deviation of returns)."""
    if portfolio_df.empty:
        return None

    if years:
        end_date = portfolio_df.index.max()
        start_date = end_date - pd.DateOffset(years=years)
        portfolio_df = portfolio_df[portfolio_df.index >= start_date]

    returns = portfolio_df['daily_return'].dropna()
    if len(returns) < 30:
        return None

    return returns.std() * np.sqrt(252)


def calculate_calmar_ratio(portfolio_df, years=None):
    """Calculate Calmar Ratio.

    Calmar = Annual Return / |Max Drawdown|
    Measures return per unit of drawdown risk.
    """
    if portfolio_df.empty:
        return None

    if years:
        end_date = portfolio_df.index.max()
        start_date = end_date - pd.DateOffset(years=years)
        period_df = portfolio_df[portfolio_df.index >= start_date]
    else:
        period_df = portfolio_df

    if len(period_df) < 30:
        return None

    # Annual return
    returns = period_df['daily_return'].dropna()
    annual_return = (1 + returns.mean()) ** 252 - 1

    # Max drawdown
    max_dd, _, _ = calculate_max_drawdown(period_df)

    if max_dd is None or max_dd == 0:
        return None

    return annual_return / abs(max_dd)


def calculate_information_ratio(portfolio_df, benchmark_series, years=None):
    """Calculate Information Ratio.

    IR = (Portfolio Return - Benchmark Return) / Tracking Error
    Measures excess return per unit of tracking error.
    """
    if portfolio_df.empty or benchmark_series is None or benchmark_series.empty:
        return None

    if years:
        end_date = portfolio_df.index.max()
        start_date = end_date - pd.DateOffset(years=years)
        portfolio_df = portfolio_df[portfolio_df.index >= start_date]

    # Align dates
    common_dates = portfolio_df.index.intersection(benchmark_series.index)
    if len(common_dates) < 30:
        return None

    port_returns = portfolio_df.loc[common_dates, 'daily_return'].dropna()
    bench_values = benchmark_series.loc[common_dates]
    bench_returns = bench_values.pct_change().dropna()

    common = port_returns.index.intersection(bench_returns.index)
    if len(common) < 30:
        return None

    excess_returns = port_returns.loc[common] - bench_returns.loc[common]

    # Annualize
    annual_excess = excess_returns.mean() * 252
    tracking_error = excess_returns.std() * np.sqrt(252)

    if tracking_error == 0:
        return None

    return annual_excess / tracking_error


def calculate_win_rate(portfolio_df, years=None):
    """Calculate win rate (percentage of positive days)."""
    if portfolio_df.empty:
        return None

    if years:
        end_date = portfolio_df.index.max()
        start_date = end_date - pd.DateOffset(years=years)
        portfolio_df = portfolio_df[portfolio_df.index >= start_date]

    returns = portfolio_df['daily_return'].dropna()
    if len(returns) < 10:
        return None

    positive_days = (returns > 0).sum()
    return positive_days / len(returns)


def calculate_all_metrics(portfolio_df, benchmarks_data, time_periods=[1, 3, 5, 7, 10]):
    """Calculate all financial metrics for multiple time periods.

    Returns a nested dict: {period: {metric: value}}
    """
    results = {}

    # Use Nifty 50 as primary benchmark
    primary_benchmark = benchmarks_data.get("Nifty 50")

    for years in time_periods:
        period_key = f"{years}Y"
        results[period_key] = {}

        # Check if we have enough data for this period
        if portfolio_df.empty:
            continue

        end_date = portfolio_df.index.max()
        start_date = end_date - pd.DateOffset(years=years)
        period_data = portfolio_df[portfolio_df.index >= start_date]

        if len(period_data) < 30:
            results[period_key]['data_available'] = False
            continue

        results[period_key]['data_available'] = True

        # Portfolio return
        port_return, _, _ = calculate_returns_for_period(portfolio_df, None, years)
        results[period_key]['portfolio_return'] = port_return

        # Benchmark returns
        results[period_key]['benchmark_returns'] = {}
        for bench_name, bench_series in benchmarks_data.items():
            _, bench_return, _ = calculate_returns_for_period(portfolio_df, bench_series, years)
            results[period_key]['benchmark_returns'][bench_name] = bench_return

        # Alpha and Beta (vs Nifty 50)
        alpha, beta, r_squared = calculate_alpha_beta(portfolio_df, primary_benchmark, years)
        results[period_key]['alpha'] = alpha
        results[period_key]['beta'] = beta
        results[period_key]['r_squared'] = r_squared

        # Risk metrics
        results[period_key]['sharpe_ratio'] = calculate_sharpe_ratio(portfolio_df, years)
        results[period_key]['sortino_ratio'] = calculate_sortino_ratio(portfolio_df, years)
        results[period_key]['volatility'] = calculate_volatility(portfolio_df, years)
        results[period_key]['calmar_ratio'] = calculate_calmar_ratio(portfolio_df, years)

        # Drawdown
        max_dd, peak_date, trough_date = calculate_max_drawdown(portfolio_df, years)
        results[period_key]['max_drawdown'] = max_dd
        results[period_key]['max_dd_peak'] = peak_date
        results[period_key]['max_dd_trough'] = trough_date

        # Information Ratio (vs Nifty 50)
        results[period_key]['information_ratio'] = calculate_information_ratio(
            portfolio_df, primary_benchmark, years
        )

        # Win rate
        results[period_key]['win_rate'] = calculate_win_rate(portfolio_df, years)

    return results


def get_unique_deposits_withdrawals(value_data):
    """Get unique deposits and withdrawals per date (avoid duplicates)."""
    eq_values = [r for r in value_data if r['segment'] == 'EQ']

    deposits_by_date = {}
    withdrawals_by_date = {}

    for r in eq_values:
        date = r['trade_date']
        d = parse_value(r['values'][0])
        w = parse_value(r['values'][1])

        if d > 0 and date not in deposits_by_date:
            deposits_by_date[date] = d
        if w > 0 and date not in withdrawals_by_date:
            withdrawals_by_date[date] = w

    return deposits_by_date, withdrawals_by_date


def analyze_eq_trades(eq_trades):
    """Analyze equity trades and calculate realized P&L using FIFO."""
    holdings = defaultdict(lambda: {'buys': [], 'realized_pnl': 0, 'qty': 0, 'cost': 0})

    for t in sorted(eq_trades, key=lambda x: x['order_execution_time']):
        symbol = t['tradingsymbol']
        qty = t['quantity']
        price = t['price']
        date = t['trade_date']

        if t['trade_type'] == 'buy':
            holdings[symbol]['buys'].append({'qty': qty, 'price': price, 'date': date})
            holdings[symbol]['qty'] += qty
            holdings[symbol]['cost'] += qty * price
        else:
            sell_qty = qty
            sell_value = qty * price
            cost_basis = 0

            while sell_qty > 0 and holdings[symbol]['buys']:
                buy = holdings[symbol]['buys'][0]
                matched_qty = min(sell_qty, buy['qty'])
                cost_basis += matched_qty * buy['price']

                buy['qty'] -= matched_qty
                sell_qty -= matched_qty

                if buy['qty'] == 0:
                    holdings[symbol]['buys'].pop(0)

            realized = sell_value - cost_basis
            holdings[symbol]['realized_pnl'] += realized
            holdings[symbol]['qty'] -= qty
            holdings[symbol]['cost'] -= cost_basis

    return holdings


def analyze_fo_trades(fo_trades, eq_trades):
    """Analyze F&O trades with physical delivery detection."""
    today = datetime.now()

    positions = defaultdict(lambda: {
        'buy_qty': 0, 'sell_qty': 0,
        'buy_val': 0, 'sell_val': 0,
        'instrument_type': '', 'expiry': '',
        'trades': []
    })

    for t in fo_trades:
        key = (t['tradingsymbol'], t.get('expiry_date', ''))
        qty = t['quantity']
        val = qty * t['price']
        positions[key]['instrument_type'] = t.get('instrument_type', '')
        positions[key]['expiry'] = t.get('expiry_date', '')
        positions[key]['trades'].append(t)

        if t['trade_type'] == 'buy':
            positions[key]['buy_qty'] += qty
            positions[key]['buy_val'] += val
        else:
            positions[key]['sell_qty'] += qty
            positions[key]['sell_val'] += val

    # Categorize positions
    results = {
        'fut_closed_pnl': 0, 'fut_expired_pnl': 0, 'fut_open_cost': 0,
        'opt_closed_pnl': 0, 'opt_expired_pnl': 0, 'opt_open_cost': 0,
        'physical_deliveries': [],
        'open_positions': []
    }

    for (sym, exp), pos in positions.items():
        net_qty = pos['buy_qty'] - pos['sell_qty']
        pnl = pos['sell_val'] - pos['buy_val']
        itype = pos['instrument_type']

        try:
            exp_date = datetime.strptime(exp, '%Y-%m-%d') if exp else today
        except:
            exp_date = today

        is_expired = exp_date < today
        is_closed = net_qty == 0

        # Check for physical delivery (expired futures with open qty)
        if itype == 'FUT' and is_expired and not is_closed:
            # Extract base symbol - strip FUT and the YYMM pattern (e.g., "23MAR" or "23JAN")
            # Symbol format: HDFCBANK23MARFUT -> HDFCBANK
            base_sym = sym.replace('FUT', '')
            # Remove trailing year+month pattern (2 digits + 3 letters like "23MAR")
            match = re.match(r'^(.+?)\d{2}[A-Z]{3}$', base_sym)
            if match:
                base_sym = match.group(1)

            # Look for EQ sales after expiry
            delivery_sales = [t for t in eq_trades
                           if t['tradingsymbol'] == base_sym
                           and t['trade_type'] == 'sell'
                           and datetime.strptime(t['trade_date'], '%Y-%m-%d') >= exp_date
                           and datetime.strptime(t['trade_date'], '%Y-%m-%d') <= exp_date + timedelta(days=10)]

            if delivery_sales:
                delivery_qty = sum(t['quantity'] for t in delivery_sales)
                delivery_val = sum(t['quantity'] * t['price'] for t in delivery_sales)

                if delivery_qty >= abs(net_qty) * 0.9:
                    actual_pnl = delivery_val - pos['buy_val'] + pos['sell_val']
                    results['physical_deliveries'].append({
                        'symbol': sym, 'base': base_sym, 'expiry': exp,
                        'qty': net_qty, 'entry_cost': pos['buy_val'],
                        'delivery_value': delivery_val, 'pnl': actual_pnl
                    })
                    results['fut_closed_pnl'] += actual_pnl
                    continue

        # Categorize normally
        if itype == 'FUT':
            if is_closed:
                results['fut_closed_pnl'] += pnl
            elif is_expired:
                results['fut_expired_pnl'] += pnl
            else:
                results['fut_open_cost'] += -pnl
                results['open_positions'].append({
                    'symbol': sym, 'expiry': exp, 'type': 'FUT',
                    'qty': net_qty, 'cost': -pnl
                })
        else:
            if is_closed:
                results['opt_closed_pnl'] += pnl
            elif is_expired:
                results['opt_expired_pnl'] += pnl
            else:
                results['opt_open_cost'] += -pnl
                results['open_positions'].append({
                    'symbol': sym, 'expiry': exp, 'type': 'OPT',
                    'qty': net_qty, 'cost': -pnl
                })

    return results


def get_quarterly_data(deposits_by_date, withdrawals_by_date, eq_trades, fo_trades, value_data):
    """Generate quarterly summaries."""
    quarters = defaultdict(lambda: {
        'deposits': 0, 'withdrawals': 0,
        'eq_buy': 0, 'eq_sell': 0,
        'fo_buy': 0, 'fo_sell': 0,
        'fo_opt_buy': 0, 'fo_opt_sell': 0,
        'fo_fut_buy': 0, 'fo_fut_sell': 0,
        'cash': 0, 'holdings': 0, 'mf': 0
    })

    def get_quarter(date_str):
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        q = (dt.month - 1) // 3 + 1
        return f"{dt.year}-Q{q}"

    # Deposits/Withdrawals
    for date, amt in deposits_by_date.items():
        q = get_quarter(date)
        quarters[q]['deposits'] += amt

    for date, amt in withdrawals_by_date.items():
        q = get_quarter(date)
        quarters[q]['withdrawals'] += amt

    # EQ trades
    for t in eq_trades:
        q = get_quarter(t['trade_date'])
        val = t['quantity'] * t['price']
        if t['trade_type'] == 'buy':
            quarters[q]['eq_buy'] += val
        else:
            quarters[q]['eq_sell'] += val

    # FO trades
    for t in fo_trades:
        q = get_quarter(t['trade_date'])
        val = t['quantity'] * t['price']
        itype = t.get('instrument_type', 'OPT')

        if t['trade_type'] == 'buy':
            quarters[q]['fo_buy'] += val
            if itype == 'FUT':
                quarters[q]['fo_fut_buy'] += val
            else:
                quarters[q]['fo_opt_buy'] += val
        else:
            quarters[q]['fo_sell'] += val
            if itype == 'FUT':
                quarters[q]['fo_fut_sell'] += val
            else:
                quarters[q]['fo_opt_sell'] += val

    # End-of-quarter values (take last record of each quarter)
    eq_values = [r for r in value_data if r['segment'] == 'EQ']
    for r in eq_values:
        q = get_quarter(r['trade_date'])
        quarters[q]['cash'] = parse_value(r['values'][3])
        quarters[q]['holdings'] = parse_value(r['values'][5])
        quarters[q]['mf'] = parse_value(r['values'][6])

    return dict(quarters)


def format_inr(amount):
    """Format amount in Indian Rupee style."""
    if amount < 0:
        return f"-₹{abs(amount):,.0f}"
    return f"₹{amount:,.0f}"


def format_inr_lakhs(amount):
    """Format amount in lakhs."""
    lakhs = amount / 100000
    if amount < 0:
        return f"-₹{abs(lakhs):.2f}L"
    return f"₹{lakhs:.2f}L"


def generate_report(data, user_id, benchmarks_data=None):
    """Generate the markdown report."""
    eq_trades = data['EQ']
    fo_trades = data['FO']
    value_data = data['value']
    profile = data.get('profile', {})

    # Get unique deposits/withdrawals
    deposits_by_date, withdrawals_by_date = get_unique_deposits_withdrawals(value_data)
    total_deposits = sum(deposits_by_date.values())
    total_withdrawals = sum(withdrawals_by_date.values())
    net_invested = total_deposits - total_withdrawals

    # Current values
    eq_values = [r for r in value_data if r['segment'] == 'EQ']
    latest = eq_values[-1] if eq_values else None

    if latest:
        cash = parse_value(latest['values'][3])
        holdings_market = parse_value(latest['values'][5])
        mf_value = parse_value(latest['values'][6])
        current_date = latest['trade_date']
    else:
        cash = holdings_market = mf_value = 0
        current_date = datetime.now().strftime('%Y-%m-%d')

    current_value = cash + holdings_market + mf_value

    # Analyze trades
    eq_holdings = analyze_eq_trades(eq_trades)
    fo_results = analyze_fo_trades(fo_trades, eq_trades)

    # Calculate metrics
    eq_realized_pnl = sum(h['realized_pnl'] for h in eq_holdings.values())
    eq_holdings_cost = sum(h['cost'] for h in eq_holdings.values() if h['qty'] > 0)
    eq_unrealized_pnl = holdings_market - eq_holdings_cost

    fut_realized = fo_results['fut_closed_pnl'] + fo_results['fut_expired_pnl']
    opt_realized = fo_results['opt_closed_pnl'] + fo_results['opt_expired_pnl']
    fo_realized = fut_realized + opt_realized
    fo_open_cost = fo_results['fut_open_cost'] + fo_results['opt_open_cost']

    total_realized_pnl = eq_realized_pnl + fo_realized
    overall_gain = current_value - net_invested
    return_pct = (overall_gain / net_invested * 100) if net_invested > 0 else 0

    # Quarterly data
    quarterly = get_quarterly_data(deposits_by_date, withdrawals_by_date, eq_trades, fo_trades, value_data)

    # Get date range
    all_dates = list(deposits_by_date.keys()) + [t['trade_date'] for t in eq_trades] + [t['trade_date'] for t in fo_trades]
    start_date = min(all_dates) if all_dates else current_date

    # Top holdings
    top_holdings = sorted(
        [(sym, h) for sym, h in eq_holdings.items() if h['qty'] > 0],
        key=lambda x: -x[1]['cost']
    )[:10]

    # Build portfolio time series and calculate advanced metrics
    portfolio_df = build_portfolio_timeseries(value_data, deposits_by_date, withdrawals_by_date)
    if benchmarks_data is None:
        benchmarks_data = {}
    financial_metrics = calculate_all_metrics(portfolio_df, benchmarks_data) if not portfolio_df.empty else {}

    # Generate report
    report = []
    user_name = profile.get('user_name') or profile.get('name') or profile.get('user_shortname') or 'Unknown'
    report.append(f"# Zerodha Account History: {user_name}")
    report.append("")

    # Table of Contents
    report.append("## Table of Contents")
    report.append("")
    report.append("1. [Profile](#profile)")
    report.append("2. [Executive Summary](#executive-summary)")
    report.append("3. [Current Portfolio](#current-portfolio)")
    report.append("4. [The Account Story](#the-account-story)")
    report.append("5. [Profit & Loss Summary](#profit--loss-summary)")
    report.append("6. [Key Metrics](#key-metrics)")
    report.append("7. [Performance vs Benchmarks](#performance-vs-benchmarks)")
    report.append("8. [Risk-Adjusted Metrics](#risk-adjusted-metrics)")
    report.append("9. [Top 10 Current Holdings](#top-10-current-holdings)")
    report.append("10. [Quarterly Summary](#quarterly-summary)")
    report.append("11. [Yearly Summary](#yearly-summary)")
    report.append("12. [Equity Transactions](#equity-transactions)")
    report.append("13. [F&O Transactions](#fo-transactions)")
    report.append("14. [Deposits](#deposits)")
    report.append("15. [Withdrawals](#withdrawals)")
    report.append("16. [Detailed Quarterly Breakdown](#detailed-quarterly-breakdown)")
    report.append("")

    # Profile
    report.append("## Profile")
    report.append("")
    report.append("| Field | Value |")
    report.append("|-------|-------|")
    report.append(f"| **User ID** | {user_id} |")
    report.append(f"| **Name** | {user_name} |")
    if profile.get('email'):
        report.append(f"| **Email** | {profile.get('email')} |")
    if profile.get('phone'):
        report.append(f"| **Phone** | {profile.get('phone')} |")
    if profile.get('pan'):
        report.append(f"| **PAN** | {profile.get('pan')} |")
    report.append(f"| **Report Generated** | {datetime.now().strftime('%Y-%m-%d %H:%M')} |")
    report.append(f"| **Data Period** | {start_date} to {current_date} |")
    report.append("")

    # Executive Summary
    report.append("## Executive Summary")
    report.append("")
    report.append("| Metric | Value |")
    report.append("|--------|------:|")
    report.append(f"| Total Deposited | {format_inr(total_deposits)} |")
    report.append(f"| Total Withdrawn | {format_inr(total_withdrawals)} |")
    report.append(f"| Net Invested | {format_inr(net_invested)} |")
    report.append(f"| **Current Value** | **{format_inr(current_value)}** |")
    report.append(f"| **Overall Gain/Loss** | **{format_inr(overall_gain)}** |")
    report.append(f"| **Return** | **{return_pct:+.1f}%** |")
    report.append("")

    # Current Portfolio
    report.append("## Current Portfolio")
    report.append("")
    report.append("| Asset | Value |")
    report.append("|-------|------:|")
    report.append(f"| Cash Balance | {format_inr(cash)} |")
    report.append(f"| Stock Holdings | {format_inr(holdings_market)} |")
    report.append(f"| Mutual Funds | {format_inr(mf_value)} |")
    if fo_open_cost > 0:
        report.append(f"| F&O Open Positions | {format_inr(fo_open_cost)} |")
    report.append(f"| **Total** | **{format_inr(current_value)}** |")
    report.append("")

    # The Story
    report.append("## The Account Story")
    report.append("")
    report.append(f"This account was opened in **{start_date[:4]}** with the first deposit on **{start_date}**.")
    report.append("")
    report.append(f"Over the years, a total of **{format_inr(total_deposits)}** was deposited across **{len(deposits_by_date)}** transactions, ")
    report.append(f"and **{format_inr(total_withdrawals)}** was withdrawn across **{len(withdrawals_by_date)}** transactions.")
    report.append("")

    # Trading activity
    eq_buy_total = sum(t['quantity'] * t['price'] for t in eq_trades if t['trade_type'] == 'buy')
    eq_sell_total = sum(t['quantity'] * t['price'] for t in eq_trades if t['trade_type'] == 'sell')

    report.append("### Equity Trading")
    report.append("")
    report.append(f"- **{len(eq_trades)}** equity trades executed")
    report.append(f"- Total bought: {format_inr(eq_buy_total)}")
    report.append(f"- Total sold: {format_inr(eq_sell_total)}")
    report.append(f"- Realized P&L: **{format_inr(eq_realized_pnl)}**")
    report.append(f"- Unrealized P&L: **{format_inr(eq_unrealized_pnl)}**")
    report.append("")

    report.append("### F&O Trading")
    report.append("")
    report.append(f"- **{len(fo_trades)}** F&O trades executed")
    report.append("")
    report.append("| Segment | Realized P&L |")
    report.append("|---------|-------------:|")
    report.append(f"| Futures | {format_inr(fut_realized)} |")
    report.append(f"| Options | {format_inr(opt_realized)} |")
    report.append(f"| **Total F&O** | **{format_inr(fo_realized)}** |")
    report.append("")

    if fo_results['physical_deliveries']:
        report.append("#### Physical Deliveries")
        report.append("")
        report.append("The following futures positions resulted in physical delivery:")
        report.append("")
        for pd in fo_results['physical_deliveries']:
            report.append(f"- **{pd['symbol']}** (expired {pd['expiry']}): {pd['qty']} shares delivered, P&L: {format_inr(pd['pnl'])}")
        report.append("")

    if fo_results['open_positions']:
        report.append("#### Open F&O Positions")
        report.append("")
        report.append("| Symbol | Type | Qty | Cost Basis |")
        report.append("|--------|------|----:|-----------:|")
        for pos in sorted(fo_results['open_positions'], key=lambda x: -abs(x['cost']))[:10]:
            report.append(f"| {pos['symbol']} | {pos['type']} | {pos['qty']} | {format_inr(pos['cost'])} |")
        report.append("")

    # P&L Summary
    report.append("## Profit & Loss Summary")
    report.append("")
    report.append("| Category | Amount |")
    report.append("|----------|-------:|")
    report.append(f"| EQ Realized P&L | {format_inr(eq_realized_pnl)} |")
    report.append(f"| EQ Unrealized P&L | {format_inr(eq_unrealized_pnl)} |")
    report.append(f"| Futures P&L | {format_inr(fut_realized)} |")
    report.append(f"| Options P&L | {format_inr(opt_realized)} |")
    report.append(f"| **Total Realized** | **{format_inr(total_realized_pnl)}** |")
    report.append(f"| **Total (incl. Unrealized)** | **{format_inr(total_realized_pnl + eq_unrealized_pnl)}** |")
    report.append("")

    # Key Metrics
    report.append("## Key Metrics")
    report.append("")
    profit_taken_home = total_withdrawals  # Money actually withdrawn
    profit_booked = eq_realized_pnl + fo_realized  # Realized trading profits
    paper_profit = eq_unrealized_pnl  # Unrealized gains

    report.append("| Metric | Value | Description |")
    report.append("|--------|------:|-------------|")
    report.append(f"| Net Invested | {format_inr(net_invested)} | Deposits minus withdrawals |")
    report.append(f"| Profit Booked | {format_inr(profit_booked)} | Realized trading P&L |")
    report.append(f"| Paper Profit | {format_inr(paper_profit)} | Unrealized gains on holdings |")
    report.append(f"| Profit Taken Home | {format_inr(profit_taken_home)} | Total withdrawals |")
    report.append(f"| Current Wealth | {format_inr(current_value)} | Cash + Holdings + MF |")
    report.append(f"| Total Return | {return_pct:+.1f}% | Overall portfolio return |")
    report.append("")

    # Performance vs Benchmarks Section
    if financial_metrics:
        report.append("## Performance vs Benchmarks")
        report.append("")
        report.append("Compare your portfolio returns against major market indices and assets over different time periods.")
        report.append("")

        # Build comparison table header
        periods_available = [p for p in ["1Y", "3Y", "5Y", "7Y", "10Y"] if p in financial_metrics and financial_metrics[p].get('data_available')]
        if periods_available:
            header = "| Asset |"
            separator = "|-------|"
            for period in periods_available:
                header += f" {period} |"
                separator += "------:|"
            report.append(header)
            report.append(separator)

            # Portfolio row
            row = "| **Your Portfolio** |"
            for period in periods_available:
                ret = financial_metrics[period].get('portfolio_return')
                row += f" **{ret*100:+.1f}%** |" if ret is not None else " N/A |"
            report.append(row)

            # Benchmark rows
            benchmark_names = ["Nifty 50", "Bank Nifty", "Nifty IT", "Gold (INR)", "S&P 500 (INR)"]
            for bench_name in benchmark_names:
                row = f"| {bench_name} |"
                for period in periods_available:
                    bench_returns = financial_metrics[period].get('benchmark_returns', {})
                    ret = bench_returns.get(bench_name)
                    row += f" {ret*100:+.1f}% |" if ret is not None else " N/A |"
                report.append(row)

            report.append("")

            # Outperformance summary
            report.append("### Outperformance Analysis")
            report.append("")
            report.append("*Positive values mean you outperformed the benchmark*")
            report.append("")

            header = "| vs Benchmark |"
            separator = "|--------------|"
            for period in periods_available:
                header += f" {period} |"
                separator += "------:|"
            report.append(header)
            report.append(separator)

            for bench_name in ["Nifty 50", "Bank Nifty", "Gold (INR)", "S&P 500 (INR)"]:
                row = f"| {bench_name} |"
                for period in periods_available:
                    port_ret = financial_metrics[period].get('portfolio_return')
                    bench_returns = financial_metrics[period].get('benchmark_returns', {})
                    bench_ret = bench_returns.get(bench_name)
                    if port_ret is not None and bench_ret is not None:
                        outperf = (port_ret - bench_ret) * 100
                        emoji = "" if outperf >= 0 else ""
                        row += f" {outperf:+.1f}% |"
                    else:
                        row += " N/A |"
                report.append(row)

            report.append("")
        else:
            report.append("*Insufficient data for benchmark comparison*")
            report.append("")

        # Risk-Adjusted Metrics Section
        report.append("## Risk-Adjusted Metrics")
        report.append("")
        report.append("These metrics help you understand if your returns justify the risks you took.")
        report.append("")

        if periods_available:
            # Metrics explanation at the top
            report.append("### Metric Definitions")
            report.append("")
            report.append("| Metric | What It Measures | Good Value |")
            report.append("|--------|-----------------|------------|")
            report.append("| **Alpha** | Excess return over market after adjusting for risk. Positive alpha = you added value beyond market exposure. | > 0% |")
            report.append("| **Beta** | Portfolio sensitivity to market. Beta=1 means moves with market. Beta>1 = more volatile, Beta<1 = less volatile. | Depends on risk appetite |")
            report.append("| **Sharpe Ratio** | Return per unit of total risk (volatility). Higher = better risk-adjusted returns. | > 1.0 |")
            report.append("| **Sortino Ratio** | Return per unit of downside risk. Like Sharpe but only penalizes losses, not upside volatility. | > 1.5 |")
            report.append("| **Calmar Ratio** | Annual return divided by max drawdown. Higher = better return per unit of drawdown pain. | > 1.0 |")
            report.append("| **Information Ratio** | Excess return vs benchmark per unit of tracking error. Measures skill at beating the benchmark. | > 0.5 |")
            report.append("| **Max Drawdown** | Largest peak-to-trough decline. Shows worst-case loss experienced. | < -20% |")
            report.append("| **Volatility** | Annualized standard deviation of returns. Higher = more price swings. | < 25% |")
            report.append("| **Win Rate** | Percentage of positive days. | > 50% |")
            report.append("")

            # Risk metrics table
            report.append("### Your Risk Metrics")
            report.append("")

            header = "| Metric |"
            separator = "|--------|"
            for period in periods_available:
                header += f" {period} |"
                separator += "------:|"
            report.append(header)
            report.append(separator)

            # Alpha
            row = "| **Alpha** |"
            for period in periods_available:
                alpha = financial_metrics[period].get('alpha')
                row += f" {alpha*100:+.2f}% |" if alpha is not None else " N/A |"
            report.append(row)

            # Beta
            row = "| **Beta** |"
            for period in periods_available:
                beta = financial_metrics[period].get('beta')
                row += f" {beta:.2f} |" if beta is not None else " N/A |"
            report.append(row)

            # Sharpe Ratio
            row = "| **Sharpe Ratio** |"
            for period in periods_available:
                sharpe = financial_metrics[period].get('sharpe_ratio')
                row += f" {sharpe:.2f} |" if sharpe is not None else " N/A |"
            report.append(row)

            # Sortino Ratio
            row = "| **Sortino Ratio** |"
            for period in periods_available:
                sortino = financial_metrics[period].get('sortino_ratio')
                row += f" {sortino:.2f} |" if sortino is not None else " N/A |"
            report.append(row)

            # Calmar Ratio
            row = "| **Calmar Ratio** |"
            for period in periods_available:
                calmar = financial_metrics[period].get('calmar_ratio')
                row += f" {calmar:.2f} |" if calmar is not None else " N/A |"
            report.append(row)

            # Information Ratio
            row = "| **Information Ratio** |"
            for period in periods_available:
                ir = financial_metrics[period].get('information_ratio')
                row += f" {ir:.2f} |" if ir is not None else " N/A |"
            report.append(row)

            # Max Drawdown
            row = "| **Max Drawdown** |"
            for period in periods_available:
                max_dd = financial_metrics[period].get('max_drawdown')
                row += f" {max_dd*100:.1f}% |" if max_dd is not None else " N/A |"
            report.append(row)

            # Volatility
            row = "| **Volatility** |"
            for period in periods_available:
                vol = financial_metrics[period].get('volatility')
                row += f" {vol*100:.1f}% |" if vol is not None else " N/A |"
            report.append(row)

            # Win Rate
            row = "| **Win Rate** |"
            for period in periods_available:
                wr = financial_metrics[period].get('win_rate')
                row += f" {wr*100:.1f}% |" if wr is not None else " N/A |"
            report.append(row)

            # R-Squared
            row = "| **R-Squared** |"
            for period in periods_available:
                r2 = financial_metrics[period].get('r_squared')
                row += f" {r2*100:.1f}% |" if r2 is not None else " N/A |"
            report.append(row)

            report.append("")

            # Interpretation section
            report.append("### Interpretation Guide")
            report.append("")

            # Get 3Y or longest available period for interpretation
            interpret_period = "3Y" if "3Y" in periods_available else periods_available[-1]
            metrics = financial_metrics.get(interpret_period, {})

            alpha = metrics.get('alpha')
            beta = metrics.get('beta')
            sharpe = metrics.get('sharpe_ratio')
            max_dd = metrics.get('max_drawdown')

            report.append(f"Based on your **{interpret_period}** performance:")
            report.append("")

            if alpha is not None:
                if alpha > 0.05:
                    report.append(f"- **Alpha ({alpha*100:+.2f}%)**: Excellent! You're generating significant excess returns beyond market exposure.")
                elif alpha > 0:
                    report.append(f"- **Alpha ({alpha*100:+.2f}%)**: Good. You're adding some value beyond passive market exposure.")
                else:
                    report.append(f"- **Alpha ({alpha*100:+.2f}%)**: Negative alpha suggests you might be better off with index funds.")

            if beta is not None:
                if beta > 1.2:
                    report.append(f"- **Beta ({beta:.2f})**: High beta - your portfolio is more volatile than the market. Higher risk, potentially higher reward.")
                elif beta < 0.8:
                    report.append(f"- **Beta ({beta:.2f})**: Low beta - your portfolio is less volatile than the market. More defensive positioning.")
                else:
                    report.append(f"- **Beta ({beta:.2f})**: Moderate beta - your portfolio moves roughly in line with the market.")

            if sharpe is not None:
                if sharpe > 1.5:
                    report.append(f"- **Sharpe Ratio ({sharpe:.2f})**: Excellent risk-adjusted returns! Well above the typical threshold of 1.0.")
                elif sharpe > 1.0:
                    report.append(f"- **Sharpe Ratio ({sharpe:.2f})**: Good risk-adjusted returns. Above the typical threshold of 1.0.")
                elif sharpe > 0:
                    report.append(f"- **Sharpe Ratio ({sharpe:.2f})**: Positive but modest risk-adjusted returns.")
                else:
                    report.append(f"- **Sharpe Ratio ({sharpe:.2f})**: Negative Sharpe - returns don't justify the volatility. Consider reducing risk.")

            if max_dd is not None:
                if max_dd > -0.10:
                    report.append(f"- **Max Drawdown ({max_dd*100:.1f}%)**: Minimal drawdown - very stable portfolio.")
                elif max_dd > -0.25:
                    report.append(f"- **Max Drawdown ({max_dd*100:.1f}%)**: Moderate drawdown - typical for a diversified equity portfolio.")
                else:
                    report.append(f"- **Max Drawdown ({max_dd*100:.1f}%)**: Significant drawdown - you experienced substantial losses at some point.")

            report.append("")

    # Top Holdings
    if top_holdings:
        report.append("## Top 10 Current Holdings")
        report.append("")
        report.append("| Stock | Qty | Cost Basis | Avg Cost |")
        report.append("|-------|----:|-----------:|---------:|")
        for sym, h in top_holdings:
            avg_cost = h['cost'] / h['qty'] if h['qty'] > 0 else 0
            report.append(f"| {sym} | {h['qty']} | {format_inr(h['cost'])} | {format_inr(avg_cost)} |")
        report.append("")

    # Quarterly Summary
    report.append("## Quarterly Summary")
    report.append("")
    report.append("| Quarter | Deposits | Withdrawals | EQ P&L | F&O P&L | Total P&L | Portfolio |")
    report.append("|---------|----------:|------------:|-------:|--------:|----------:|----------:|")

    for q in sorted(quarterly.keys()):
        qd = quarterly[q]
        eq_pnl = qd['eq_sell'] - qd['eq_buy']
        fo_pnl = qd['fo_sell'] - qd['fo_buy']
        total_pnl = eq_pnl + fo_pnl
        portfolio = qd['cash'] + qd['holdings'] + qd['mf']

        report.append(f"| {q} | {format_inr_lakhs(qd['deposits'])} | {format_inr_lakhs(qd['withdrawals'])} | {format_inr_lakhs(eq_pnl)} | {format_inr_lakhs(fo_pnl)} | {format_inr_lakhs(total_pnl)} | {format_inr_lakhs(portfolio)} |")

    report.append("")

    # Yearly Summary
    report.append("## Yearly Summary")
    report.append("")

    yearly = defaultdict(lambda: {'deposits': 0, 'withdrawals': 0, 'eq_trades': 0, 'fo_trades': 0})
    for q, qd in quarterly.items():
        year = q.split('-')[0]
        yearly[year]['deposits'] += qd['deposits']
        yearly[year]['withdrawals'] += qd['withdrawals']
        yearly[year]['eq_trades'] += qd['eq_buy'] + qd['eq_sell']
        yearly[year]['fo_trades'] += qd['fo_buy'] + qd['fo_sell']
        yearly[year]['portfolio'] = qd['cash'] + qd['holdings'] + qd['mf']

    report.append("| Year | Deposits | Withdrawals | Net Flow | EQ Turnover | F&O Turnover |")
    report.append("|------|----------:|------------:|---------:|------------:|-------------:|")

    for year in sorted(yearly.keys()):
        yd = yearly[year]
        net_flow = yd['deposits'] - yd['withdrawals']
        report.append(f"| {year} | {format_inr_lakhs(yd['deposits'])} | {format_inr_lakhs(yd['withdrawals'])} | {format_inr_lakhs(net_flow)} | {format_inr_lakhs(yd['eq_trades'])} | {format_inr_lakhs(yd['fo_trades'])} |")

    report.append("")

    # Equity Transactions Table
    report.append("## Equity Transactions")
    report.append("")
    if eq_trades:
        # Group trades by symbol and match opens with closes
        eq_by_symbol = defaultdict(list)
        for t in sorted(eq_trades, key=lambda x: x['order_execution_time']):
            eq_by_symbol[t['tradingsymbol']].append(t)

        report.append("| Symbol | Open Date | Qty | Price | Total | Close Date | Qty | Price | Total | P&L |")
        report.append("|--------|-----------|----:|------:|------:|------------|----:|------:|------:|----:|")

        for symbol in sorted(eq_by_symbol.keys()):
            trades = eq_by_symbol[symbol]
            buys = []
            for t in trades:
                if t['trade_type'] == 'buy':
                    buys.append({'date': t['trade_date'], 'qty': t['quantity'], 'price': t['price']})
                else:  # sell
                    sell_qty = t['quantity']
                    sell_price = t['price']
                    sell_date = t['trade_date']
                    sell_total = sell_qty * sell_price

                    # Match with buys (FIFO)
                    while sell_qty > 0 and buys:
                        buy = buys[0]
                        matched_qty = min(sell_qty, buy['qty'])
                        buy_total = matched_qty * buy['price']
                        matched_sell_total = matched_qty * sell_price
                        pnl = matched_sell_total - buy_total

                        report.append(
                            f"| {symbol} | {buy['date']} | {matched_qty} | {buy['price']:.2f} | {format_inr(buy_total)} | "
                            f"{sell_date} | {matched_qty} | {sell_price:.2f} | {format_inr(matched_sell_total)} | {format_inr(pnl)} |"
                        )

                        buy['qty'] -= matched_qty
                        sell_qty -= matched_qty
                        if buy['qty'] == 0:
                            buys.pop(0)

            # Show remaining open positions
            for buy in buys:
                if buy['qty'] > 0:
                    buy_total = buy['qty'] * buy['price']
                    report.append(
                        f"| {symbol} | {buy['date']} | {buy['qty']} | {buy['price']:.2f} | {format_inr(buy_total)} | "
                        f"- | - | - | - | *Open* |"
                    )
    else:
        report.append("*No equity transactions*")
    report.append("")

    # F&O Transactions Table
    report.append("## F&O Transactions")
    report.append("")
    if fo_trades:
        # Group trades by symbol
        fo_by_symbol = defaultdict(list)
        for t in sorted(fo_trades, key=lambda x: x['order_execution_time']):
            fo_by_symbol[t['tradingsymbol']].append(t)

        report.append("| Symbol | Open Date | Type | Qty | Price | Total | Close Date | Qty | Price | Total | P&L |")
        report.append("|--------|-----------|------|----:|------:|------:|------------|----:|------:|------:|----:|")

        for symbol in sorted(fo_by_symbol.keys()):
            trades = fo_by_symbol[symbol]
            # For F&O, track net position
            opens = []  # list of {date, type, qty, price}

            for t in trades:
                qty = t['quantity']
                price = t['price']
                date = t['trade_date']
                trade_type = t['trade_type']

                # Check if this closes an existing position
                if opens and opens[0]['type'] != trade_type:
                    # Closing trade
                    close_qty = qty
                    while close_qty > 0 and opens and opens[0]['type'] != trade_type:
                        open_pos = opens[0]
                        matched_qty = min(close_qty, open_pos['qty'])
                        open_total = matched_qty * open_pos['price']
                        close_total = matched_qty * price

                        if open_pos['type'] == 'buy':
                            pnl = close_total - open_total
                        else:
                            pnl = open_total - close_total

                        report.append(
                            f"| {symbol} | {open_pos['date']} | {open_pos['type']} | {matched_qty} | {open_pos['price']:.2f} | {format_inr(open_total)} | "
                            f"{date} | {matched_qty} | {price:.2f} | {format_inr(close_total)} | {format_inr(pnl)} |"
                        )

                        open_pos['qty'] -= matched_qty
                        close_qty -= matched_qty
                        if open_pos['qty'] == 0:
                            opens.pop(0)

                    # If there's remaining qty, it's a new position
                    if close_qty > 0:
                        opens.append({'date': date, 'type': trade_type, 'qty': close_qty, 'price': price})
                else:
                    # Opening or adding to position
                    opens.append({'date': date, 'type': trade_type, 'qty': qty, 'price': price})

            # Show remaining open positions
            for pos in opens:
                if pos['qty'] > 0:
                    total = pos['qty'] * pos['price']
                    report.append(
                        f"| {symbol} | {pos['date']} | {pos['type']} | {pos['qty']} | {pos['price']:.2f} | {format_inr(total)} | "
                        f"- | - | - | - | *Open* |"
                    )
    else:
        report.append("*No F&O transactions*")
    report.append("")

    # Deposits Table
    report.append("## Deposits")
    report.append("")
    if deposits_by_date:
        report.append("| Date | Amount (₹) |")
        report.append("|------|-------:|")
        for date in sorted(deposits_by_date.keys()):
            amount = deposits_by_date[date]
            report.append(f"| {date} | {amount:,.0f} |")
        report.append(f"| **Total** | **{total_deposits:,.0f}** |")
    else:
        report.append("*No deposits*")
    report.append("")

    # Withdrawals Table
    report.append("## Withdrawals")
    report.append("")
    if withdrawals_by_date:
        report.append("| Date | Amount (₹) |")
        report.append("|------|-------:|")
        for date in sorted(withdrawals_by_date.keys()):
            amount = withdrawals_by_date[date]
            report.append(f"| {date} | {amount:,.0f} |")
        report.append(f"| **Total** | **{total_withdrawals:,.0f}** |")
    else:
        report.append("*No withdrawals*")
    report.append("")

    # Detailed Quarterly Breakdown (at the end)
    report.append("## Detailed Quarterly Breakdown")
    report.append("")

    cumulative_deposits = 0
    cumulative_withdrawals = 0

    for q in sorted(quarterly.keys()):
        qd = quarterly[q]
        cumulative_deposits += qd['deposits']
        cumulative_withdrawals += qd['withdrawals']
        cumulative_invested = cumulative_deposits - cumulative_withdrawals
        portfolio = qd['cash'] + qd['holdings'] + qd['mf']

        if portfolio > 0 or qd['deposits'] > 0:
            gain = portfolio - cumulative_invested if portfolio > 0 else 0
            gain_pct = (gain / cumulative_invested * 100) if cumulative_invested > 0 else 0

            report.append(f"### {q}")
            report.append("")
            report.append("| Metric | Value |")
            report.append("|--------|------:|")
            report.append(f"| Deposits | {format_inr(qd['deposits'])} |")
            report.append(f"| Withdrawals | {format_inr(qd['withdrawals'])} |")
            report.append(f"| EQ Buy | {format_inr(qd['eq_buy'])} |")
            report.append(f"| EQ Sell | {format_inr(qd['eq_sell'])} |")
            report.append(f"| Options Buy | {format_inr(qd['fo_opt_buy'])} |")
            report.append(f"| Options Sell | {format_inr(qd['fo_opt_sell'])} |")
            report.append(f"| Futures Buy | {format_inr(qd['fo_fut_buy'])} |")
            report.append(f"| Futures Sell | {format_inr(qd['fo_fut_sell'])} |")
            if portfolio > 0:
                report.append(f"| Cash | {format_inr(qd['cash'])} |")
                report.append(f"| Holdings | {format_inr(qd['holdings'])} |")
                report.append(f"| MF | {format_inr(qd['mf'])} |")
                report.append(f"| **Portfolio Value** | **{format_inr(portfolio)}** |")
                report.append(f"| Cumulative Invested | {format_inr(cumulative_invested)} |")
                report.append(f"| **Gain/Loss** | **{format_inr(gain)}** |")
                report.append(f"| **Return** | **{gain_pct:+.1f}%** |")
            report.append("")

    # Footer
    report.append("---")
    report.append("")
    report.append("*Report generated by zerodha_account_history.py*")
    report.append("")

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Generate Zerodha account history report")
    parser.add_argument("-u", "--user", help="User ID (data stored in tmp/<user>/). Auto-detected when using --fetch")
    parser.add_argument("-o", "--output", help="Output markdown file (default: tmp/<user>/<user_id>_<name>.md)")
    parser.add_argument("--fetch", action="store_true", help="Fetch fresh data from Zerodha (opens browser for manual login)")
    parser.add_argument("--from-year", type=int, default=2020, help="Start year for fetching data (default: 2020)")
    parser.add_argument("--no-benchmarks", action="store_true", help="Skip fetching benchmark data (faster, but no comparisons)")
    args = parser.parse_args()

    user = args.user

    if args.fetch:
        print("Fetching fresh data from Zerodha Console...")
        print(f"Fetching from {args.from_year} to {datetime.now().year}")
        print("=" * 50)
        user, _ = download_all_data(start_year=args.from_year)
        print("=" * 50)
        print()
    elif not user:
        parser.error("--user is required when not using --fetch")

    print(f"Loading data for user: {user}...")
    data = load_data(user)

    if not data['EQ'] and not data['FO'] and not data['value']:
        print(f"Error: No data files found for user '{user}'. Run with --fetch to download data.")
        return

    # Get user name from profile for filename
    profile = data.get('profile', {})
    user_name = profile.get('user_name') or profile.get('name') or profile.get('user_shortname') or 'Unknown'
    # Sanitize user_name for filename (remove special chars, replace spaces with underscore)
    safe_name = re.sub(r'[^\w\s-]', '', user_name).strip().replace(' ', '_')

    # Set default output path based on user
    if args.output is None:
        args.output = str(get_user_data_dir(user) / f"{user}_{safe_name}.md")

    print(f"  EQ trades: {len(data['EQ'])}")
    print(f"  FO trades: {len(data['FO'])}")
    print(f"  Value records: {len(data['value'])}")
    print()

    # Fetch benchmark data for comparisons
    benchmarks_data = {}
    if not args.no_benchmarks and data['value']:
        # Determine date range from data
        eq_values = [r for r in data['value'] if r['segment'] == 'EQ']
        if eq_values:
            dates = [r['trade_date'] for r in eq_values]
            start_date = min(dates)
            end_date = max(dates)
            # Add buffer for benchmark data
            start_dt = datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=30)
            end_dt = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=7)
            benchmarks_data = fetch_benchmark_data(start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d'), user)
            print()

    print("Analyzing account history...")
    report = generate_report(data, user, benchmarks_data)

    print(f"Writing report to {args.output}...")
    with open(args.output, 'w') as f:
        f.write(report)

    print(f"\nDone! Report saved to {args.output}")


if __name__ == "__main__":
    main()
