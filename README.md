# Zerodha History

**Finally know the truth about your trading journey.**

Are you profitable? Have you beaten the market? Or has your trading been slowly bleeding money while you weren't looking?

This tool generates a **comprehensive, brutally honest report** of your entire Zerodha trading history. No sugar-coating. No hiding. Just the cold, hard numbers.

---

## What You'll Discover

- **The Real P&L** - Not just what Zerodha shows you, but a complete breakdown of realized vs unrealized gains across equity AND F&O
- **Your Money Flow** - Every rupee deposited, every rupee withdrawn, and what's left
- **The Full Picture** - Quarterly and yearly summaries that reveal patterns you never noticed
- **Every Single Trade** - Complete transaction history with entry/exit prices and P&L per trade
- **Physical Deliveries** - F&O positions that resulted in stock delivery? We track those too
- **Benchmark Comparisons** - Did you beat Nifty 50? Bank Nifty? Gold? S&P 500? See exactly where you stand
- **Risk-Adjusted Metrics** - Alpha, Beta, Sharpe Ratio, Sortino Ratio, Max Drawdown and more - the metrics pros use
- **XIRR Calculation** - True annualized returns accounting for timing of deposits/withdrawals
- **Beautiful HTML Reports** - Styled reports with green/red profit/loss coloring

---

## Sample Report Output

```
┌─────────────────────────────────────────────────────────┐
│                   EXECUTIVE SUMMARY                      │
├─────────────────────────────────────────────────────────┤
│  Total Deposited        ₹25,00,000                      │
│  Total Withdrawn        ₹8,50,000                       │
│  Net Invested           ₹16,50,000                      │
│  Current Value          ₹21,34,567                      │
│  Overall Gain/Loss      ₹4,84,567                       │
│  Absolute Return        +29.4%                          │
│  XIRR (Annualized)      +12.8%                          │
└─────────────────────────────────────────────────────────┘
```

### Profit & Loss Breakdown

| Category | Amount |
|----------|-------:|
| EQ Realized P&L | ₹2,45,320 |
| EQ Unrealized P&L | ₹1,87,450 |
| Futures P&L | ₹85,200 |
| Options P&L | -₹33,403 |
| **Total Realized** | **₹2,97,117** |

### Quarterly Trends

| Quarter | Deposits | Withdrawals | Portfolio Value |
|---------|----------|-------------|-----------------|
| 2023-Q1 | ₹2.50L | ₹0.00L | ₹12.45L |
| 2023-Q2 | ₹1.00L | ₹0.50L | ₹14.23L |
| 2023-Q3 | ₹0.00L | ₹2.00L | ₹15.67L |
| 2023-Q4 | ₹3.00L | ₹0.00L | ₹18.90L |

### Performance vs Benchmarks

| Asset | 1Y | 3Y | 5Y | All Time |
|-------|---:|---:|---:|---------:|
| **Your Portfolio** | **+18.5%** | **+52.3%** | **+89.2%** | **+112.4%** |
| Nifty 50 | +12.3% | +38.7% | +72.4% | +95.6% |
| Bank Nifty | +8.7% | +25.4% | +58.9% | +78.3% |
| Gold (INR) | +15.2% | +42.1% | +68.3% | +82.1% |
| S&P 500 (INR) | +22.1% | +48.9% | +95.7% | +124.5% |

*Only periods shorter than your account age are shown.*

### Risk-Adjusted Metrics

| Metric | 1Y | 3Y | 5Y | All Time |
|--------|---:|---:|---:|---------:|
| **Alpha** | +5.23% | +4.12% | +3.87% | +3.94% |
| **Beta** | 0.92 | 0.88 | 0.85 | 0.87 |
| **Sharpe Ratio** | 1.24 | 1.18 | 1.32 | 1.28 |
| **Sortino Ratio** | 1.67 | 1.52 | 1.71 | 1.65 |
| **Max Drawdown** | -12.3% | -18.7% | -24.5% | -24.5% |
| **Volatility** | 18.4% | 19.2% | 20.1% | 19.8% |

---

## Installation

### Prerequisites

- Python 3.10 or higher

### Option A: Using uv (Recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager that handles everything automatically.

**Install uv:**

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Clone and run:**

```bash
git clone https://github.com/anuphw/zerodha_history.git
cd zerodha_history
uv sync
uv run playwright install chromium
uv run python zerodha_history.py --fetch
```

Or just use the easy run scripts after cloning:
- **Windows:** Double-click `run.bat`
- **macOS/Linux:** Run `./run.sh`

### Option B: Using pip

```bash
git clone https://github.com/anuphw/zerodha_history.git
cd zerodha_history
pip install -r requirements.txt
playwright install chromium
python zerodha_history.py --fetch
```

---

## Usage

> **Note:** Replace `uv run python` with just `python` if using pip.

### Fetch Data & Generate Report (Recommended)

```bash
uv run python zerodha_history.py --fetch
# or with pip: python zerodha_history.py --fetch
```

This will:
1. Open a browser window
2. Navigate to Zerodha Kite login
3. Wait for you to complete login (including 2FA)
4. Automatically fetch all your trading data
5. Generate comprehensive Markdown and HTML reports

### Generate Report from Existing Data

If you've already fetched data before:

```bash
uv run python zerodha_history.py -u YOUR_USER_ID
```

### Custom Date Range

By default, data is fetched from 2020 onwards. To fetch from an earlier year:

```bash
uv run python zerodha_history.py --fetch --from-year 2018
```

### Custom Output File

```bash
uv run python zerodha_history.py --fetch -o my_report.md
```

---

## Command Line Options

| Option | Description |
|--------|-------------|
| `--fetch` | Fetch fresh data from Zerodha (opens browser for login) |
| `-u, --user` | User ID (required when not using --fetch) |
| `-o, --output` | Output file path (default: `tmp/<user>/<user_id>_<name>.md`) |
| `--from-year` | Start year for fetching data (default: 2020) |
| `--no-benchmarks` | Skip fetching benchmark data (faster, but no market comparisons) |

---

## How It Works

1. **Secure Login** - Opens a real browser for you to log in. Your credentials never touch this script.

2. **Data Collection** - Fetches your complete trading history from Zerodha Console:
   - Equity (EQ) trades
   - Futures & Options (F&O) trades
   - Account values over time
   - Profile information

3. **Benchmark Fetching** - Downloads historical data from Yahoo Finance for:
   - Nifty 50 (Indian large-cap index)
   - Bank Nifty (Banking sector index)
   - Nifty IT (Technology sector index)
   - Gold (converted to INR)
   - S&P 500 (converted to INR)

4. **FIFO Analysis** - Calculates realized P&L using First-In-First-Out matching, just like tax authorities expect.

5. **Physical Delivery Detection** - Identifies when your F&O positions resulted in stock delivery and traces them through to sale.

6. **Risk Metrics Calculation** - Computes professional-grade metrics:
   - Time-Weighted Returns (adjusted for deposits/withdrawals)
   - Alpha and Beta vs market indices
   - Sharpe and Sortino ratios
   - Maximum drawdown analysis
   - Win rate and volatility

7. **Report Generation** - Creates both Markdown and styled HTML reports with green/red color coding for profits/losses.

---

## Data Storage

All data is stored locally in the `tmp/<user_id>/` directory:

```
tmp/
└── AB1234/
    ├── EQ.jsonl          # Equity trades
    ├── FO.jsonl          # F&O trades
    ├── value.jsonl       # Account values
    ├── benchmarks.json   # Cached benchmark data
    ├── profile.json      # User profile
    ├── AB1234_Name.md    # Markdown report
    └── AB1234_Name.html  # Styled HTML report
```

**Your data never leaves your machine.** This tool only communicates with Zerodha's servers using your authenticated session.

---

## Report Sections

The generated report includes:

1. **Profile** - Account details and report metadata
2. **Executive Summary** - The bottom line at a glance
3. **Current Portfolio** - Cash, holdings, and mutual funds breakdown
4. **The Account Story** - A narrative of your trading journey
5. **Profit & Loss Summary** - Detailed P&L across all segments
6. **Key Metrics** - Important numbers every trader should know
7. **Performance vs Benchmarks** - Compare your returns against Nifty, Gold, S&P 500 over 1Y, 3Y, 5Y, 7Y, 10Y
8. **Risk-Adjusted Metrics** - Alpha, Beta, Sharpe, Sortino, Max Drawdown with explanations
9. **Top 10 Holdings** - Your largest current positions
10. **Quarterly Summary** - Performance trends over time
11. **Yearly Summary** - Year-over-year comparison
12. **Equity Transactions** - Complete EQ trade log with P&L
13. **F&O Transactions** - Complete F&O trade log with P&L
14. **Deposits** - All money added to the account
15. **Withdrawals** - All money taken out
16. **Detailed Quarterly Breakdown** - Deep dive into each quarter

---

## Understanding the Metrics

The report includes professional-grade financial metrics. Here's what they mean:

### Performance Metrics

| Metric | What It Tells You |
|--------|-------------------|
| **Absolute Return** | Simple return: (Current Value - Net Invested) / Net Invested |
| **XIRR** | Annualized return accounting for timing of each deposit/withdrawal. This is the true measure of your performance. |
| **Portfolio Return** | Your total return over the period, adjusted for deposits and withdrawals |
| **Benchmark Return** | How the market index performed over the same period |
| **Outperformance** | Difference between your return and the benchmark - positive = you beat the market |

### Risk-Adjusted Metrics

| Metric | What It Tells You | Good Value |
|--------|-------------------|------------|
| **Alpha** | Excess return you generated beyond what the market gave you. Positive alpha means your stock picking added value. | > 0% |
| **Beta** | How much your portfolio moves with the market. Beta=1 means 1:1 with market. Beta=0.5 means half as volatile. | Depends on risk appetite |
| **Sharpe Ratio** | Return per unit of risk. Higher is better - you're getting more return for the volatility you're taking. | > 1.0 is good, > 2.0 is excellent |
| **Sortino Ratio** | Like Sharpe but only counts downside volatility. Better for most investors since upside volatility is good. | > 1.5 |
| **Calmar Ratio** | Return divided by max drawdown. How much return you get per unit of "pain". | > 1.0 |
| **Information Ratio** | How consistently you beat the benchmark. Higher = more skill, less luck. | > 0.5 |
| **Max Drawdown** | Your worst peak-to-trough loss. If this is -30%, you once saw your portfolio drop 30% from its high. | > -25% |
| **Volatility** | How much your portfolio swings. Higher = more uncertainty. | < 25% for most investors |
| **Win Rate** | Percentage of profitable days. | > 50% |
| **R-Squared** | How much of your portfolio's movement is explained by the market. Low R² = more independent bets. | Context dependent |

### Time Period Comparison

The report shows metrics for **1Y, 3Y, 5Y, 7Y, 10Y, and All Time** - but only periods shorter than your account age are displayed. This helps you see:

- **Short-term (1Y)**: Recent performance, could be luck or skill
- **Medium-term (3Y-5Y)**: More reliable signal about your strategy
- **Long-term (7Y+)**: True test of your approach through multiple market cycles
- **All Time**: Your complete trading history since account opening

---

## FAQ

### Is this safe to use?

Yes. This tool:
- Opens a real browser for you to log in manually
- Never stores or transmits your password
- Only reads data from your account
- Stores everything locally on your machine

### Why do I need to log in manually?

Zerodha uses 2FA (TOTP) and other security measures. Manual login ensures you maintain full control of your credentials and the tool remains compatible with any authentication changes.

### Can I use this for tax filing?

The data can help you understand your trading activity, but always verify against your official Contract Notes and tax statements from Zerodha for actual tax filing.

### The numbers don't match Zerodha Console exactly?

Small differences can occur due to:
- Brokerage and taxes (not deducted in this report)
- Timing of data snapshots
- Corporate actions (splits, bonuses)

This report focuses on gross trade P&L before charges.

---

## License

MIT License - Do whatever you want with it.

---

## Disclaimer

This tool is for **personal use only**. The author is not responsible for any financial decisions made based on reports generated by this tool. Always verify important financial data with official sources.

Not affiliated with Zerodha. Zerodha and Kite are trademarks of Zerodha Broking Ltd.

---

**Stop wondering. Start knowing.**

```bash
uv run python zerodha_history.py --fetch
```
