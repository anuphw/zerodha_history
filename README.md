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
│  Return                 +29.4%                          │
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

---

## Installation

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) - Fast Python package manager

### Quick Start

```bash
# Clone the repository
git clone https://github.com/anuphw/zerodha_history.git
cd zerodha_history

# Install dependencies with uv
uv sync

# Install Playwright browsers (required for login)
uv run playwright install chromium
```

---

## Usage

### Fetch Data & Generate Report (Recommended)

```bash
uv run python zerodha_history.py --fetch
```

This will:
1. Open a browser window
2. Navigate to Zerodha Kite login
3. Wait for you to complete login (including 2FA)
4. Automatically fetch all your trading data
5. Generate a comprehensive markdown report

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

---

## How It Works

1. **Secure Login** - Opens a real browser for you to log in. Your credentials never touch this script.

2. **Data Collection** - Fetches your complete trading history from Zerodha Console:
   - Equity (EQ) trades
   - Futures & Options (F&O) trades
   - Account values over time
   - Profile information

3. **FIFO Analysis** - Calculates realized P&L using First-In-First-Out matching, just like tax authorities expect.

4. **Physical Delivery Detection** - Identifies when your F&O positions resulted in stock delivery and traces them through to sale.

5. **Report Generation** - Creates a detailed markdown report you can view in any text editor or render beautifully on GitHub.

---

## Data Storage

All data is stored locally in the `tmp/<user_id>/` directory:

```
tmp/
└── AB1234/
    ├── EQ.jsonl      # Equity trades
    ├── FO.jsonl      # F&O trades
    ├── value.jsonl   # Account values
    ├── profile.json  # User profile
    └── AB1234_Name.md   # Generated report
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
7. **Top 10 Holdings** - Your largest current positions
8. **Quarterly Summary** - Performance trends over time
9. **Yearly Summary** - Year-over-year comparison
10. **Equity Transactions** - Complete EQ trade log with P&L
11. **F&O Transactions** - Complete F&O trade log with P&L
12. **Deposits** - All money added to the account
13. **Withdrawals** - All money taken out
14. **Detailed Quarterly Breakdown** - Deep dive into each quarter

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
