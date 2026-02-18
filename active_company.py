import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

active_companies = pd.read_excel(
    "Python_data.xlsx",
    sheet_name="Sheet1",
    decimal=","
)

# Aritmetiska returns
returns = active_companies.pct_change()
returns = returns.replace([np.inf, -np.inf], np.nan)
all_returns = returns.stack()

# Klipp extrema värden (för tydligare histogram)
lo, hi = all_returns.quantile([0.001, 0.99])
returns_clip = all_returns.clip(lo, hi)

plt.figure(figsize=(10,6))
plt.hist(returns_clip, bins=50)
plt.title("Histogram: 1-period aritmetiska returns (alla bolag)")
plt.xlabel("Return")
plt.ylabel("Antal observationer")
plt.show()


# -----------------------------
# Log-returns
# -----------------------------
log_returns = np.log(active_companies / active_companies.shift(1))
log_returns = log_returns.replace([np.inf, -np.inf], np.nan)
all_log = log_returns.stack()

# Klipp även här för jämförbarhet
lo_log, hi_log = all_log.quantile([0.001, 0.999])
log_clip = all_log.clip(lo_log, hi_log)

plt.figure(figsize=(10,6))
plt.hist(log_clip, bins=50)
plt.title("Histogram: 1-period log-returns (alla bolag)")
plt.xlabel("Log-return")
plt.ylabel("Antal observationer")
plt.show()