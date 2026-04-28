import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize

# =========================================================
# SETTINGS
# =========================================================

FILE_NAME = "MCReturns.xlsx"
MC_SHEET = "Sheet1"
RETURNS_SHEET = "Returns"

START_DATE = "1982-01-01"
END_DATE = "2026-01-01"
PORTFOLIO_START_DATE = "1996-01-01"

PORTFOLIO_SIZES = [10, 50, 100]
CAP_SEGMENTS = ["small", "mid", "large"]
WEIGHTING_METHODS = ["equal", "market_cap", "risk_parity"]

SELECTION_STRATEGIES = ["high_skew", "low_skew", "momentum"]

SKEW_LOOKBACK = 60
SKEW_MIN_OBS = 60

MOMENTUM_LOOKBACK = 12
MOMENTUM_MIN_OBS = 12

RISK_PARITY_LOOKBACK = 36
RISK_PARITY_MIN_OBS = 24

INITIAL_WEALTH = 1.0

OUTPUT_FOLDER = "RuleBasedResults"


# =========================================================
# 1. LOAD DATA
# =========================================================

def load_data(file_name, mc_sheet, returns_sheet, start_date, end_date):
    mc = pd.read_excel(file_name, sheet_name=mc_sheet, na_values=["NA", "#VALUE!"])
    rets = pd.read_excel(file_name, sheet_name=returns_sheet, na_values=["NA", "#VALUE!"])

    dates = pd.date_range(start=start_date, end=end_date, freq="MS")

    if len(mc) != len(dates):
        raise ValueError(
            f"MC-sheet har {len(mc)} rader, men datumintervallet ger {len(dates)} månader."
        )

    if len(rets) != len(dates):
        raise ValueError(
            f"Returns-sheet har {len(rets)} rader, men datumintervallet ger {len(dates)} månader."
        )

    mc.columns = mc.columns.map(lambda x: str(x).strip())
    rets.columns = rets.columns.map(lambda x: str(x).strip())

    mc.index = dates
    rets.index = dates

    mc = mc.apply(pd.to_numeric, errors="coerce")
    rets = rets.apply(pd.to_numeric, errors="coerce")

    common_cols = mc.columns.intersection(rets.columns)

    if len(common_cols) == 0:
        raise ValueError("Inga gemensamma bolagskolumner hittades mellan MC och Returns.")

    mc = mc[common_cols].copy()
    rets = rets[common_cols].copy()
    rets = rets[mc.columns]

    print(f"Antal gemensamma bolag: {len(common_cols)}")
    print(f"Första datum: {mc.index.min().date()}, sista datum: {mc.index.max().date()}")

    return mc, rets


# =========================================================
# 2. REBALANCE DATES
# =========================================================

def get_rebalance_dates(index, start_date):
    start_date = pd.Timestamp(start_date)
    return pd.DatetimeIndex([d for d in index if d >= start_date and d.month == 1])


# =========================================================
# 3. MARKET CAP CLASSIFICATION
# =========================================================

def classify_market_caps(mc_row):
    mc_row = mc_row.dropna()
    mc_row = mc_row[mc_row > 0].sort_values()

    n = len(mc_row)

    if n < 3:
        return {"small": [], "mid": [], "large": []}

    small = mc_row.index[: n // 3].tolist()
    mid = mc_row.index[n // 3 : 2 * n // 3].tolist()
    large = mc_row.index[2 * n // 3 :].tolist()

    return {
        "small": small,
        "mid": mid,
        "large": large
    }


# =========================================================
# 4. WEIGHTING METHODS
# =========================================================

def equal_weights(tickers):
    n = len(tickers)

    if n == 0:
        return None

    return pd.Series(np.repeat(1.0 / n, n), index=tickers)


def market_cap_weights(mc_row, tickers):
    vals = mc_row[tickers].clip(lower=0)
    total = vals.sum()

    if total <= 0:
        return None

    return vals / total


def risk_contributions(weights, cov_matrix):
    w = np.asarray(weights, dtype=float)
    port_vol = np.sqrt(w @ cov_matrix @ w)

    if port_vol <= 0:
        return np.zeros(len(w))

    marginal = cov_matrix @ w
    return w * marginal / port_vol


def risk_parity_weights(cov_matrix, tickers):
    n = len(tickers)

    if n == 1:
        return pd.Series([1.0], index=tickers)

    cov_matrix = np.asarray(cov_matrix, dtype=float)
    cov_matrix = cov_matrix + np.eye(n) * 1e-8

    x0 = np.repeat(1.0 / n, n)

    def objective(w):
        rc = risk_contributions(w, cov_matrix)
        target = np.repeat(np.mean(rc), n)
        return np.sum((rc - target) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(1e-8, 1.0) for _ in range(n)]

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 100, "ftol": 1e-8, "disp": False}
    )

    if not result.success:
        return None

    w = result.x / result.x.sum()

    return pd.Series(w, index=tickers)


def estimate_covariance(returns_df, tickers, rebalance_date, lookback_months, min_obs):
    start_date = rebalance_date - pd.DateOffset(months=lookback_months)

    window = returns_df.loc[
        (returns_df.index < rebalance_date) &
        (returns_df.index >= start_date),
        tickers
    ].copy()

    valid_counts = window.notna().sum()
    valid_tickers = valid_counts[valid_counts >= min_obs].index.tolist()

    if len(valid_tickers) != len(tickers):
        return None, []

    window = window[valid_tickers].dropna(how="any")

    if len(window) < min_obs:
        return None, []

    cov = window.cov().values

    return cov, valid_tickers


def get_weights(weighting_method, tickers, mc_row, returns_df, rebalance_date):
    if weighting_method == "equal":
        return equal_weights(tickers)

    if weighting_method == "market_cap":
        return market_cap_weights(mc_row, tickers)

    if weighting_method == "risk_parity":
        cov, rp_tickers = estimate_covariance(
            returns_df=returns_df,
            tickers=tickers,
            rebalance_date=rebalance_date,
            lookback_months=RISK_PARITY_LOOKBACK,
            min_obs=RISK_PARITY_MIN_OBS
        )

        if cov is None or len(rp_tickers) != len(tickers):
            return None

        return risk_parity_weights(cov, rp_tickers)

    raise ValueError(f"Okänd viktmetod: {weighting_method}")


# =========================================================
# 5. RULE-BASED SELECTION
# =========================================================

def get_lookback_window(returns_df, universe, rebalance_date, lookback_months):
    start_date = rebalance_date - pd.DateOffset(months=lookback_months)

    return returns_df.loc[
        (returns_df.index < rebalance_date) &
        (returns_df.index >= start_date),
        universe
    ].copy()


def select_by_high_skew(universe, returns_df, rebalance_date, size):
    window = get_lookback_window(
        returns_df=returns_df,
        universe=universe,
        rebalance_date=rebalance_date,
        lookback_months=SKEW_LOOKBACK
    )

    valid_counts = window.notna().sum()
    valid_tickers = valid_counts[valid_counts >= SKEW_MIN_OBS].index.tolist()

    if len(valid_tickers) < size:
        return None

    skew_values = window[valid_tickers].skew(skipna=True)
    skew_values = skew_values.dropna().sort_values(ascending=False)

    if len(skew_values) < size:
        return None

    return skew_values.index[:size].tolist()


def select_by_low_skew(universe, returns_df, rebalance_date, size):
    window = get_lookback_window(
        returns_df=returns_df,
        universe=universe,
        rebalance_date=rebalance_date,
        lookback_months=SKEW_LOOKBACK
    )

    valid_counts = window.notna().sum()
    valid_tickers = valid_counts[valid_counts >= SKEW_MIN_OBS].index.tolist()

    if len(valid_tickers) < size:
        return None

    skew_values = window[valid_tickers].skew(skipna=True)
    skew_values = skew_values.dropna().sort_values(ascending=True)

    if len(skew_values) < size:
        return None

    return skew_values.index[:size].tolist()


def select_by_momentum(universe, returns_df, rebalance_date, size):
    window = get_lookback_window(
        returns_df=returns_df,
        universe=universe,
        rebalance_date=rebalance_date,
        lookback_months=MOMENTUM_LOOKBACK
    )

    valid_counts = window.notna().sum()
    valid_tickers = valid_counts[valid_counts >= MOMENTUM_MIN_OBS].index.tolist()

    if len(valid_tickers) < size:
        return None

    window = window[valid_tickers]

    momentum = (1.0 + window).prod(axis=0) - 1.0
    momentum = momentum.dropna().sort_values(ascending=False)

    if len(momentum) < size:
        return None

    return momentum.index[:size].tolist()


def select_stocks(strategy, universe, returns_df, rebalance_date, size):
    if strategy == "high_skew":
        return select_by_high_skew(universe, returns_df, rebalance_date, size)

    if strategy == "low_skew":
        return select_by_low_skew(universe, returns_df, rebalance_date, size)

    if strategy == "momentum":
        return select_by_momentum(universe, returns_df, rebalance_date, size)

    raise ValueError(f"Okänd strategi: {strategy}")


# =========================================================
# 6. MONTHLY WEALTH EVOLUTION
# =========================================================

def evolve_portfolio_one_year(positions, returns_df, start_date, end_date):
    positions = positions.copy().astype(float)
    tickers = positions.index.tolist()

    period_returns = returns_df.loc[
        (returns_df.index > start_date) &
        (returns_df.index <= end_date),
        tickers
    ].copy()

    records = []

    active_status = {ticker: True for ticker in tickers}

    for dt, row in period_returns.iterrows():
        beginning_wealth = positions.sum()

        bankrupt_this_month = 0
        acquired_this_month = 0

        for ticker in tickers:
            r = row[ticker]

            if not active_status[ticker]:
                continue

            if pd.isna(r):
                if positions[ticker] > 0:
                    active_status[ticker] = False
                    acquired_this_month += 1
                continue

            positions[ticker] = positions[ticker] * (1.0 + r)

            if r == -1:
                positions[ticker] = 0.0
                active_status[ticker] = False
                bankrupt_this_month += 1

        ending_wealth = positions.sum()

        if beginning_wealth > 0:
            portfolio_return = ending_wealth / beginning_wealth - 1.0
        else:
            portfolio_return = np.nan

        if ending_wealth > 0:
            current_weights = positions / ending_wealth
            largest_weight = float(current_weights.max())
            active_holdings = int((positions > 0).sum())
        else:
            largest_weight = np.nan
            active_holdings = 0

        records.append({
            "date": dt,
            "beginning_wealth": float(beginning_wealth),
            "ending_wealth": float(ending_wealth),
            "monthly_return": float(portfolio_return) if pd.notna(portfolio_return) else np.nan,
            "n_holdings": len(positions),
            "n_active_holdings_end_month": active_holdings,
            "largest_weight_end_month": largest_weight,
            "bankruptcies_this_month": bankrupt_this_month,
            "acquisitions_this_month": acquired_this_month
        })

    return positions, records


# =========================================================
# 7. BUILD RULE-BASED BACKTEST
# =========================================================

def build_rule_based_backtest(mc_df, returns_df):
    rebalance_dates = get_rebalance_dates(mc_df.index, PORTFOLIO_START_DATE)

    constituents_records = []
    monthly_records = []
    annual_rebalance_records = []

    current_positions = {}

    for t in range(len(rebalance_dates) - 1):
        reb_date = rebalance_dates[t]
        next_reb_date = rebalance_dates[t + 1]

        print(f"Rebalanserar: {reb_date.date()}")

        mc_row = mc_df.loc[reb_date]
        segments = classify_market_caps(mc_row)

        for strategy in SELECTION_STRATEGIES:
            for cap in CAP_SEGMENTS:
                universe = segments[cap]

                for size in PORTFOLIO_SIZES:
                    selected = select_stocks(
                        strategy=strategy,
                        universe=universe,
                        returns_df=returns_df,
                        rebalance_date=reb_date,
                        size=size
                    )

                    if selected is None:
                        continue

                    for weighting in WEIGHTING_METHODS:
                        portfolio_group = f"{strategy}_{cap}_{size}_{weighting}"

                        if t == 0:
                            portfolio_wealth = INITIAL_WEALTH
                        else:
                            portfolio_wealth = float(
                                current_positions.get(
                                    portfolio_group,
                                    pd.Series(dtype=float)
                                ).sum()
                            )

                            if portfolio_wealth <= 0:
                                portfolio_wealth = INITIAL_WEALTH

                        weights = get_weights(
                            weighting_method=weighting,
                            tickers=selected,
                            mc_row=mc_row,
                            returns_df=returns_df,
                            rebalance_date=reb_date
                        )

                        if weights is None:
                            continue

                        positions = weights * portfolio_wealth

                        annual_rebalance_records.append({
                            "strategy": strategy,
                            "portfolio_group": portfolio_group,
                            "rebalance_date": reb_date,
                            "cap_segment": cap,
                            "portfolio_size": size,
                            "weighting": weighting,
                            "portfolio_wealth_at_rebalance": float(portfolio_wealth),
                            "n_constituents": len(weights)
                        })

                        for ticker in weights.index:
                            constituents_records.append({
                                "strategy": strategy,
                                "portfolio_group": portfolio_group,
                                "rebalance_date": reb_date,
                                "cap_segment": cap,
                                "portfolio_size": size,
                                "weighting": weighting,
                                "ticker": ticker,
                                "weight_at_rebalance": float(weights.loc[ticker]),
                                "capital_allocated_at_rebalance": float(positions.loc[ticker])
                            })

                        evolved_positions, records = evolve_portfolio_one_year(
                            positions=positions,
                            returns_df=returns_df,
                            start_date=reb_date,
                            end_date=next_reb_date
                        )

                        for rec in records:
                            rec.update({
                                "strategy": strategy,
                                "portfolio_group": portfolio_group,
                                "cap_segment": cap,
                                "portfolio_size": size,
                                "weighting": weighting
                            })

                            monthly_records.append(rec)

                        current_positions[portfolio_group] = evolved_positions

    constituents_df = pd.DataFrame(constituents_records)
    monthly_df = pd.DataFrame(monthly_records)
    annual_df = pd.DataFrame(annual_rebalance_records)

    return constituents_df, monthly_df, annual_df


# =========================================================
# 8. SUMMARY
# =========================================================

def create_summary(monthly_df):
    summary = (
        monthly_df
        .groupby([
            "strategy",
            "portfolio_group",
            "cap_segment",
            "portfolio_size",
            "weighting"
        ])
        .agg(
            mean_monthly_return=("monthly_return", "mean"),
            volatility=("monthly_return", "std"),
            skewness=("monthly_return", "skew"),
            n_obs=("monthly_return", "count"),
            final_wealth=("ending_wealth", "last"),
            total_bankruptcies=("bankruptcies_this_month", "sum"),
            total_acquisitions=("acquisitions_this_month", "sum")
        )
        .reset_index()
    )

    return summary

# =========================================================
# 9. T-TESTS
# =========================================================

from scipy.stats import ttest_ind

def create_t_tests(monthly_df):
    records = []

    # Test 1: market cap-segment mot varandra
    cap_pairs = [
        ("small", "mid"),
        ("small", "large"),
        ("mid", "large")
    ]

    for strategy in monthly_df["strategy"].unique():
        for weighting in monthly_df["weighting"].unique():
            for size in monthly_df["portfolio_size"].unique():

                base = monthly_df[
                    (monthly_df["strategy"] == strategy) &
                    (monthly_df["weighting"] == weighting) &
                    (monthly_df["portfolio_size"] == size)
                ]

                for cap_1, cap_2 in cap_pairs:
                    r1 = base.loc[base["cap_segment"] == cap_1, "monthly_return"].dropna()
                    r2 = base.loc[base["cap_segment"] == cap_2, "monthly_return"].dropna()

                    if len(r1) < 2 or len(r2) < 2:
                        continue

                    t_stat, p_value = ttest_ind(
                        r1,
                        r2,
                        equal_var=False,
                        nan_policy="omit"
                    )

                    records.append({
                        "test_type": "cap_segment",
                        "strategy": strategy,
                        "weighting": weighting,
                        "portfolio_size": size,
                        "group_1": cap_1,
                        "group_2": cap_2,
                        "mean_1": r1.mean(),
                        "mean_2": r2.mean(),
                        "difference_mean_1_minus_2": r1.mean() - r2.mean(),
                        "t_stat": t_stat,
                        "p_value": p_value,
                        "n_1": len(r1),
                        "n_2": len(r2)
                    })

    # Test 2: portföljstorlekar mot varandra
    size_pairs = [
        (10, 50),
        (10, 100),
        (50, 100)
    ]

    for strategy in monthly_df["strategy"].unique():
        for weighting in monthly_df["weighting"].unique():
            for cap in monthly_df["cap_segment"].unique():

                base = monthly_df[
                    (monthly_df["strategy"] == strategy) &
                    (monthly_df["weighting"] == weighting) &
                    (monthly_df["cap_segment"] == cap)
                ]

                for size_1, size_2 in size_pairs:
                    r1 = base.loc[base["portfolio_size"] == size_1, "monthly_return"].dropna()
                    r2 = base.loc[base["portfolio_size"] == size_2, "monthly_return"].dropna()

                    if len(r1) < 2 or len(r2) < 2:
                        continue

                    t_stat, p_value = ttest_ind(
                        r1,
                        r2,
                        equal_var=False,
                        nan_policy="omit"
                    )

                    records.append({
                        "test_type": "portfolio_size",
                        "strategy": strategy,
                        "weighting": weighting,
                        "cap_segment": cap,
                        "group_1": size_1,
                        "group_2": size_2,
                        "mean_1": r1.mean(),
                        "mean_2": r2.mean(),
                        "difference_mean_1_minus_2": r1.mean() - r2.mean(),
                        "t_stat": t_stat,
                        "p_value": p_value,
                        "n_1": len(r1),
                        "n_2": len(r2)
                    })

    return pd.DataFrame(records)

# =========================================================
# 10. PLOTS
# =========================================================

def plot_final_wealth_bar_chart(summary_df, output_folder):
    plot_folder = os.path.join(output_folder, "Plots")
    os.makedirs(plot_folder, exist_ok=True)

    for strategy in summary_df["strategy"].unique():
        df = summary_df[summary_df["strategy"] == strategy].copy()

        df["label"] = (
            df["cap_segment"].astype(str) + "_" +
            df["portfolio_size"].astype(str) + "_" +
            df["weighting"].astype(str)
        )

        df = df.sort_values("final_wealth", ascending=False)

        plt.figure(figsize=(14, 7))
        plt.bar(df["label"], df["final_wealth"])
        plt.title(f"Final wealth - {strategy}")
        plt.xlabel("Portfolio")
        plt.ylabel("Final wealth")
        plt.xticks(rotation=90)
        plt.tight_layout()

        filepath = os.path.join(plot_folder, f"final_wealth_{strategy}.png")
        plt.savefig(filepath, dpi=300)
        plt.close()

    print(f"Plots sparade i: {plot_folder}")


# =========================================================
# 10. MAIN
# =========================================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    mc_df, returns_df = load_data(
        file_name=FILE_NAME,
        mc_sheet=MC_SHEET,
        returns_sheet=RETURNS_SHEET,
        start_date=START_DATE,
        end_date=END_DATE
    )

    constituents_df, monthly_df, annual_df = build_rule_based_backtest(
        mc_df=mc_df,
        returns_df=returns_df
    )

    summary_df = create_summary(monthly_df)
    t_tests_df = create_t_tests(monthly_df)
    
    monthly_df = monthly_df.sort_values(
        ["strategy", "cap_segment", "portfolio_size", "weighting", "date"]
    ).reset_index(drop=True)

    annual_df = annual_df.sort_values(
        ["strategy", "cap_segment", "portfolio_size", "weighting", "rebalance_date"]
    ).reset_index(drop=True)

    constituents_df = constituents_df.sort_values(
        ["strategy", "cap_segment", "portfolio_size", "weighting", "rebalance_date", "ticker"]
    ).reset_index(drop=True)

    summary_df = summary_df.sort_values(
        ["strategy", "cap_segment", "portfolio_size", "weighting"]
    ).reset_index(drop=True)

    annual_df.to_csv(os.path.join(OUTPUT_FOLDER, "rule_based_annual_rebalances.csv"), index=False)
    summary_df.to_csv(os.path.join(OUTPUT_FOLDER, "rule_based_summary.csv"), index=False)
    monthly_df.to_csv(os.path.join(OUTPUT_FOLDER, "rule_based_monthly_wealth_path.csv"), index=False)
    constituents_df.to_csv(os.path.join(OUTPUT_FOLDER, "rule_based_constituents.csv"), index=False)
    t_tests_df.to_csv(os.path.join(OUTPUT_FOLDER, "rule_based_t_tests.csv"), index=False)

    plot_final_wealth_bar_chart(summary_df, OUTPUT_FOLDER)

    print("Allt klart. Resultat sparade som CSV-filer.")