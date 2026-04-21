import numpy as np
import pandas as pd
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

KEEP_RATE = 0.30
RISK_PARITY_LOOKBACK = 36
RISK_PARITY_MIN_OBS = 24

N_SIMULATIONS = 1   # börja lågt, höj senare
RANDOM_SEED = 42

INITIAL_WEALTH = 1.0
OUTPUT_FILE = "MC_portfolio_results_wealth_based.xlsx"


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

    return {"small": small, "mid": mid, "large": large}


# =========================================================
# 4. WEIGHTING METHODS
# =========================================================

def equal_weights(tickers):
    n = len(tickers)
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
        constraints=constraints
    )

    if not result.success:
        return None

    w = result.x / result.x.sum()
    return pd.Series(w, index=tickers)


def estimate_covariance(returns_df, tickers, rebalance_date, lookback_months, min_obs):
    start_date = rebalance_date - pd.DateOffset(months=lookback_months)

    window = returns_df.loc[
        (returns_df.index < rebalance_date) & (returns_df.index >= start_date),
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
# 5. SELECTION
# =========================================================

def initial_random_selection(universe, size, rng):
    if len(universe) < size:
        return None
    return rng.choice(universe, size=size, replace=False).tolist()


def rebalance_selection(old_tickers, universe, size, rng, keep_rate=0.30):
    old_valid = [x for x in old_tickers if x in universe]

    keep_n = int(round(size * keep_rate))
    keep_n = min(keep_n, len(old_valid))

    kept = rng.choice(old_valid, size=keep_n, replace=False).tolist() if keep_n > 0 else []

    remaining_universe = [x for x in universe if x not in kept]
    new_n = size - len(kept)

    if len(remaining_universe) < new_n:
        return None

    new_tickers = rng.choice(remaining_universe, size=new_n, replace=False).tolist()
    return kept + new_tickers


# =========================================================
# 6. MONTHLY WEALTH EVOLUTION
# =========================================================

def evolve_portfolio_one_year(positions, returns_df, start_date, end_date):
    """
    positions: pd.Series med kronor investerade i varje ticker vid årets början

    Logik:
    - giltig return används normalt
    - om return = -1 -> konkurs -> position går till 0
    - om return saknas efter en tidigare giltig return som inte var -1
      -> uppköp -> positionens värde hålls konstant fram till nästa rebalansering

    Returnerar:
    - slutpositioner efter årets månadsutveckling
    - records med månatlig portföljinfo
    """
    positions = positions.copy().astype(float)
    tickers = positions.index.tolist()

    period_returns = returns_df.loc[
        (returns_df.index > start_date) & (returns_df.index <= end_date),
        tickers
    ].copy()

    records = []

    # håller koll på om bolaget fortfarande är aktivt
    # True = uppdateras med returns
    # False = fryst efter uppköp eller redan dött efter konkurs
    active_status = {ticker: True for ticker in tickers}

    # för tydligare statistik
    bankrupt_status = {ticker: False for ticker in tickers}
    acquired_status = {ticker: False for ticker in tickers}

    for dt, row in period_returns.iterrows():
        beginning_positions = positions.copy()
        beginning_wealth = beginning_positions.sum()

        bankrupt_this_month = 0
        acquired_this_month = 0

        for ticker in tickers:
            r = row[ticker]

            # om position redan är fryst eller död:
            if not active_status[ticker]:
                continue

            # missing return efter tidigare giltigt värde:
            if pd.isna(r):
                # om positionen fortfarande har värde när return blir missing,
                # tolka som uppköp och frys värdet till nästa rebalansering
                if positions[ticker] > 0:
                    active_status[ticker] = False
                    acquired_status[ticker] = True
                    acquired_this_month += 1
                continue

            # normal giltig return
            positions[ticker] = positions[ticker] * (1.0 + r)

            # konkurs
            if r == -1:
                positions[ticker] = 0.0
                active_status[ticker] = False
                bankrupt_status[ticker] = True
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
# 7. BUILD ONE SIMULATION
# =========================================================

def build_one_simulation(mc_df, returns_df, simulation_id, rng):
    rebalance_dates = get_rebalance_dates(mc_df.index, PORTFOLIO_START_DATE)

    constituents_records = []
    monthly_records = []
    annual_rebalance_records = []

    # Gemensamma urval per (cap, storlek)
    current_constituents_base = {}

    # Varje faktisk portfölj får eget wealth path
    current_positions = {}

    for t in range(len(rebalance_dates) - 1):
        reb_date = rebalance_dates[t]
        next_reb_date = rebalance_dates[t + 1]

        mc_row = mc_df.loc[reb_date]
        segments = classify_market_caps(mc_row)

        for cap in CAP_SEGMENTS:
            universe = segments[cap]

            for size in PORTFOLIO_SIZES:
                base_key = (cap, size)

                if t == 0:
                    selected = initial_random_selection(universe, size, rng)
                else:
                    previous_selected = current_constituents_base.get(base_key, None)
                    if previous_selected is None:
                        selected = initial_random_selection(universe, size, rng)
                    else:
                        selected = rebalance_selection(
                            old_tickers=previous_selected,
                            universe=universe,
                            size=size,
                            rng=rng,
                            keep_rate=KEEP_RATE
                        )

                if selected is None:
                    continue

                current_constituents_base[base_key] = selected.copy()

                for weighting in WEIGHTING_METHODS:
                    portfolio_group = f"sim{simulation_id}_{cap}_{size}_{weighting}"

                    # wealth in i årets rebalansering
                    if t == 0:
                        portfolio_wealth = INITIAL_WEALTH
                    else:
                        if portfolio_group not in current_positions:
                            continue
                        portfolio_wealth = float(current_positions[portfolio_group].sum())

                    weights = get_weights(
                        weighting_method=weighting,
                        tickers=selected,
                        mc_row=mc_row,
                        returns_df=returns_df,
                        rebalance_date=reb_date
                    )

                    if weights is None:
                        continue

                    # Fördela HELA aktuella wealth på årets nya innehav
                    positions = weights * portfolio_wealth

                    annual_rebalance_records.append({
                        "simulation": simulation_id,
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
                            "simulation": simulation_id,
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
                            "simulation": simulation_id,
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
# 8. RUN MONTE CARLO
# =========================================================

def run_monte_carlo(mc_df, returns_df, n_simulations=3, seed=42):
    rng = np.random.default_rng(seed)

    all_constituents = []
    all_monthly = []
    all_annual = []

    for sim in range(1, n_simulations + 1):
        sim_rng = np.random.default_rng(rng.integers(0, 1_000_000_000))

        constituents_df, monthly_df, annual_df = build_one_simulation(
            mc_df=mc_df,
            returns_df=returns_df,
            simulation_id=sim,
            rng=sim_rng
        )

        all_constituents.append(constituents_df)
        all_monthly.append(monthly_df)
        all_annual.append(annual_df)

        print(f"Klar med simulering {sim}/{n_simulations}")

    all_constituents = pd.concat(all_constituents, ignore_index=True)
    all_monthly = pd.concat(all_monthly, ignore_index=True)
    all_annual = pd.concat(all_annual, ignore_index=True)

    return all_constituents, all_monthly, all_annual


# =========================================================
# 9. SUMMARY
# =========================================================

def create_summary(monthly_df):
    summary = (
        monthly_df
        .groupby(["portfolio_group", "cap_segment", "portfolio_size", "weighting"])
        .agg(
            mean_monthly_return=("monthly_return", "mean"),
            volatility=("monthly_return", "std"),
            skewness=("monthly_return", "skew"),
            n_obs=("monthly_return", "count"),
            final_wealth=("ending_wealth", "last")
        )
        .reset_index()
    )

    return summary


# =========================================================
# 10. MAIN
# =========================================================

if __name__ == "__main__":
    mc_df, returns_df = load_data(
        file_name=FILE_NAME,
        mc_sheet=MC_SHEET,
        returns_sheet=RETURNS_SHEET,
        start_date=START_DATE,
        end_date=END_DATE
    )

    constituents_df, monthly_df, annual_df = run_monte_carlo(
        mc_df=mc_df,
        returns_df=returns_df,
        n_simulations=N_SIMULATIONS,
        seed=RANDOM_SEED
    )

    summary_df = create_summary(monthly_df)
    
    monthly_df = monthly_df.sort_values(
    ["simulation", "cap_segment", "portfolio_size", "weighting", "date"]
    ).reset_index(drop=True)

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        constituents_df.to_excel(writer, sheet_name="Constituents", index=False)
        monthly_df.to_excel(writer, sheet_name="MonthlyWealthPath", index=False)
        annual_df.to_excel(writer, sheet_name="AnnualRebalances", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    print(f"Allt klart. Resultat sparade i {OUTPUT_FILE}")