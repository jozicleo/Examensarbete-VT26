import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.optimize import minimize
from scipy.stats import t as student_t


# =========================================================
# SETTINGS
# =========================================================

FILE_NAME = "MCReturnsOD.xlsx"
MC_SHEET = "Sheet1"
RETURNS_SHEET = "Returns"
MARKET_RETURNS_SHEET = "Sheet2"

START_DATE = "1982-01-01"
END_DATE = "2026-01-01"

PORTFOLIO_START_DATE = "1996-01-01"

MARKET_START_DATE = "1996-01-01"
MARKET_END_DATE = "2025-12-01"

PORTFOLIO_SIZES = [10, 50, 100]
CAP_SEGMENTS = ["small", "mid", "large"]
WEIGHTING_METHODS = ["equal", "market_cap", "risk_parity"]

SELECTION_STRATEGIES = ["high_skew", "low_skew", "momentum"]

KEEP_RATE = 0.30

SKEW_LOOKBACK = 60
SKEW_MIN_OBS = 60

MOMENTUM_LOOKBACK = 6
MOMENTUM_MIN_OBS = 6

RISK_PARITY_LOOKBACK = 36
RISK_PARITY_MIN_OBS = 24

MOMENTUM_ELIGIBILITY_LOOKBACK = RISK_PARITY_LOOKBACK
MOMENTUM_ELIGIBILITY_MIN_OBS = RISK_PARITY_MIN_OBS

N_SIMULATIONS = 300
RANDOM_SEED = 42
USE_PARALLEL = True

INITIAL_WEALTH = 1.0

OUTPUT_FOLDER = "Combined_MC_RuleBased_Results"

# Ändra dessa 
MARKET_FINAL_WEALTH = 22.31106
RISK_FREE_FINAL_WEALTH = 1.76461

# Histogram-inställningar
HIST_BINS = 50
WEALTH_CAP = 150


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


def load_market_returns(file_name, market_sheet, start_date, end_date):
    market = pd.read_excel(
        file_name,
        sheet_name=market_sheet,
        header=None,
        usecols=[0],
        na_values=["NA", "#VALUE!"]
    )

    dates = pd.date_range(start=start_date, end=end_date, freq="MS")

    if len(market) != len(dates):
        raise ValueError(
            f"Sheet2 har {len(market)} rader, men datumintervallet ger {len(dates)} månader."
        )

    market.columns = ["market_return"]
    market.index = dates
    market["market_return"] = pd.to_numeric(market["market_return"], errors="coerce")

    print(f"Market returns: {market.index.min().date()} till {market.index.max().date()}")

    return market


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
    
    # diag = np.diag(cov_matrix)
    # vol = np.sqrt(np.where(diag <= 0, np.nan, diag))
    # inv_vol = 1.0 / vol
    # inv_vol = np.where(np.isfinite(inv_vol), inv_vol, 0.0)

    # if inv_vol.sum() > 0:
    #     x0 = inv_vol / inv_vol.sum()
    # else:
    #     x0 = np.repeat(1.0 / n, n)

    def objective(w):
        rc = risk_contributions(w, cov_matrix)
        target = np.repeat(np.mean(rc), n)
        return np.sum((rc - target) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(1e-6, 1.0) for _ in range(n)]

    result = minimize(
    objective,
    x0=x0,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={
        "maxiter": 100,
        "ftol": 1e-6,
        "disp": False
    }
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
# 5. RANDOM MONTE CARLO SELECTION
# =========================================================

def filter_universe_by_return_history(universe, returns_df, rebalance_date, lookback_months, min_obs):
    start_date = rebalance_date - pd.DateOffset(months=lookback_months)

    window = returns_df.loc[
        (returns_df.index < rebalance_date) &
        (returns_df.index >= start_date),
        universe
    ]

    valid_counts = window.notna().sum()
    valid_universe = valid_counts[valid_counts >= min_obs].index.tolist()

    return valid_universe


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
# 6. RULE-BASED SELECTION
# =========================================================

def get_lookback_window(returns_df, universe, rebalance_date, lookback_months):
    start_date = rebalance_date - pd.DateOffset(months=lookback_months)

    return returns_df.loc[
        (returns_df.index < rebalance_date) &
        (returns_df.index >= start_date),
        universe
    ].copy()


def select_by_high_skew(universe, returns_df, rebalance_date, size):
    window = get_lookback_window(returns_df, universe, rebalance_date, SKEW_LOOKBACK)

    valid_counts = window.notna().sum()
    valid_tickers = valid_counts[valid_counts >= SKEW_MIN_OBS].index.tolist()

    if len(valid_tickers) < size:
        return None

    skew_values = window[valid_tickers].skew(skipna=True).dropna().sort_values(ascending=False)

    if len(skew_values) < size:
        return None

    return skew_values.index[:size].tolist()


def select_by_low_skew(universe, returns_df, rebalance_date, size):
    window = get_lookback_window(returns_df, universe, rebalance_date, SKEW_LOOKBACK)

    valid_counts = window.notna().sum()
    valid_tickers = valid_counts[valid_counts >= SKEW_MIN_OBS].index.tolist()

    if len(valid_tickers) < size:
        return None

    skew_values = window[valid_tickers].skew(skipna=True).dropna().sort_values(ascending=True)

    if len(skew_values) < size:
        return None

    return skew_values.index[:size].tolist()


# def select_by_momentum(universe, returns_df, rebalance_date, size):
#     window = get_lookback_window(returns_df, universe, rebalance_date, MOMENTUM_LOOKBACK)

#     valid_counts = window.notna().sum()
#     valid_tickers = valid_counts[valid_counts >= MOMENTUM_MIN_OBS].index.tolist()

#     if len(valid_tickers) < size:
#         return None

#     window = window[valid_tickers]

#     momentum = (1.0 + window).prod(axis=0) - 1.0
#     momentum = momentum.dropna().sort_values(ascending=False)

#     if len(momentum) < size:
#         return None

#     return momentum.index[:size].tolist()

def select_by_momentum(universe, returns_df, rebalance_date, size):
    # Steg 1: eligibility-filter
    # Momentum får bara välja bland aktier som har tillräcklig historik
    # för att risk parity senare ska kunna beräkna kovariansmatrisen.
    eligibility_window = get_lookback_window(
        returns_df=returns_df,
        universe=universe,
        rebalance_date=rebalance_date,
        lookback_months=MOMENTUM_ELIGIBILITY_LOOKBACK
    )

    eligibility_counts = eligibility_window.notna().sum()

    eligible_tickers = eligibility_counts[
        eligibility_counts >= MOMENTUM_ELIGIBILITY_MIN_OBS
    ].index.tolist()

    if len(eligible_tickers) < size:
        return None

    # Steg 2: momentum-signal
    # Rangordna bara de eligible aktierna efter senaste 6 månaders avkastning.
    momentum_window = get_lookback_window(
        returns_df=returns_df,
        universe=eligible_tickers,
        rebalance_date=rebalance_date,
        lookback_months=MOMENTUM_LOOKBACK
    )

    momentum_counts = momentum_window.notna().sum()

    valid_momentum_tickers = momentum_counts[
        momentum_counts >= MOMENTUM_MIN_OBS
    ].index.tolist()

    if len(valid_momentum_tickers) < size:
        return None

    momentum = (1.0 + momentum_window[valid_momentum_tickers]).prod(axis=0) - 1.0
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
# 7. MONTHLY WEALTH EVOLUTION
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

def evolve_portfolio_one_year_final_only(positions, returns_df, start_date, end_date):
    positions = positions.copy().astype(float)
    tickers = positions.index.tolist()

    raw = returns_df.loc[
        (returns_df.index > start_date) &
        (returns_df.index <= end_date),
        tickers
    ].values.astype(float)  # shape: (n_months, n_tickers)

    if raw.shape[0] == 0:
        return positions

    nan_mask = np.isnan(raw)
    has_nan = nan_mask.any(axis=0)

    # For each ticker that has a NaN (acquisition), freeze it from the first NaN onward.
    # argmax returns 0 for columns with no True, so guard with has_nan.
    first_nan_row = np.where(has_nan, np.argmax(nan_mask, axis=0), raw.shape[0])
    frozen = np.arange(raw.shape[0])[:, np.newaxis] >= first_nan_row[np.newaxis, :]

    # Non-frozen cells have no NaN by construction; frozen cells use 1.0.
    # Bankruptcy (r == -1) gives a growth factor of 0 and propagates naturally.
    growth = np.where(frozen, 1.0, 1.0 + raw)

    return pd.Series(positions.values * growth.prod(axis=0), index=positions.index)
# =========================================================
# 8. BUILD ONE MONTE CARLO SIMULATION
# =========================================================

def build_one_simulation(mc_df, returns_df, simulation_id, rng):
    rebalance_dates = get_rebalance_dates(mc_df.index, PORTFOLIO_START_DATE)

    final_wealth_records = []

    current_constituents_base = {}
    current_positions = {}

    for t in range(len(rebalance_dates) - 1):
        reb_date = rebalance_dates[t]
        next_reb_date = rebalance_dates[t + 1]

        mc_row = mc_df.loc[reb_date]
        segments = classify_market_caps(mc_row)

        for cap in CAP_SEGMENTS:
            universe = segments[cap]

            for size in PORTFOLIO_SIZES:
                for weighting in WEIGHTING_METHODS:

                    if weighting == "risk_parity":
                        selection_universe = filter_universe_by_return_history(
                            universe=universe,
                            returns_df=returns_df,
                            rebalance_date=reb_date,
                            lookback_months=RISK_PARITY_LOOKBACK,
                            min_obs=RISK_PARITY_MIN_OBS
                        )
                    else:
                        selection_universe = universe

                    base_key = (cap, size, weighting)

                    if t == 0:
                        selected = initial_random_selection(selection_universe, size, rng)
                    else:
                        previous_selected = current_constituents_base.get(base_key, None)

                        if previous_selected is None:
                            selected = initial_random_selection(selection_universe, size, rng)
                        else:
                            selected = rebalance_selection(
                                old_tickers=previous_selected,
                                universe=selection_universe,
                                size=size,
                                rng=rng,
                                keep_rate=KEEP_RATE
                            )

                    if selected is None:
                        continue

                    current_constituents_base[base_key] = selected.copy()

                    portfolio_group = f"sim{simulation_id}_{cap}_{size}_{weighting}"

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

                    positions = weights * portfolio_wealth

                    evolved_positions = evolve_portfolio_one_year_final_only(
                        positions=positions,
                        returns_df=returns_df,
                        start_date=reb_date,
                        end_date=next_reb_date
                    )

                    current_positions[portfolio_group] = evolved_positions

    for portfolio_group, positions in current_positions.items():
        parts = portfolio_group.split("_")

        final_wealth_records.append({
            "simulation": simulation_id,
            "portfolio_group": portfolio_group,
            "cap_segment": parts[1],
            "portfolio_size": int(parts[2]),
            "weighting": "_".join(parts[3:]),
            "final_wealth": float(positions.sum())
        })

    final_wealth_df = pd.DataFrame(final_wealth_records)

    return final_wealth_df

# =========================================================
# 9. RUN MONTE CARLO
# =========================================================

_worker_mc_df = None
_worker_returns_df = None


def _init_worker(mc_df, returns_df):
    global _worker_mc_df, _worker_returns_df
    _worker_mc_df = mc_df
    _worker_returns_df = returns_df


def _run_one_sim(args):
    sim_id, seed_int = args
    sim_rng = np.random.default_rng(seed_int)
    return build_one_simulation(_worker_mc_df, _worker_returns_df, sim_id, sim_rng)


def run_monte_carlo(mc_df, returns_df, n_simulations=10000, seed=42):
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 1_000_000_000, size=n_simulations).tolist()
    tasks = list(enumerate(seeds, start=1))

    if not USE_PARALLEL or n_simulations <= 4:
        results = [_run_one_sim(t) for t in tasks]
    else:
        results = [None] * n_simulations
        with ProcessPoolExecutor(
            max_workers=os.cpu_count(),
            initializer=_init_worker,
            initargs=(mc_df, returns_df)
        ) as ex:
            futures = {ex.submit(_run_one_sim, t): t[0] for t in tasks}
            done = 0
            batch_start = time.perf_counter()
            for fut in as_completed(futures):
                sim_id = futures[fut]
                results[sim_id - 1] = fut.result()
                done += 1
                if done % 100 == 0 or done == n_simulations:
                    elapsed = time.perf_counter() - batch_start
                    print(f"Monte Carlo: {done}/{n_simulations} klar  [{elapsed:.1f}s för senaste {done % 100 or 100}]")
                    batch_start = time.perf_counter()

    return pd.concat(results, ignore_index=True)


# =========================================================
# 10. BUILD RULE-BASED BACKTEST
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

        print(f"Regelbaserad rebalansering: {reb_date.date()}")

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
# 11. SUMMARIES
# =========================================================

# def create_mc_summary(monthly_df):
#     summary = (
#         monthly_df
#         .groupby(["simulation", "portfolio_group", "cap_segment", "portfolio_size", "weighting"])
#         .agg(
#             mean_monthly_return=("monthly_return", "mean"),
#             volatility=("monthly_return", "std"),
#             skewness=("monthly_return", "skew"),
#             n_obs=("monthly_return", "count"),
#             final_wealth=("ending_wealth", "last")
#         )
#         .reset_index()
#     )

#     return summary


def create_rule_summary(monthly_df):
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
# 12. MARKET T-TESTS
# =========================================================

def paired_t_test_return_difference(diff_series):
    d = pd.Series(diff_series).dropna().astype(float)
    T = len(d)

    if T < 2:
        return np.nan, np.nan, np.nan, np.nan, T

    mean_diff = d.mean()
    historical_variance = d.var(ddof=1)
    historical_std = np.sqrt(historical_variance)
    standard_error = historical_std / np.sqrt(T)

    if standard_error <= 0 or np.isnan(standard_error):
        return mean_diff, historical_variance, np.nan, np.nan, T

    t_stat = mean_diff / standard_error
    p_value = 2 * (1 - student_t.cdf(abs(t_stat), df=T - 1))

    return mean_diff, historical_variance, t_stat, p_value, T


def create_market_t_tests(rule_monthly_df, market_returns_df):
    df = rule_monthly_df.copy()
    market = market_returns_df.copy()

    records = []

    grouping_cols = [
        "strategy",
        "portfolio_group",
        "cap_segment",
        "portfolio_size",
        "weighting"
    ]

    for keys, group in df.groupby(grouping_cols):
        strategy, portfolio_group, cap_segment, portfolio_size, weighting = keys

        portfolio_returns = group[["date", "monthly_return"]].copy()

        merged = portfolio_returns.merge(
            market,
            left_on="date",
            right_index=True,
            how="inner"
        ).dropna(subset=["monthly_return", "market_return"])

        diff = merged["monthly_return"] - merged["market_return"]

        mean_diff, hist_var, t_stat, p_value, T = paired_t_test_return_difference(diff)

        records.append({
            "strategy": strategy,
            "portfolio_group": portfolio_group,
            "cap_segment": cap_segment,
            "portfolio_size": portfolio_size,
            "weighting": weighting,
            "mean_portfolio_return": merged["monthly_return"].mean(),
            "mean_market_return": merged["market_return"].mean(),
            "mean_difference_vs_market": mean_diff,
            "historical_variance_of_difference": hist_var,
            "t_stat": t_stat,
            "p_value": p_value,
            "n_obs": T
        })

    results = pd.DataFrame(records)

    if len(results) > 0:
        results["significant_5pct"] = results["p_value"] < 0.05
        results["significant_10pct"] = results["p_value"] < 0.10

    return results


# =========================================================
# 13. PLOTTING WITH OVERLAYS AND PERCENTILES
# =========================================================

def percentile_rank(value, distribution):
    distribution = pd.Series(distribution).dropna().astype(float)

    if len(distribution) == 0 or pd.isna(value):
        return np.nan

    return 100.0 * (distribution <= value).mean()


def plot_mc_histograms_with_rule_overlays(
    mc_summary_df,
    rule_summary_df,
    output_folder,
    bins=50,
    wealth_cap=150,
    market_final_wealth=None,
    risk_free_final_wealth=None
):
    plot_folder = os.path.join(output_folder, "Plots_MC_With_Overlays")
    os.makedirs(plot_folder, exist_ok=True)

    percentile_records = []

    color_map = {
        "high_skew": "red",
        "low_skew": "green",
        "momentum": "orange",
        "market_index": "black",
        "risk_free": "purple",
        "MC mean": "blue",
        "MC median": "cyan"
    }

    grouped = mc_summary_df.groupby(["cap_segment", "portfolio_size", "weighting"])

    for (cap, size, weighting), df in grouped:
        mc_wealth_raw = df["final_wealth"].dropna().astype(float)

        if len(mc_wealth_raw) == 0:
            continue

        mc_wealth_plot = mc_wealth_raw.clip(upper=wealth_cap)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(
            mc_wealth_plot,
            bins=bins,
            edgecolor="black",
            color="lightgray"
        )

        ax.set_title(
            f"Final Wealth Distribution\n"
            f"{cap.capitalize()} cap | {size} stocks | {weighting}"
        )
        ax.set_xlabel(f"Final wealth, capped at {wealth_cap}")
        ax.set_ylabel("Number of simulations")
        ax.set_xlim(left=0, right=wealth_cap)

        overflow_count = int((mc_wealth_raw > wealth_cap).sum())

        rule_subset = rule_summary_df[
            (rule_summary_df["cap_segment"] == cap) &
            (rule_summary_df["portfolio_size"] == size) &
            (rule_summary_df["weighting"] == weighting)
        ].copy()

        overlay_values = []

        for _, row in rule_subset.iterrows():
            strategy = row["strategy"]
            value = float(row["final_wealth"])
            pct = percentile_rank(value, mc_wealth_raw)
            plot_value = min(value, wealth_cap)

            overlay_values.append((strategy, value, plot_value, pct))

            percentile_records.append({
                "cap_segment": cap,
                "portfolio_size": size,
                "weighting": weighting,
                "marker": strategy,
                "final_wealth": value,
                "percentile_vs_mc": pct,
                "is_above_wealth_cap": value > wealth_cap
            })

        if market_final_wealth is not None:
            value = float(market_final_wealth)
            pct = percentile_rank(value, mc_wealth_raw)
            plot_value = min(value, wealth_cap)

            overlay_values.append(("market_index", value, plot_value, pct))

            percentile_records.append({
                "cap_segment": cap,
                "portfolio_size": size,
                "weighting": weighting,
                "marker": "market_index",
                "final_wealth": value,
                "percentile_vs_mc": pct,
                "is_above_wealth_cap": value > wealth_cap
            })

        if risk_free_final_wealth is not None:
            value = float(risk_free_final_wealth)
            pct = percentile_rank(value, mc_wealth_raw)
            plot_value = min(value, wealth_cap)

            overlay_values.append(("risk_free", value, plot_value, pct))

            percentile_records.append({
                "cap_segment": cap,
                "portfolio_size": size,
                "weighting": weighting,
                "marker": "risk_free",
                "final_wealth": value,
                "percentile_vs_mc": pct,
                "is_above_wealth_cap": value > wealth_cap
            })

        for label, original_value, plot_value, pct in overlay_values:
            ax.axvline(
                plot_value,
                linestyle="--",
                linewidth=2.5,
                color=color_map.get(label, "blue"),
                label=f"{label}: {original_value:.2f}, pct {pct:.1f}"
            )

        ax.axvline(
            mc_wealth_raw.mean(),
            linestyle=":",
            linewidth=2.5,
            color=color_map["MC mean"],
            label=f"MC mean: {mc_wealth_raw.mean():.2f}"
        )

        ax.axvline(
            mc_wealth_raw.median(),
            linestyle="-.",
            linewidth=2.5,
            color=color_map["MC median"],
            label=f"MC median: {mc_wealth_raw.median():.2f}"
        )

        ax.text(
            0.99,
            0.95,
            f"Overflow > {wealth_cap}: {overflow_count}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9
        )

        ax.legend(fontsize=8)
        fig.tight_layout()

        filename = f"hist_final_wealth_{cap}_{size}_{weighting}_with_overlays.png"
        filepath = os.path.join(plot_folder, filename)

        fig.savefig(filepath, dpi=300)
        plt.close(fig)

    percentile_df = pd.DataFrame(percentile_records)

    percentile_df.to_csv(
        os.path.join(output_folder, "overlay_percentiles_vs_mc.csv"),
        index=False
    )

    print(f"Histogram sparade i: {plot_folder}")
    print("Percentiler sparade i overlay_percentiles_vs_mc.csv")

    return percentile_df

# =========================================================
# 14. MAIN
# =========================================================

if __name__ == "__main__":
    _program_start = time.perf_counter()
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    mc_df, returns_df = load_data(
        file_name=FILE_NAME,
        mc_sheet=MC_SHEET,
        returns_sheet=RETURNS_SHEET,
        start_date=START_DATE,
        end_date=END_DATE
    )

    market_returns_df = load_market_returns(
        file_name=FILE_NAME,
        market_sheet=MARKET_RETURNS_SHEET,
        start_date=MARKET_START_DATE,
        end_date=MARKET_END_DATE
    )

    print("\nStartar Monte Carlo...\n")

    # mc_monthly_df, mc_annual_df = run_monte_carlo(
    #     mc_df=mc_df,
    #     returns_df=returns_df,
    #     n_simulations=N_SIMULATIONS,
    #     seed=RANDOM_SEED
    # )

    mc_summary_df = run_monte_carlo(
        mc_df=mc_df,
        returns_df=returns_df,
        n_simulations=N_SIMULATIONS,
        seed=RANDOM_SEED
    )

    # mc_summary_df = create_mc_summary(mc_monthly_df)

    print("\nStartar regelbaserade strategier...\n")

    rule_constituents_df, rule_monthly_df, rule_annual_df = build_rule_based_backtest(
        mc_df=mc_df,
        returns_df=returns_df
    )

    rule_summary_df = create_rule_summary(rule_monthly_df)

    print("\nSkapar t-tester...\n")

    t_tests_df = create_market_t_tests(
        rule_monthly_df=rule_monthly_df,
        market_returns_df=market_returns_df
    )

    print("\nSkapar histogram med overlays...\n")

    percentile_df = plot_mc_histograms_with_rule_overlays(
        mc_summary_df=mc_summary_df,
        rule_summary_df=rule_summary_df,
        output_folder=OUTPUT_FOLDER,
        bins=HIST_BINS,
        wealth_cap=WEALTH_CAP,
        market_final_wealth=MARKET_FINAL_WEALTH,
        risk_free_final_wealth=RISK_FREE_FINAL_WEALTH
    )

    print("\nSparar filer...\n")

    # mc_monthly_df.to_csv(
    #     os.path.join(OUTPUT_FOLDER, "mc_monthly_wealth_path.csv"),
    #     index=False
    # )

    # mc_annual_df.to_csv(
    #     os.path.join(OUTPUT_FOLDER, "mc_annual_rebalances.csv"),
    #     index=False
    # )

    mc_summary_df.to_csv(
        os.path.join(OUTPUT_FOLDER, "mc_summary.csv"),
        index=False
    )

    # rule_monthly_df.to_csv(
    #     os.path.join(OUTPUT_FOLDER, "rule_based_monthly_wealth_path.csv"),
    #     index=False
    # )

    # rule_annual_df.to_csv(
    #     os.path.join(OUTPUT_FOLDER, "rule_based_annual_rebalances.csv"),
    #     index=False
    # )

    rule_constituents_df.to_csv(
        os.path.join(OUTPUT_FOLDER, "rule_based_constituents.csv"),
        index=False
    )

    rule_summary_df.to_csv(
        os.path.join(OUTPUT_FOLDER, "rule_based_summary.csv"),
        index=False
    )

    # market_returns_df.to_csv(
    #     os.path.join(OUTPUT_FOLDER, "market_returns.csv"),
    #     index=True
    # )

    t_tests_df.to_csv(
        os.path.join(OUTPUT_FOLDER, "rule_based_t_tests_vs_market.csv"),
        index=False
    )

    percentile_df.to_csv(
        os.path.join(OUTPUT_FOLDER, "overlay_percentiles_vs_mc.csv"),
        index=False
    )

    _total_elapsed = time.perf_counter() - _program_start
    print("\nAllt klart.")
    print(f"Resultat sparade i mappen: {OUTPUT_FOLDER}")
    print(f"Total körtid: {_total_elapsed/60:.1f} minuter")