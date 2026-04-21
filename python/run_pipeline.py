"""
Pipeline orchestrator — runs Model A + Model B end-to-end and writes
data/fragility_output.json which the dashboard reads.

Usage:
  cd python
  python run_pipeline.py

Requirements:
  pip install -r requirements.txt
  export FRED_API_KEY=your_key   (get free key at fred.stlouisfed.org)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent / 'model_b'))
sys.path.insert(0, str(Path(__file__).parent / 'model_a'))

# ── imports ───────────────────────────────────────────────────────────────────
from fetch_market_data import fetch_market_data, derive_extended_features, handle_missing_data
from fetch_macro_data  import fetch_macro_data, resample_to_daily, handle_missing_data as handle_macro
from preprocessing_b   import (merge_market_and_macro, handle_missing_values,
                                add_cbrt_governor_dummy, resample_to_monthly)
from feature_engineering_b import (compute_rolling_correlation, compute_permutation_entropy,
                                    compute_rolling_volatility, compute_fragility_score_b,
                                    add_lag_features)
def label_regimes(scores):
    """Fixed-threshold Minsky regime labels (matches schema spec)."""
    return scores.apply(lambda s: 'PONZI' if s >= 70 else ('SPECULATIVE' if s >= 40 else 'HEDGE'))
from models_b          import (train_random_forest_walk_forward, train_ols_walk_forward,
                                train_var_walk_forward, compute_shap_values_b)
import preprocessing   as pre_a
import feature_engineering as feat_a
import models          as mod_a
import export_unified  as exporter

warnings.filterwarnings('ignore')


# ── constants ─────────────────────────────────────────────────────────────────
START_DATE = '2003-01-01'
END_DATE   = '2026-04-30'
FRED_KEY   = os.environ.get('FRED_API_KEY')


# ── Model B pipeline ──────────────────────────────────────────────────────────

def run_model_b():
    print("\n" + "=" * 60)
    print("MODEL B PIPELINE  (2003–2026 Extended)")
    print("=" * 60)

    # 1. Fetch & clean market data
    print("\n[1] Fetching market data...")
    mkt = fetch_market_data(start_date=START_DATE, end_date=END_DATE)
    mkt = derive_extended_features(mkt)
    mkt = handle_missing_data(mkt, max_gap=5)

    # 2. Fetch & clean macro data
    print("\n[2] Fetching macro data (FRED)...")
    if not FRED_KEY:
        print("  WARNING: FRED_API_KEY not set — macro features will be missing")
        macro = pd.DataFrame(index=mkt.index)
    else:
        macro_raw = fetch_macro_data(api_key=FRED_KEY, start_date=START_DATE, end_date=END_DATE)
        macro = resample_to_daily(macro_raw)
        macro = handle_macro(macro)

    # 3. Merge & add CBRT dummy
    print("\n[3] Merging & adding CBRT governor dummy...")
    merged = merge_market_and_macro(mkt, macro) if not macro.empty else mkt.copy()
    merged = add_cbrt_governor_dummy(merged)
    daily  = handle_missing_values(merged, max_gap=5)

    # 4. Feature engineering (daily)
    print("\n[4] Computing features...")
    corr_df = compute_rolling_correlation(daily)

    daily['mean_corr']      = corr_df['mean_corr']
    daily['eigenvalue_ratio'] = corr_df.get('eigenvalue_ratio', np.nan)
    # Add pairwise correlations vs ISE_USD
    for col in [c for c in corr_df.columns if '_corr' in c or '_' in c]:
        daily[col] = corr_df[col]

    ise_ret = np.log(daily['ISE_USD'] / daily['ISE_USD'].shift(1)) if 'ISE_USD' in daily.columns else daily.get('BIST100')
    daily['rolling_volatility']   = compute_rolling_volatility(ise_ret if ise_ret is not None else daily.iloc[:, 0])
    daily['permutation_entropy']  = compute_permutation_entropy(ise_ret if ise_ret is not None else daily.iloc[:, 0])

    # Fragility score (fixed 6-component, TRY_weakness as dominant Turkey signal)
    daily['fragility_score'] = compute_fragility_score_b(
        corr             = daily['mean_corr'],
        pe               = daily['permutation_entropy'],
        vol              = daily['rolling_volatility'],
        eigenvalue_ratio = daily['eigenvalue_ratio'],
        vix              = daily.get('VIX'),
        dxy              = daily.get('DXY'),
        try_usd          = daily.get('TRY_USD'),
    )
    daily['regime'] = label_regimes(daily['fragility_score'])

    # 5. Monthly resampling + lag features
    print("\n[5] Monthly resampling + lag features...")
    monthly = resample_to_monthly(daily)
    monthly = add_lag_features(monthly)

    # 6. Model training (RF + OLS + VAR)
    # Use log returns as target (stationary — avoids price-level drift across splits)
    print("\n[6] Training models (RF + OLS + VAR)...")
    if 'ISE_USD' in daily.columns:
        daily['ISE_USD_ret'] = np.log(daily['ISE_USD'] / daily['ISE_USD'].shift(1))
    rf_results  = train_random_forest_walk_forward(daily, target_col='ISE_USD_ret')
    ols_results = train_ols_walk_forward(daily,          target_col='ISE_USD_ret')
    var_results = train_var_walk_forward(daily,          target_col='ISE_USD_ret')

    # 7. SHAP values
    print("\n[7] Computing SHAP values...")
    shap_results = compute_shap_values_b(daily)

    # Aggregate performance — use last split as overall representative metrics
    # (or recompute over all available data)
    def _agg_split_metrics(split_results: dict) -> dict:
        """Average metrics across all splits, weighted by test size."""
        if not split_results:
            return {}
        vals = list(split_results.values())
        if isinstance(vals[0], dict) and 'r2' in vals[0]:
            return {
                'r2':   round(np.nanmean([v.get('r2', np.nan)  or np.nan for v in vals]), 4),
                'rmse': round(np.nanmean([v.get('rmse', np.nan) or np.nan for v in vals]), 4),
                'mae':  round(np.nanmean([v.get('mae', np.nan)  or np.nan for v in vals]), 4),
                'hedge_rmse': round(np.nanmean([v.get('HEDGE', {}).get('rmse', np.nan) or np.nan for v in vals]), 4),
                'spec_rmse':  round(np.nanmean([v.get('SPECULATIVE', {}).get('rmse', np.nan) or np.nan for v in vals]), 4),
                'ponzi_rmse': round(np.nanmean([v.get('PONZI', {}).get('rmse', np.nan) or np.nan for v in vals]), 4),
            }
        # split_results is {split_name: {r2: ..., rmse: ...}}
        return {
            'r2':   round(np.nanmean([v.get('r2', np.nan)  or np.nan for v in vals]), 4),
            'rmse': round(np.nanmean([v.get('rmse', np.nan) or np.nan for v in vals]), 4),
            'mae':  round(np.nanmean([v.get('mae', np.nan)  or np.nan for v in vals]), 4),
        }

    # Extract RF metrics — results[split]['metrics']['test_r2'] + results[split]['regime_rmse']
    def _rf_perf(split_dict):
        m = split_dict.get('metrics', {})
        rr = split_dict.get('regime_rmse', {})
        return {
            'r2':         round(m.get('test_r2')   or 0, 4),
            'rmse':       round(m.get('test_rmse')  or 0, 4),
            'mae':        round(m.get('test_mae')   or 0, 4),
            'hedge_rmse': round(rr.get('HEDGE')     or 0, 4),
            'spec_rmse':  round(rr.get('SPECULATIVE') or 0, 4),
            'ponzi_rmse': round(rr.get('PONZI')     or 0, 4),
        }

    # Use the 2021 Turkey split (longest, most academically important) as headline metrics
    rf_last  = rf_results.get('split_2021_turkey') or (list(rf_results.values())[-1]  if rf_results  else {})
    ols_last = ols_results.get('split_2021_turkey') or (list(ols_results.values())[-1] if ols_results else {})
    var_last = var_results.get('split_2021_turkey') or (list(var_results.values())[-1] if var_results else {})

    performance = {
        'random_forest': _rf_perf(rf_last),
        'ols':           ols_last,   # already has {r2, rmse, mae, hedge_rmse, ...}
        'var':           var_last,
    }

    # SHAP — aggregate mean_abs_shap across all crisis periods
    shap_dict = {}
    crisis_shaps = (shap_results or {}).get('crisis_periods', {})
    if crisis_shaps:
        # Average across all crisis periods for a robust overall picture
        all_feats = set()
        for p in crisis_shaps.values():
            all_feats.update(p.get('mean_abs_shap', {}).keys())
        for feat in all_feats:
            vals = [p['mean_abs_shap'][feat] for p in crisis_shaps.values()
                    if feat in p.get('mean_abs_shap', {})]
            shap_dict[feat] = round(float(np.mean(vals)), 6) if vals else 0.0

    walk_fwd = [
        {**{'split': k, **v}}
        for k, v in {**rf_results}.items()
    ]

    return monthly, performance, shap_dict, walk_fwd


# ── Model A pipeline ──────────────────────────────────────────────────────────

def _run_model_a_variant(train_df, ext_df, target_col, label):
    """Train and score one Model A variant (USD or TL target)."""
    print(f"\n  [{label}] Computing training features...")
    train_feat = feat_a.compute_features(train_df, target_col=target_col)

    print(f"  [{label}] Training RF...")
    rf, shap_vals, feat_names = mod_a.train_model_a(train_feat, train_df, target_col=target_col)

    rf_err_train = mod_a.score_oos(rf, train_feat, train_df, target_col=target_col)
    train_feat   = feat_a.compute_features(train_df, rf_error_series=rf_err_train, target_col=target_col)

    rf_err_ext = mod_a.score_oos(rf, feat_a.compute_features(ext_df, target_col=target_col), ext_df, target_col=target_col)
    ext_feat   = feat_a.compute_features(ext_df, rf_error_series=rf_err_ext, target_col=target_col)

    print(f"  [{label}] Walk-forward validation...")
    wf_results  = mod_a.run_walk_forward(rf, ext_feat, ext_df, target_col=target_col)

    print(f"  [{label}] Monthly resampling...")
    monthly     = feat_a.resample_to_monthly(ext_feat, ext_df, target_col=target_col)

    performance = mod_a.compute_full_performance(rf, ext_feat, ext_df, target_col=target_col)
    shap_dict   = dict(zip(feat_names, shap_vals.tolist()))

    return monthly, performance, shap_dict, wf_results


def run_model_a(extended_market: pd.DataFrame):
    print("\n" + "=" * 60)
    print("MODEL A PIPELINE  (2009–2011 Original → scored 2003-2026)")
    print("  Two variants: ISE_USD (currency-adjusted) + ISE_TL (nominal)")
    print("=" * 60)

    print("\n[1] Loading Group_5.csv...")
    train_df = pre_a.load_group5()

    print("\n[2] Preparing extended dataset for OOS scoring...")
    try:
        ext_df = pre_a.load_extended_for_model_a(extended_market)
    except Exception as e:
        print(f"  Warning: {e} — using training data only")
        ext_df = train_df.copy()

    print("\n[3] Running Model A — ISE_USD variant...")
    usd = _run_model_a_variant(train_df, ext_df, 'ISE_USD', 'USD')

    print("\n[4] Running Model A — ISE_TL variant...")
    tl  = _run_model_a_variant(train_df, ext_df, 'ISE_TL',  'TL')

    return usd, tl   # each is (monthly, performance, shap_dict, wf_results)


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Financial Fragility Clock — Pipeline v2.0")
    print("=" * 60)

    # Model B first (fetches extended market data — reused by Model A)
    b_monthly, b_perf, b_shap, b_wf = run_model_b()

    print("\nRe-fetching market data for Model A OOS scoring...")
    try:
        mkt_ext = fetch_market_data(start_date=START_DATE, end_date=END_DATE)
        mkt_ext = derive_extended_features(mkt_ext)
        mkt_ext = handle_missing_data(mkt_ext, max_gap=5)
    except Exception as e:
        print(f"  Warning: {e}")
        mkt_ext = pd.DataFrame()

    (a_usd_monthly, a_usd_perf, a_usd_shap, a_usd_wf), \
    (a_tl_monthly,  a_tl_perf,  a_tl_shap,  a_tl_wf)  = run_model_a(mkt_ext)

    model_a_usd = exporter.build_model_a_object(a_usd_monthly, a_usd_perf, a_usd_shap, a_usd_wf, variant='usd')
    model_a_tl  = exporter.build_model_a_object(a_tl_monthly,  a_tl_perf,  a_tl_shap,  a_tl_wf,  variant='tl')
    model_b_obj = exporter.build_model_b_object(b_monthly, b_perf, b_shap, b_wf)

    exporter.export(model_a_usd, model_b_obj, model_a_tl=model_a_tl)

    print("\nPipeline complete. Open index.html in a browser (served via HTTP).")
    print("Quick serve: python -m http.server 8000 --directory ..")
