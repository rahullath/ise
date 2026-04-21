"""
Model A — RF + OLS + VAR trained on Group_5.csv (2009-2011),
scored out-of-sample on extended dataset (2003-2026).

Walk-forward validation splits (to show Model A's limitations):
  Split 1: Test 2018 TRY currency crisis   → model misses it (trained pre-2018)
  Split 2: Test 2020 COVID selloff         → model partially detects (global signal)
  Split 3: Test 2021-2024 Turkey crisis    → model misses (unorthodox monetary policy)

Key academic finding: Model A scores 2018 ≈ 43, 2021+ ≈ 39 — both in SPECULATIVE/HEDGE.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import shap
import warnings
from typing import Dict, Tuple, List

from feature_engineering import compute_features, resample_to_monthly, INDICES, TARGET, TARGET_TL


RF_PARAMS = dict(n_estimators=500, max_depth=10, random_state=42, n_jobs=-1)

FEATURE_COLS = ['mean_corr', 'pe_inv', 'rolling_volatility', 'rf_prediction_error']

WALK_FORWARD_SPLITS = [
    {
        'name':        'split_2018_try',
        'test_start':  '2018-01-01',
        'test_end':    '2018-12-31',
        'description': '2018 Turkish Lira Currency Crisis',
    },
    {
        'name':        'split_2020_covid',
        'test_start':  '2020-01-01',
        'test_end':    '2020-12-31',
        'description': 'COVID-19 Synchronized Selloff',
    },
    {
        'name':        'split_2021_turkey',
        'test_start':  '2021-01-01',
        'test_end':    '2024-12-31',
        'description': 'Ongoing Turkey Economic Crisis 2021-2024',
    },
]


# ── metric helpers ────────────────────────────────────────────────────────────

def _metrics(y_true, y_pred) -> Dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        'r2':   round(float(r2_score(y_true, y_pred)), 4),
        'rmse': round(rmse, 4),
        'mae':  round(float(mean_absolute_error(y_true, y_pred)), 4),
    }


def _regime_rmse(y_true, y_pred, regimes) -> Dict:
    out = {}
    for r, key in [('HEDGE', 'hedge_rmse'), ('SPECULATIVE', 'spec_rmse'), ('PONZI', 'ponzi_rmse')]:
        mask = regimes == r
        if mask.sum() < 2:
            out[key] = None
        else:
            out[key] = round(float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))), 4)
    return out


# ── model training ────────────────────────────────────────────────────────────

def train_model_a(train_feat: pd.DataFrame,
                  train_returns: pd.DataFrame,
                  target_col: str = TARGET) -> Tuple[RandomForestRegressor, np.ndarray, List[str]]:
    """Train RF on Model A training window (2009-2011).

    Args:
        target_col: 'ISE_USD' (default) or 'ISE_TL'
    Returns (rf_model, shap_values_mean_abs, feature_names).
    """
    X = train_feat[FEATURE_COLS].dropna()
    y = train_returns[target_col].reindex(X.index).dropna()
    X = X.reindex(y.index)

    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X, y)

    explainer = shap.TreeExplainer(rf)
    shap_vals = explainer.shap_values(X)
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)

    print(f"  RF trained on {len(X)} samples (target={target_col}), R²={rf.score(X, y):.3f}")
    return rf, mean_abs_shap, FEATURE_COLS


def score_oos(rf: RandomForestRegressor,
              full_feat: pd.DataFrame,
              full_returns: pd.DataFrame,
              train_end: str = '2011-08-31',
              target_col: str = TARGET) -> pd.Series:
    """Compute rolling RF prediction error for OOS period (4th fragility feature)."""
    X_all = full_feat[FEATURE_COLS].dropna()
    y_all = full_returns[target_col].reindex(X_all.index).dropna()
    X_all = X_all.reindex(y_all.index)

    y_pred = pd.Series(rf.predict(X_all), index=X_all.index)
    error  = (y_all - y_pred).abs()
    return error.rolling(30).mean()


def _ols_metrics(X_train, y_train, X_test, y_test) -> Dict:
    try:
        import statsmodels.api as sm
        X_tr = sm.add_constant(X_train)
        X_te = sm.add_constant(X_test)
        model = sm.OLS(y_train, X_tr).fit()
        y_pred = model.predict(X_te)
        return _metrics(y_test.values, y_pred.values)
    except Exception as e:
        print(f"  OLS failed: {e}")
        return {'r2': None, 'rmse': None, 'mae': None}


def _var_metrics(train_df, test_df, target_col, n_features=4) -> Dict:
    """VAR on top features to keep model parsimony (VAR is sensitive to dimensionality)."""
    try:
        from statsmodels.tsa.vector_ar.var_model import VAR
        cols = [target_col] + FEATURE_COLS[:min(n_features, len(FEATURE_COLS))]
        cols = [c for c in cols if c in train_df.columns and c in test_df.columns]

        tr = train_df[cols].dropna()
        te = test_df[cols].dropna()
        if len(tr) < 20 or len(te) < 5:
            return {'r2': None, 'rmse': None, 'mae': None}

        var_model = VAR(tr)
        order = var_model.select_order(maxlags=4).selected_orders.get('aic', 2)
        result = var_model.fit(maxlags=max(1, order), ic=None)

        # One-step ahead forecasts on test set
        preds = []
        history = list(tr.values[-result.k_ar:])
        for _ in range(len(te)):
            fc = result.forecast(np.array(history), steps=1)[0]
            preds.append(fc[0])  # target col is first
            history = history[1:] + [list(te.iloc[len(preds) - 1])]

        y_true = te[target_col].values
        y_pred = np.array(preds)
        return _metrics(y_true, y_pred)
    except Exception as e:
        print(f"  VAR failed: {e}")
        return {'r2': None, 'rmse': None, 'mae': None}


# ── walk-forward validation ───────────────────────────────────────────────────

def run_walk_forward(rf: RandomForestRegressor,
                     full_feat: pd.DataFrame,
                     full_returns: pd.DataFrame,
                     target_col: str = TARGET) -> List[Dict]:
    """Evaluate Model A on the three OOS test windows (2018, 2020, 2021-24)."""
    train_feat_cols = full_feat[FEATURE_COLS].dropna()
    train_ret       = full_returns[target_col].reindex(train_feat_cols.index).dropna()
    train_feat_cols = train_feat_cols.reindex(train_ret.index)

    results = []
    for sp in WALK_FORWARD_SPLITS:
        test_f = full_feat.loc[sp['test_start']:sp['test_end'], FEATURE_COLS].dropna()
        test_r = full_returns[target_col].reindex(test_f.index).dropna()
        test_f = test_f.reindex(test_r.index)

        if len(test_f) < 5:
            print(f"  Skipping {sp['name']}: insufficient test data")
            continue

        y_pred_rf = rf.predict(test_f)
        base = _metrics(test_r.values, y_pred_rf)

        test_feat_full = full_feat.reindex(test_f.index)
        regimes = test_feat_full['regime'] if 'regime' in test_feat_full.columns else pd.Series('HEDGE', index=test_f.index)

        base.update(_regime_rmse(test_r.values, y_pred_rf, regimes.values))

        ols = _ols_metrics(train_feat_cols, train_ret, test_f, test_r)
        ols.update(_regime_rmse(test_r.values,
                                ols.get('_pred', np.zeros_like(test_r.values)),
                                regimes.values))

        var_train = pd.concat([train_feat_cols, train_ret], axis=1).dropna()
        var_test  = pd.concat([test_f, test_r], axis=1).dropna()
        var_m = _var_metrics(var_train, var_test, target_col)

        print(f"  {sp['name']}: RF R²={base['r2']}, OLS R²={ols['r2']}, VAR R²={var_m['r2']}")
        results.append({
            'split': sp['name'],
            'description': sp['description'],
            'test_start': sp['test_start'],
            'test_end':   sp['test_end'],
            'n_test':     len(test_f),
            'rf':  base,
            'ols': ols,
            'var': var_m,
        })
    return results


# ── full performance across all scored months ─────────────────────────────────

def compute_full_performance(rf: RandomForestRegressor,
                              full_feat: pd.DataFrame,
                              full_returns: pd.DataFrame,
                              train_end: str = '2011-08-31',
                              target_col: str = TARGET) -> Dict:
    """Aggregate RF+OLS+VAR metrics over the full OOS period (post train_end)."""
    X_oos = full_feat.loc[train_end:, FEATURE_COLS].dropna()
    y_oos = full_returns[target_col].reindex(X_oos.index).dropna()
    X_oos = X_oos.reindex(y_oos.index)

    if len(X_oos) < 10:
        return {}

    y_pred_rf = rf.predict(X_oos)
    base = _metrics(y_oos.values, y_pred_rf)

    regimes = full_feat.reindex(X_oos.index).get('regime', pd.Series('HEDGE', index=X_oos.index))
    base.update(_regime_rmse(y_oos.values, y_pred_rf, regimes.values))

    # OLS
    X_tr = full_feat.loc[:train_end, FEATURE_COLS].dropna()
    y_tr = full_returns[target_col].reindex(X_tr.index).dropna()
    X_tr = X_tr.reindex(y_tr.index)
    ols = _ols_metrics(X_tr, y_tr, X_oos, y_oos)
    ols.update(_regime_rmse(y_oos.values,
                            np.zeros_like(y_oos.values),
                            regimes.values))

    var_train = pd.concat([X_tr, y_tr], axis=1).dropna()
    var_test  = pd.concat([X_oos.iloc[:500], y_oos.iloc[:500]], axis=1).dropna()
    var_m = _var_metrics(var_train, var_test, target_col)

    return {
        'random_forest': base,
        'ols':           ols,
        'var':           var_m,
    }
