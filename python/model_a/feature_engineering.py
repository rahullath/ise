"""
Model A Feature Engineering.

Four-component fixed-weight fragility score (no dynamic redistribution):

  score = (0.40 * corr_norm + 0.30 * pe_inv_norm + 0.20 * vol_norm + 0.10 * rf_err_norm) * 100

where each component is min-max normalised to [0, 1] over the full series.
"""

import pandas as pd
import numpy as np
from itertools import combinations
from math import factorial
from typing import Optional

INDICES   = ['SP500', 'DAX', 'FTSE', 'NIKKEI', 'BOVESPA', 'EU', 'EM']
TARGET    = 'ISE_USD'
TARGET_TL = 'ISE_TL'

WEIGHTS = {
    'corr':   0.40,
    'pe_inv': 0.30,
    'vol':    0.20,
    'rf_err': 0.10,
}

CORR_WINDOW  = 60  # days
VOL_WINDOW   = 30  # days
PE_ORDER     = 3   # permutation entropy order


# ── helpers ──────────────────────────────────────────────────────────────────

def _permutation_entropy(series: np.ndarray, order: int = 3) -> float:
    """Compute permutation entropy for a 1-D array (normalized, 0-1)."""
    n = len(series)
    if n < order:
        return np.nan
    count: dict = {}
    for i in range(n - order + 1):
        pattern = tuple(np.argsort(series[i:i + order]))
        count[pattern] = count.get(pattern, 0) + 1
    n_patterns = sum(count.values())
    probs = [c / n_patterns for c in count.values()]
    pe = -sum(p * np.log2(p) for p in probs if p > 0)
    max_pe = np.log2(factorial(order))
    return pe / max_pe if max_pe > 0 else 0.0


def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


# ── main feature computation ──────────────────────────────────────────────────

def compute_features(df: pd.DataFrame,
                     rf_error_series: Optional[pd.Series] = None,
                     target_col: str = TARGET) -> pd.DataFrame:
    """Compute Model A features on any daily returns DataFrame.

    Args:
        df: Daily log-return DataFrame with columns [ISE_USD, SP500, ..., EM].
            Must contain target_col.
        rf_error_series: Optional pre-computed rolling RF prediction error series.
        target_col: Which ISE series to use as the fragility target.
                    'ISE_USD' (default) or 'ISE_TL'.
    Returns:
        DataFrame with feature columns + fragility_score + regime.
    """
    feat = pd.DataFrame(index=df.index)

    all_indices = [c for c in INDICES if c in df.columns]
    tgt = target_col if target_col in df.columns else TARGET
    pair_cols   = [c for c in all_indices] + [tgt]

    # ── 1. Rolling mean pairwise Pearson correlation (60-day) ────────────────
    mean_corrs = []
    for i in range(len(df)):
        if i < CORR_WINDOW - 1:
            mean_corrs.append(np.nan)
        else:
            window = df[pair_cols].iloc[i - CORR_WINDOW + 1: i + 1]
            cm = window.corr().values
            upper = cm[np.triu_indices_from(cm, k=1)]
            mean_corrs.append(float(np.nanmean(np.abs(upper))))
    feat['mean_corr'] = mean_corrs

    # ── 2. Permutation entropy on target series (inverted) ───────────────────
    pe_vals = []
    ise_arr = df[tgt].values if tgt in df.columns else df.iloc[:, 0].values
    for i in range(len(df)):
        if i < PE_ORDER - 1:
            pe_vals.append(np.nan)
        else:
            pe_vals.append(_permutation_entropy(ise_arr[max(0, i - 29): i + 1], PE_ORDER))
    feat['permutation_entropy'] = pe_vals
    feat['pe_inv'] = 1.0 - feat['permutation_entropy']

    # ── 3. Rolling 30-day volatility of target ────────────────────────────────
    feat['rolling_volatility'] = df[tgt].rolling(VOL_WINDOW).std() if tgt in df.columns else pd.Series(np.nan, index=df.index)

    # ── 4. RF prediction error ───────────────────────────────────────────────
    if rf_error_series is not None:
        feat['rf_prediction_error'] = rf_error_series.reindex(df.index)
    else:
        feat['rf_prediction_error'] = pd.Series(0.0, index=df.index)

    # ── Per-index correlations with target (for monthly export) ──────────────
    for idx in all_indices:
        feat[f'{idx}_corr'] = df[tgt].rolling(CORR_WINDOW).corr(df[idx]) if tgt in df.columns else np.nan

    # ── Normalise & score ─────────────────────────────────────────────────────
    feat = feat.dropna(subset=['mean_corr', 'rolling_volatility'])

    corr_n   = _minmax(feat['mean_corr'])
    pe_inv_n = _minmax(feat['pe_inv'])
    vol_n    = _minmax(feat['rolling_volatility'])
    rf_err_n = _minmax(feat['rf_prediction_error'])

    w = WEIGHTS
    feat['fragility_score'] = (
        w['corr']   * corr_n   +
        w['pe_inv'] * pe_inv_n +
        w['vol']    * vol_n    +
        w['rf_err'] * rf_err_n
    ) * 100.0

    feat['regime'] = feat['fragility_score'].apply(
        lambda s: 'PONZI' if s >= 70 else ('SPECULATIVE' if s >= 40 else 'HEDGE')
    )

    return feat


def resample_to_monthly(feat: pd.DataFrame, returns: pd.DataFrame,
                        target_col: str = TARGET) -> pd.DataFrame:
    """Resample daily feature DataFrame to monthly (last trading day of each month)."""
    monthly = pd.DataFrame(index=feat.resample('ME').last().index)
    monthly['fragility_score']     = feat['fragility_score'].resample('ME').last()
    monthly['regime']              = feat['regime'].resample('ME').last()
    monthly['mean_corr']           = feat['mean_corr'].resample('ME').mean()
    monthly['permutation_entropy'] = feat['permutation_entropy'].resample('ME').mean()
    monthly['pe_inv']              = feat['pe_inv'].resample('ME').mean()
    monthly['rolling_volatility']  = feat['rolling_volatility'].resample('ME').mean()
    monthly['rf_prediction_error'] = feat['rf_prediction_error'].resample('ME').mean()
    monthly['target_col']          = target_col   # tag so exporter knows which series

    corr_cols = [c for c in feat.columns if c.endswith('_corr')]
    for col in corr_cols:
        monthly[col] = feat[col].resample('ME').last()

    monthly = monthly.dropna(subset=['fragility_score'])
    return monthly
