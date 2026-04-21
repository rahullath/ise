"""
Unified JSON exporter — writes src/data/fragility_output.json.

Reads processed results from Model A and Model B and combines them into
the single schema the dashboard expects:

  {
    "version": "2.0",
    "generated_at": "...",
    "models": {
      "model_2009": { meta, performance, shap_values, monthly_scores },
      "model_2003": { meta, performance, shap_values, monthly_scores }
    }
  }
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "fragility_output.json"


# ── helpers ───────────────────────────────────────────────────────────────────

def _safe(v):
    """Convert numpy scalars and NaN to JSON-safe Python types."""
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        if np.isnan(v) or np.isinf(v):
            return None
        return round(float(v), 6)
    return v


def _perf_dict(metrics: dict) -> dict:
    """Standardise a metrics dict to the schema's performance sub-object."""
    return {
        'r2':          _safe(metrics.get('r2')),
        'rmse':        _safe(metrics.get('rmse')),
        'mae':         _safe(metrics.get('mae')),
        'hedge_rmse':  _safe(metrics.get('hedge_rmse') or metrics.get('HEDGE', {}).get('rmse')),
        'spec_rmse':   _safe(metrics.get('spec_rmse')  or metrics.get('SPECULATIVE', {}).get('rmse')),
        'ponzi_rmse':  _safe(metrics.get('ponzi_rmse') or metrics.get('PONZI', {}).get('rmse')),
    }


def _build_monthly_scores(monthly_df: pd.DataFrame) -> list:
    """Convert monthly DataFrame to the list-of-dicts the dashboard expects."""
    records = []
    corr_map = {
        'SP500': 'sp500', 'DAX': 'dax', 'FTSE': 'ftse',
        'NIKKEI': 'nikkei', 'BOVESPA': 'bovespa', 'EU': 'eu', 'EM': 'em',
    }
    for ts, row in monthly_df.iterrows():
        # pe_inv may be stored directly or derived from permutation_entropy
        pe_inv_val = row.get('pe_inv')
        if pe_inv_val is None and row.get('permutation_entropy') is not None:
            pe_inv_val = 1.0 - row['permutation_entropy']

        rec = {
            'date':            ts.strftime('%Y-%m-%d'),
            'fragility_score': _safe(row.get('fragility_score')),
            'regime':          str(row.get('regime', 'HEDGE')),
            'components': {
                'rolling_correlation_60d': _safe(row.get('mean_corr')),
                'permutation_entropy_inv': _safe(pe_inv_val),
                'rolling_volatility_30d':  _safe(row.get('rolling_volatility')),
                'rf_prediction_error':     _safe(row.get('rf_prediction_error')),
            },
            'correlations': {},
            'features': {},
        }
        # Pairwise correlations vs ISE_USD
        for raw_key, json_key in corr_map.items():
            col = f'{raw_key}_corr'
            if col in row.index:
                rec['correlations'][json_key] = _safe(row[col])

        # Extended features for Model B
        for col in ['VIX', 'DXY', 'BRENT', 'EUR_USD', 'TRY_USD', 'US_10Y_YIELD',
                    'cbrt_governor_change']:
            if col in row.index:
                rec['features'][col.lower()] = _safe(row[col])
        # Lag features
        for col in [c for c in row.index if '_lag' in c]:
            rec['features'][col.lower()] = _safe(row[col])

        records.append(rec)
    return records


# ── model object builders ──────────────────────────────────────────────────────

def build_model_a_object(monthly_scores: pd.DataFrame,
                          performance: dict,
                          shap_mean_abs: dict,
                          walk_forward: list = None,
                          variant: str = 'usd') -> dict:
    """Build model_2009 export object.

    variant: 'usd' → ISE_USD (currency-adjusted, current Model A)
             'tl'  → ISE_TL (nominal TRY, inflation-tracking baseline)
    """
    is_tl = variant == 'tl'
    model_id    = 'model_2009_tl' if is_tl else 'model_2009'
    label       = '2009–2011 TL-Nominal' if is_tl else '2009–2011 Original'
    target_name = 'ise_tl_ret' if is_tl else 'ise_usd_ret'
    desc        = ('TRY-denominated ISE — tracks nominal inflation, misses '
                   'currency crises entirely') if is_tl else \
                  ('USD-denominated ISE — embeds TRY implicitly, still misses '
                   '2018 & 2021-26 without explicit macro features')
    return {
        'meta': {
            'id':            model_id,
            'label':         label,
            'description':   desc,
            'variant':       variant,
            'training_start': '2009-01-01',
            'training_end':   '2011-07-31',
            'scoring_start':  monthly_scores.index.min().strftime('%Y-%m-%d'),
            'scoring_end':    monthly_scores.index.max().strftime('%Y-%m-%d'),
            'n_features':     7,
            'feature_names':  ['sp500_ret', 'dax_ret', 'ftse_ret', 'nikkei_ret',
                               'bovespa_ret', 'eu_ret', 'em_ret'],
            'target':         target_name,
        },
        'performance': {
            'random_forest': _perf_dict(performance.get('random_forest', {})),
            'ols':           _perf_dict(performance.get('ols', {})),
            'var':           _perf_dict(performance.get('var', {})),
        },
        'shap_values':    {k: _safe(v) for k, v in shap_mean_abs.items()},
        'walk_forward':   walk_forward or [],
        'monthly_scores': _build_monthly_scores(monthly_scores),
    }


def build_model_b_object(monthly_scores: pd.DataFrame,
                          performance: dict,
                          shap_mean_abs: dict,
                          walk_forward: list = None) -> dict:
    return {
        'meta': {
            'id':            'model_2003',
            'label':         '2003–2026 Extended',
            'training_start': '2003-01-01',
            'training_end':   monthly_scores.index.max().strftime('%Y-%m-%d'),
            'scoring_start':  monthly_scores.index.min().strftime('%Y-%m-%d'),
            'scoring_end':    monthly_scores.index.max().strftime('%Y-%m-%d'),
            'n_features':     14,
            'feature_names':  ['sp500_ret', 'dax_ret', 'ftse_ret', 'nikkei_ret',
                               'bovespa_ret', 'eu_ret', 'em_ret',
                               'vix', 'dxy', 'us_10y_yield', 'brent_crude',
                               'eur_usd', 'try_usd', 'cbrt_governor_change'],
            'target':         'ise_usd_ret',
        },
        'performance': {
            'random_forest': _perf_dict(performance.get('random_forest', {})),
            'ols':           _perf_dict(performance.get('ols', {})),
            'var':           _perf_dict(performance.get('var', {})),
        },
        'shap_values':    {k: _safe(v) for k, v in shap_mean_abs.items()},
        'walk_forward':   walk_forward or [],
        'monthly_scores': _build_monthly_scores(monthly_scores),
    }


# ── main export ───────────────────────────────────────────────────────────────

def export(model_a: dict, model_b: dict,
           model_a_tl: dict = None,
           output_path: Path = OUTPUT_PATH) -> None:
    """Write the unified fragility_output.json."""
    models = {
        'model_2009':    model_a,   # ISE_USD variant
        'model_2003':    model_b,   # Extended with TRY_weakness
    }
    if model_a_tl is not None:
        models['model_2009_tl'] = model_a_tl  # ISE_TL nominal baseline

    output = {
        'version':      '2.0',
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'models':       models,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nExported → {output_path}")
    for key, obj in models.items():
        n = len(obj.get('monthly_scores', []))
        print(f"  {key}: {n} monthly observations")
