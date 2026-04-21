"""
Model A Preprocessing — loads Group_5.csv (2009-01 to 2011-02).

Group_5.csv is already daily log returns:
  row 0  = meta header (TL BASED / USD BASED)
  row 1  = column names
  cols   = date | ISE_TL | ISE_USD | SP500 | DAX | FTSE | NIKKEI | BOVESPA | EU | EM

Target: ISE_USD (USD-denominated returns — avoids the inflation confound).
"""

import pandas as pd
import numpy as np
from pathlib import Path

GROUP5_CSV = Path(__file__).parent.parent.parent / "context-dump" / "converted" / "Group_5.csv"

INDICES = ['SP500', 'DAX', 'FTSE', 'NIKKEI', 'BOVESPA', 'EU', 'EM']
TARGET     = 'ISE_USD'
TARGET_TL  = 'ISE_TL'
ALL_COLS   = [TARGET, TARGET_TL] + INDICES


def load_group5() -> pd.DataFrame:
    """Return daily log-return DataFrame from Group_5.csv.

    Returns a clean DataFrame with DatetimeIndex, columns:
      ISE_USD, ISE_TL, SP500, DAX, FTSE, NIKKEI, BOVESPA, EU, EM

    ISE_TL is kept alongside ISE_USD so Model A can be scored on both
    targets (TL = nominal/inflation-tracking, USD = real/currency-adjusted).
    """
    df = pd.read_csv(GROUP5_CSV, header=1)
    df.columns = ['date', TARGET_TL, TARGET, 'SP500', 'DAX',
                  'FTSE', 'NIKKEI', 'BOVESPA', 'EU', 'EM']

    df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=True)
    df = df.set_index('date').sort_index()

    # Exact 0.0 entries in the CSV indicate missing data for that day
    df = df.replace(0.0, np.nan)
    df = df.ffill().bfill()
    df = df.dropna()

    print(f"Group_5.csv loaded: {df.shape[0]} rows, {df.index.min().date()} → {df.index.max().date()}")
    return df


def load_extended_for_model_a(extended_market: pd.DataFrame) -> pd.DataFrame:
    """Extract Model A columns from the extended (Model B) market DataFrame.

    Returns both ISE_USD and ISE_TL so we can score OOS on both targets:
    - ISE_USD: USD-denominated (captures TRY weakness implicitly)
    - ISE_TL:  TRY-denominated (nominal, inflation-tracking only)

    Args:
        extended_market: DataFrame with at least [SP500, DAX, FTSE, NIKKEI,
                         BOVESPA, EU, EM, BIST100, USDTRY].
    Returns:
        DataFrame with columns [ISE_USD, ISE_TL, SP500, …, EM], daily log returns.
    """
    df = extended_market.copy()

    if 'BIST100' not in df.columns:
        raise ValueError("Extended data missing BIST100")

    # ISE_USD = BIST100 / USDTRY (price level), then log returns
    if 'ISE_USD' not in df.columns:
        if 'USDTRY' not in df.columns:
            raise ValueError("Extended data missing USDTRY to derive ISE_USD")
        ise_usd_prices = df['BIST100'] / df['USDTRY']
        df['ISE_USD'] = np.log(ise_usd_prices / ise_usd_prices.shift(1))

    # ISE_TL = BIST100 price level → log returns
    if 'ISE_TL' not in df.columns:
        bist = df['BIST100']
        if bist.abs().mean() > 1.0:   # price level
            df['ISE_TL'] = np.log(bist / bist.shift(1))
        else:
            df['ISE_TL'] = bist        # already returns

    # Compute log returns for the 7 indices if they are price levels
    for col in INDICES:
        if col in df.columns and df[col].abs().mean() > 1.0:
            df[col] = np.log(df[col] / df[col].shift(1))

    df = df[ALL_COLS].replace([np.inf, -np.inf], np.nan).dropna()
    return df


if __name__ == '__main__':
    df = load_group5()
    print(df.describe().round(6))
