"""
Reusable GNSS-IR analysis: arc split → detrend → Lomb-Scargle.

Returns one summary row per satellite arc, suitable for aggregation across
stations and days.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import lombscargle

__all__ = [
    "IRConfig",
    "analyze_merged_full",
]


@dataclass(slots=True)
class IRConfig:
    """Parameters controlling the GNSS-IR analysis pipeline."""

    elev_min: float = 5.0
    elev_max: float = 30.0
    poly_deg: int = 2
    system: str = "G"
    obs_code: str = "1C"
    wavelength: float = 0.1903  # L1 GPS
    h_min: float = 0.1
    h_max: float = 10.0
    h_step: float = 0.01
    sample_interval_s: int = 15
    gap_threshold_min: int = 10


# ---------------------------------------------------------------------------
# Arc splitting + detrending
# ---------------------------------------------------------------------------


def split_arcs(df: pd.DataFrame, gap_threshold_min: int = 10) -> list[pd.DataFrame]:
    """Split a single-satellite DataFrame into rising/setting arcs."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    time_diff = df["timestamp"].diff().dt.total_seconds().fillna(0)
    time_gap = time_diff > gap_threshold_min * 60
    elev_diff = df["elevation"].diff()
    elev_rev = ((elev_diff * elev_diff.shift(1)) < 0).fillna(False)
    arc_id = (time_gap | elev_rev).cumsum()
    return [g.reset_index(drop=True) for _, g in df.groupby(arc_id) if len(g) >= 10]


def detrend_snr(arc: pd.DataFrame, poly_deg: int) -> pd.DataFrame | None:
    """Remove a polynomial trend from SNR using sin(elevation) as the x-axis."""
    x = np.sin(np.radians(arc["elevation"].values))
    y = arc["snr"].values
    valid = np.isfinite(y)
    if valid.sum() < poly_deg + 2:
        return None
    if arc["elevation"].max() - arc["elevation"].min() < 2.0:
        return None
    coeffs = np.polyfit(x[valid], y[valid], poly_deg)
    trend = np.polyval(coeffs, x)
    arc = arc.copy()
    arc["snr_detrend"] = y - trend
    arc["sin_elev"] = x
    return arc


# ---------------------------------------------------------------------------
# Lomb-Scargle
# ---------------------------------------------------------------------------


def lomb_scargle_arc(
    arc: pd.DataFrame,
    heights: np.ndarray,
    omegas: np.ndarray,
    full_periodogram: bool = False,
) -> tuple[float, float] | tuple[float, float, np.ndarray] | None:
    """Return (peak_height, peak_power) or, with full_periodogram, also the pgram array."""
    x = arc["sin_elev"].values
    y = arc["snr_detrend"].values
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 10:
        return None
    y_norm = y[valid] - y[valid].mean()
    pgram = lombscargle(x[valid], y_norm, omegas, normalize=True)
    peak_idx = np.argmax(pgram)
    if full_periodogram:
        return heights[peak_idx], pgram[peak_idx], pgram
    return heights[peak_idx], pgram[peak_idx]


# ---------------------------------------------------------------------------
# Top-level: analyse a pre-merged dataset across multiple days
# ---------------------------------------------------------------------------


def analyze_merged_full(
    merged: pd.DataFrame,
    station: str,
    dates: Sequence[dt.date],
    cfg: IRConfig | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Run the full arc + Lomb-Scargle pipeline on a pre-merged SNR + ephemeris
    DataFrame, slicing by UTC calendar day for each entry in *dates*.

    *merged* must include columns ``timestamp``, ``satellite``, ``snr``,
    ``obs_code``, ``elevation``, and ``azimuth`` (e.g. from an inner join of
    SNR observations with ephemeris positions on timestamp + satellite).

    Returns ``(summary_df, pgram_matrix, heights_grid)`` where *pgram_matrix*
    has shape ``(n_arcs, len(heights_grid))``.
    """
    cfg = cfg or IRConfig()
    heights = np.arange(cfg.h_min, cfg.h_max, cfg.h_step)
    omegas = 2 * np.pi * (2 * heights / cfg.wavelength)

    merged = merged.copy()
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True)

    rows_all: list[dict[str, Any]] = []
    pgrams_all: list[np.ndarray] = []

    for date in dates:
        start = dt.datetime(date.year, date.month, date.day, tzinfo=dt.timezone.utc)
        end = start + dt.timedelta(days=1)
        sub = merged.loc[(merged["timestamp"] >= start) & (merged["timestamp"] < end)]
        if sub.empty:
            continue

        top_code = sub.groupby("obs_code")["snr"].count().idxmax()
        filtered = sub[
            (sub["obs_code"] == top_code)
            & sub["snr"].notna()
            & (sub["elevation"] >= cfg.elev_min)
            & (sub["elevation"] <= cfg.elev_max)
        ]
        if filtered.empty:
            continue

        for sat, grp in filtered.groupby("satellite"):
            for arc in split_arcs(grp, cfg.gap_threshold_min):
                arc = detrend_snr(arc, cfg.poly_deg)
                if arc is None:
                    continue
                out = lomb_scargle_arc(arc, heights, omegas, full_periodogram=True)
                if out is None:
                    continue
                peak_h, peak_pwr, pgram = out
                rows_all.append(
                    {
                        "station": station,
                        "date": date,
                        "satellite": sat,
                        "obs_code": top_code,
                        "azimuth_mean": arc["azimuth"].mean(),
                        "elev_mean": arc["elevation"].mean(),
                        "t_center": arc["timestamp"].mean(),
                        "peak_height": peak_h,
                        "peak_power": peak_pwr,
                        "n_pts": len(arc),
                    }
                )
                pgrams_all.append(pgram)

    if not rows_all:
        return pd.DataFrame(), np.empty((0, len(heights))), heights

    return pd.DataFrame(rows_all), np.array(pgrams_all), heights
