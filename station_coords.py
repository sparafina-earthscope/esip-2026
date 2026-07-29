"""
Load GAGE GPS station coordinates from igs14 text file.

File format (space-delimited):
  4CHARID, Station Name, Latitude (deg), Longitude (deg), Ellipsoidal Elevation (m),
  X (m), Y (m), Z (m), Epoch Date (YYYYMMDD)
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

__all__ = ["StationCoords", "load_station_coords"]


@dataclass(frozen=True, slots=True)
class StationCoords:
    """Geodetic position of a single GNSS station."""

    id: str           # 4-char ID
    name: str
    lat: float        # degrees
    lon: float        # degrees
    height: float     # ellipsoidal height, meters


def load_station_coords(path: str | Path) -> dict[str, StationCoords]:
    """Parse gage_gps.igs14.txt and return a dict keyed by 4-char station ID."""
    stations: dict[str, StationCoords] = {}
    path = Path(path)

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            try:
                sid = parts[0].strip().upper()
                name = parts[1].strip()
                lat = float(parts[2])
                lon = float(parts[3])
                height = float(parts[4])
                stations[sid] = StationCoords(id=sid, name=name, lat=lat, lon=lon, height=height)
            except (ValueError, IndexError):
                continue

    return stations


if __name__ == "__main__":
    coords = load_station_coords("/Users/berglund/Desktop/gage_gps.igs14.txt")
    print(f"Loaded {len(coords)} stations")
    # Spot check
    for sid in ["P057", "AB02", "1LSU"]:
        if sid in coords:
            print(f"  {sid}: {coords[sid]}")
