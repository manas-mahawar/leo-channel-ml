"""Hourly ITU-R attenuation labels at 39 GHz (gas + cloud + rain = EPL).

Paper (Sect. 3.1): EPL is the sum of gaseous (P.676), cloud (P.840 with an
RH/precipitation LWP proxy) and rain (P.838 + P.618, rain height from P.839)
attenuation, at a fixed elevation angle.

Usage:
    python physics/build_attenuation.py            # six training cities
    python physics/build_attenuation.py --unseen   # ten unseen cities

Output: one CSV per city in config.PHYSICS_DIR with the weather columns,
every intermediate quantity and the three components plus EPL.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import itur
from itur.models import itu676, itu838, itu839, itu840, itu1511

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from data_pipeline.cities import CITIES, UNSEEN_CITIES

# The paper cites P.840-8. itur 0.4.0 supports it (default is -7).
# P.676-12, P.618-13, P.838-3, P.839-4 are the itur defaults.
itu840.change_version(8)

F = config.FREQUENCY_GHZ
EL = config.ELEVATION_DEG
SIN_EL = np.sin(np.deg2rad(EL))


def value(x):
    """Strip astropy units returned by itur."""
    return np.asarray(getattr(x, "value", x), dtype=float)


# -----------------------------------------------------------------
# Gaseous attenuation: ITU-R P.676-12, Annex 2 slant path
# -----------------------------------------------------------------
def water_vapour(temp_c, rh_percent):
    """Water vapour pressure e (hPa) and density rho (g/m^3), P.453."""
    es = 6.1121 * np.exp((18.678 - temp_c / 234.5) * temp_c / (257.14 + temp_c))
    e = rh_percent / 100.0 * es
    rho = 216.7 * e / (temp_c + 273.15)
    return e, rho


def gaseous_attenuation(temp_c, rh_percent, pressure_hpa):
    """A_gas = (gamma_o * h_o + gamma_w * h_w) / sin(el).

    gamma_o, gamma_w: surface specific attenuation, Annex 1 line-by-line.
    h_o, h_w: Annex 2 equivalent heights. P.676 uses dry-air pressure.
    """
    t_k = temp_c + 273.15
    e, rho = water_vapour(temp_c, rh_percent)
    p_dry = pressure_hpa - e

    gamma_o = value(itu676.gamma0_exact(F, p_dry, rho, t_k))
    gamma_w = value(itu676.gammaw_exact(F, p_dry, rho, t_k))
    # itur returns one row [h_o, h_w] per hour, in km (labelled as m).
    heights = np.atleast_2d(value(
        itu676.slant_inclined_path_equivalent_height(F, p_dry, rho, t_k)))
    h_o, h_w = heights[:, 0], heights[:, 1]

    a_gas = (gamma_o * h_o + gamma_w * h_w) / SIN_EL
    return pd.DataFrame({
        "rho_g_m3": rho,
        "gamma_o_dB_km": gamma_o,
        "gamma_w_dB_km": gamma_w,
        "h_o_km": h_o,
        "h_w_km": h_w,
        "gas_attenuation_dB": a_gas,
    })


# -----------------------------------------------------------------
# Cloud attenuation: ITU-R P.840-8 with an LWP proxy
# -----------------------------------------------------------------
def load_lwp_proxy():
    with open(config.LWP_PROXY_FILE, encoding="utf-8") as file:
        return json.load(file)


def cloud_attenuation(rh_percent, precip_mm, proxy):
    """A_cloud = LWP * K_l(f, 0 degC) / sin(el)."""
    rh_c = proxy["rh_c_percent"]
    s = np.clip((rh_percent - rh_c) / (100.0 - rh_c), 0.0, 1.0)
    lwp = proxy["a_kg_m2"] * s + proxy["b_kg_m2_per_mm"] * precip_mm

    k_l = float(value(itu840.specific_attenuation_coefficients(
        F, config.CLOUD_KL_TEMP_C)))
    return pd.DataFrame({
        "lwp_proxy_kg_m2": lwp,
        "cloud_kl_dB_km_per_g_m3": k_l,
        "cloud_attenuation_dB": lwp * k_l / SIN_EL,
    })


# -----------------------------------------------------------------
# Rain attenuation: ITU-R P.838-3 + P.618-13 steps 1-9, P.839-4 height
# -----------------------------------------------------------------
def rain_attenuation(rain_rate, lat, lon):
    """P.618 Sect. 2.2.1.1 steps 1-9 with the hourly rain rate R used in
    place of R_0.01 (step 10, the percentage scaling, is not applied)."""
    h_r = float(value(itu839.rain_height(lat, lon)))                  # step 1
    h_s = float(value(itu1511.topographic_altitude(lat, lon)))        # km
    dh = max(h_r - h_s, 0.0)
    l_s = dh / SIN_EL                                                 # step 2
    l_g = l_s * np.cos(np.deg2rad(EL))                                # step 3

    r = np.asarray(rain_rate, dtype=float)
    gamma = np.zeros_like(r)
    wet = r > 0
    if wet.any():                                                     # step 5
        gamma[wet] = value(itu838.rain_specific_attenuation(
            r[wet], F, EL, config.POLARIZATION_TILT_DEG))

    # step 6: horizontal reduction factor
    r001 = 1.0 / (1.0 + 0.78 * np.sqrt(l_g * gamma / F)
                  - 0.38 * (1.0 - np.exp(-2.0 * l_g)))

    # step 7: vertical adjustment factor
    zeta = np.rad2deg(np.arctan2(dh, l_g * r001))
    l_r = np.where(zeta > EL, l_g * r001 / np.cos(np.deg2rad(EL)), l_s)
    chi = 36.0 - abs(lat) if abs(lat) < 36.0 else 0.0
    nu = 1.0 / (1.0 + np.sqrt(SIN_EL) * (
        31.0 * (1.0 - np.exp(-EL / (1.0 + chi)))
        * np.sqrt(l_r * gamma) / F**2 - 0.45))

    l_e = l_r * nu                                                    # step 8
    a_rain = np.where(wet & (dh > 0), gamma * l_e, 0.0)              # step 9

    return pd.DataFrame({
        "rain_rate_mm_h": r,
        "rain_height_km": h_r,
        "station_height_km": h_s,
        "slant_path_km": l_s,
        "gamma_rain_dB_km": gamma,
        "r001": r001,
        "nu001": nu,
        "effective_path_km": np.where(wet, l_e, 0.0),
        "rain_attenuation_dB": a_rain,
    })


# -----------------------------------------------------------------
# Whole city
# -----------------------------------------------------------------
def compute_attenuation(weather, lat, lon, proxy):
    """Add every intermediate quantity and the EPL label to `weather`."""
    gas = gaseous_attenuation(weather["temperature_C"].to_numpy(),
                              weather["relative_humidity_percent"].to_numpy(),
                              weather["pressure_hPa"].to_numpy())
    cloud = cloud_attenuation(weather["relative_humidity_percent"].to_numpy(),
                              weather["precipitation_mm"].to_numpy(), proxy)
    # Hourly rain accumulation (mm) is used as the hourly-mean rate (mm/h).
    rain = rain_attenuation(weather["rain_mm"].to_numpy(), lat, lon)

    out = pd.concat([weather.reset_index(drop=True), gas, cloud, rain], axis=1)
    out[config.EPL_COLUMN] = (out["gas_attenuation_dB"]
                              + out["cloud_attenuation_dB"]
                              + out["rain_attenuation_dB"])
    out["elevation_deg"] = EL
    out["frequency_GHz"] = F
    return out


def load_city_weather(city, folder, years):
    frames = [pd.read_csv(folder / f"{city}_{year}.csv", parse_dates=["valid_time"])
              for year in years]
    df = pd.concat(frames, ignore_index=True)

    expected = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31 23:00", freq="h")
    if not df["valid_time"].reset_index(drop=True).equals(pd.Series(expected, name="valid_time")):
        raise ValueError(f"{city}: timestamps are not a complete hourly series.")
    if df.isna().any().any():
        raise ValueError(f"{city}: missing values in weather data.")
    return df


def check_physics(df, city):
    """Basic sanity checks on the labels."""
    parts = ["gas_attenuation_dB", "cloud_attenuation_dB", "rain_attenuation_dB"]
    values = df[parts + [config.EPL_COLUMN]].to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{city}: non-finite or negative attenuation.")
    if not np.allclose(df[parts].sum(axis=1), df[config.EPL_COLUMN]):
        raise ValueError(f"{city}: EPL is not the sum of its components.")
    if ((df["rain_mm"] > 0) != (df["rain_attenuation_dB"] > 0)).any():
        raise ValueError(f"{city}: rain attenuation without rain (or vice versa).")


def main():
    unseen = "--unseen" in sys.argv
    cities = UNSEEN_CITIES if unseen else CITIES
    folder = config.UNSEEN_WEATHER_DIR if unseen else config.WEATHER_DIR
    years = config.UNSEEN_YEARS if unseen else config.YEARS
    proxy = load_lwp_proxy()

    config.PHYSICS_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for city, coords in cities.items():
        weather = load_city_weather(city, folder, years)
        df = compute_attenuation(weather, coords["latitude"], coords["longitude"], proxy)
        check_physics(df, city)
        df.to_csv(config.PHYSICS_DIR / f"{city}_physics.csv", index=False)

        summary.append({
            "city": city,
            "hours": len(df),
            "gas_mean": df["gas_attenuation_dB"].mean(),
            "cloud_mean": df["cloud_attenuation_dB"].mean(),
            "rain_mean": df["rain_attenuation_dB"].mean(),
            "epl_mean": df[config.EPL_COLUMN].mean(),
            "epl_p99": df[config.EPL_COLUMN].quantile(0.99),
            "epl_max": df[config.EPL_COLUMN].max(),
            "rain_hours_pct": 100 * (df["rain_mm"] > 0).mean(),
            "rain_height_km": df["rain_height_km"].iloc[0],
            "station_height_km": df["station_height_km"].iloc[0],
        })
        print(f"{city}: {len(df)} hours, mean EPL {df[config.EPL_COLUMN].mean():.3f} dB")

    summary = pd.DataFrame(summary)
    name = "physics_summary_unseen.csv" if unseen else "physics_summary.csv"
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.RESULTS_DIR / name, index=False)
    print(summary.round(4).to_string(index=False))
    print(f"itur {itur.__version__}; P.676-{itu676.get_version()}, "
          f"P.840-{itu840.get_version()}, P.838-{itu838.get_version()}, "
          f"P.839-{itu839.get_version()}")


if __name__ == "__main__":
    main()
