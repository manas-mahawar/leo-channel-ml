"""Rain (and gas/cloud) label diagnostic on representative hours (read-only).

For dry, moderate-rain and heavy-rain hours in three cities, the intermediate
values stored by physics/build_attenuation.py are compared with:

  R1  an independent re-implementation of ITU-R P.618-13 Sect. 2.2.1.1 steps 1-9,
      written from the Recommendation (not copied from build_attenuation.py);
  R2  itur's own P.618 function, itur.models.itu618.rain_attenuation(..., R001=R);
  R3  the P.838-3 combination of horizontal/vertical coefficients for the
      elevation and polarisation tilt, applied by hand to itur's k_H, a_H, k_V, a_V;
  G   itur's P.676 approximate slant-path gas attenuation with dry-air pressure and
      an independent P.453 vapour pressure (with the enhancement factor, which
      our pipeline omits);
  C   P.840 K_l from an independent implementation of the double-Debye model.

It then quantifies what the paper's visible label range (max about 6.7 dB)
would require, for diagnosis only (nothing is changed):
  * an alternative rain-path model, A = gamma(R) * L_s (no P.618 reduction factors);
  * the rain rate at which our model reaches 6.7 dB.

Usage:
    python evaluation/rain_label_diagnostic.py
Outputs: results/replication_audit/rain_label_diagnostic.csv, rain_label_diagnostic.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import itur
from itur.models import itu618, itu676, itu838, itu839, itu840

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from data_pipeline.cities import CITIES

OUT = ROOT / "results" / "replication_audit"
F, EL, TAU = config.FREQUENCY_GHZ, config.ELEVATION_DEG, config.POLARIZATION_TILT_DEG
itu840.change_version(8)


def val(x):
    return float(np.asarray(getattr(x, "value", x)))


def p838_combined(f, el, tau):
    """P.838-3 Eqs. 4-5: combine H and V coefficients for elevation el, tilt tau."""
    k_h, a_h = (val(v) for v in itu838.rain_specific_attenuation_coefficients(f, 0.0, 0.0))
    k_v, a_v = (val(v) for v in itu838.rain_specific_attenuation_coefficients(f, 0.0, 90.0))
    c = np.cos(np.deg2rad(el)) ** 2 * np.cos(np.deg2rad(2 * tau))
    k = (k_h + k_v + (k_h - k_v) * c) / 2
    a = (k_h * a_h + k_v * a_v + (k_h * a_h - k_v * a_v) * c) / (2 * k)
    return {"k_H": k_h, "alpha_H": a_h, "k_V": k_v, "alpha_V": a_v, "k": k, "alpha": a}


def p618_reference(R, lat, h_r, h_s, k, alpha, f=F, el=EL):
    """P.618-13 Sect. 2.2.1.1 steps 2-9 with R in place of R_0.01 (el >= 5 deg)."""
    if R <= 0 or h_r <= h_s:
        return {"A": 0.0}
    th = np.deg2rad(el)
    L_s = (h_r - h_s) / np.sin(th)                                          # step 2
    L_G = L_s * np.cos(th)                                                  # step 3
    g = k * R ** alpha                                                      # step 5
    r = 1 / (1 + 0.78 * np.sqrt(L_G * g / f) - 0.38 * (1 - np.exp(-2 * L_G)))   # step 6
    zeta = np.degrees(np.arctan((h_r - h_s) / (L_G * r)))                   # step 7
    L_R = L_G * r / np.cos(th) if zeta > el else (h_r - h_s) / np.sin(th)
    chi = 36 - abs(lat) if abs(lat) < 36 else 0.0
    nu = 1 / (1 + np.sqrt(np.sin(th)) * (31 * (1 - np.exp(-(el / (1 + chi)))) * np.sqrt(L_R * g) / f ** 2 - 0.45))
    L_E = L_R * nu                                                          # step 8
    return {"L_s": L_s, "L_G": L_G, "gamma": g, "r001": r, "zeta": zeta, "L_R": L_R, "nu": nu,
            "L_E": L_E, "A": g * L_E}                                       # step 9


def vapour_pressure_p453(t_c, rh, p_total):
    """ITU-R P.453 water vapour pressure (hPa) over water, with the enhancement factor EF."""
    ef = 1 + 1e-4 * (7.2 + p_total * (0.0320 + 5.9e-6 * t_c ** 2))
    return rh / 100 * ef * 6.1121 * np.exp((18.678 - t_c / 234.5) * t_c / (t_c + 257.14))


def kl_double_debye(f, t_c):
    """P.840 Sect. 2 (Eqs. 2-9): K_l in (dB/km)/(g/m^3)."""
    theta = 300.0 / (t_c + 273.15)
    e0 = 77.66 + 103.3 * (theta - 1)
    e1 = 0.0671 * e0
    e2 = 3.52
    fp = 20.20 - 146 * (theta - 1) + 316 * (theta - 1) ** 2
    fs = 39.8 * fp
    e_im = f * (e0 - e1) / (fp * (1 + (f / fp) ** 2)) + f * (e1 - e2) / (fs * (1 + (f / fs) ** 2))
    e_re = (e0 - e1) / (1 + (f / fp) ** 2) + (e1 - e2) / (1 + (f / fs) ** 2) + e2
    eta = (2 + e_re) / e_im
    return 0.819 * f / (e_im * (1 + eta ** 2))


def representative_hours():
    rows = []
    for city in ("Athens", "Bergen", "Vigo"):
        p = pd.read_csv(config.PHYSICS_DIR / f"{city}_physics.csv", parse_dates=["valid_time"])
        dry = p[(p.rain_mm == 0) & (p.precipitation_mm == 0)].iloc[0]
        moderate = p.iloc[(p.rain_mm - 1.0).abs().idxmin()]
        heavy = p.loc[p.rain_mm.idxmax()]
        for label, r in (("dry", dry), ("moderate (about 1 mm/h)", moderate), ("heavy (city max)", heavy)):
            rows.append((city, label, r))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    coef = p838_combined(F, EL, TAU)
    kl_ref = kl_double_debye(F, config.CLOUD_KL_TEMP_C)
    rows = []
    for city, label, r in representative_hours():
        lat, lon = CITIES[city]["latitude"], CITIES[city]["longitude"]
        R = float(r.rain_mm)
        ref = p618_reference(R, lat, float(r.rain_height_km), float(r.station_height_km), coef["k"], coef["alpha"])
        itur_a = val(itu618.rain_attenuation(lat, lon, F, EL, hs=float(r.station_height_km), p=0.01,
                                             R001=R, tau=TAU)) if R > 0 else 0.0
        # P.676-12 Annex 1 uses dry-air pressure p = P_total - e (itur adds e back for the heights)
        t_c, rh, p_tot = float(r.temperature_C), float(r.relative_humidity_percent), float(r.pressure_hPa)
        e = vapour_pressure_p453(t_c, rh, p_tot)
        rho = 216.7 * e / (t_c + 273.15)
        gas_ref = val(itu676.gaseous_attenuation_slant_path(F, EL, rho, p_tot - e, t_c + 273.15, mode="approx"))
        gas_total_p = val(itu676.gaseous_attenuation_slant_path(F, EL, float(r.rho_g_m3), p_tot,
                                                                t_c + 273.15, mode="approx"))
        rows.append({
            "city": city, "case": label, "valid_time": str(r.valid_time), "rain_mm_h": R,
            "precip_mm": float(r.precipitation_mm), "T_C": float(r.temperature_C), "RH_pct": float(r.relative_humidity_percent),
            "P_hPa": float(r.pressure_hPa), "h_R_km": float(r.rain_height_km),
            "h_R_itur_P839": val(itu839.rain_height(lat, lon)), "h_s_km": float(r.station_height_km),
            "L_s_km": float(r.slant_path_km), "L_s_ref": ref.get("L_s", np.nan),
            "gamma_dB_km": float(r.gamma_rain_dB_km), "gamma_ref_kR^a": ref.get("gamma", 0.0),
            "r001": float(r.r001), "r001_ref": ref.get("r001", np.nan),
            "nu001": float(r.nu001), "nu001_ref": ref.get("nu", np.nan),
            "L_E_km": float(r.effective_path_km), "L_E_ref": ref.get("L_E", 0.0),
            "A_rain_dB": float(r.rain_attenuation_dB), "A_rain_ref_R1": ref["A"], "A_rain_itur_R2": itur_a,
            "A_rain_gamma_x_Ls_alt": ref.get("gamma", 0.0) * ref.get("L_s", 0.0),
            "A_gas_dB": float(r.gas_attenuation_dB), "A_gas_itur_approx": gas_ref,
            "A_gas_itur_if_total_P_were_passed": gas_total_p,
            "LWP_kg_m2": float(r.lwp_proxy_kg_m2), "K_l": float(r.cloud_kl_dB_km_per_g_m3), "K_l_ref": kl_ref,
            "A_cloud_dB": float(r.cloud_attenuation_dB),
            "A_cloud_ref": float(r.lwp_proxy_kg_m2) * kl_ref / np.sin(np.deg2rad(EL)),
            "EPL_dB": float(r.epl_dB),
        })
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "rain_label_diagnostic.csv", index=False)

    # Rain-path alternative and the rate needed for 6.7 dB (all hours, six cities)
    alt, needed = {}, {}
    rates = np.linspace(0.05, 12, 2000)
    for city, c in CITIES.items():
        p = pd.read_csv(config.PHYSICS_DIR / f"{city}_physics.csv")
        alt[city] = {"ours_max_dB": float(p.rain_attenuation_dB.max()),
                     "gamma_x_Ls_max_dB": float((p.gamma_rain_dB_km * p.slant_path_km).max()),
                     "ours_mean_rainy_hours_dB": float(p.rain_attenuation_dB[p.rain_mm > 0].mean()),
                     "gamma_x_Ls_mean_rainy_hours_dB": float((p.gamma_rain_dB_km * p.slant_path_km)[p.rain_mm > 0].mean())}
        h_r, h_s = float(p.rain_height_km.iloc[0]), float(p.station_height_km.iloc[0])
        gas_cloud_max = float((p.gas_attenuation_dB + p.cloud_attenuation_dB).max())
        a = np.array([p618_reference(R, c["latitude"], h_r, h_s, coef["k"], coef["alpha"])["A"] for R in rates])
        target = 6.7 - gas_cloud_max                    # rain share of a 6.7 dB EPL in the worst gas/cloud hour
        needed[city] = {"rain_rate_for_A_rain=6.7-max(gas+cloud)_mm_h": float(rates[np.argmax(a >= target)]),
                        "observed_max_rain_mm_h": float(p.rain_mm.max()),
                        "hours_with_rain_above_that_rate": int((p.rain_mm > rates[np.argmax(a >= target)]).sum())}

    diffs = {
        "max_abs_rain_ours_vs_R1_dB": float((table.A_rain_dB - table.A_rain_ref_R1).abs().max()),
        "max_abs_rain_ours_vs_itur_R2_dB": float((table.A_rain_dB - table.A_rain_itur_R2).abs().max()),
        "max_abs_gamma_ours_vs_kR^a": float((table.gamma_dB_km - table["gamma_ref_kR^a"]).abs().max()),
        "max_abs_gas_ours_vs_itur_approx_dry_P_with_P453_EF_dB": float((table.A_gas_dB - table.A_gas_itur_approx).abs().max()),
        "max_abs_gas_effect_of_passing_total_instead_of_dry_P_dB":
            float((table.A_gas_dB - table.A_gas_itur_if_total_P_were_passed).abs().max()),
        "max_abs_cloud_ours_vs_double_debye_dB": float((table.A_cloud_dB - table.A_cloud_ref).abs().max()),
        "K_l_ours": float(table.K_l.iloc[0]), "K_l_double_debye": kl_ref,
        "max_abs_rain_height_vs_P839": float((table.h_R_km - table.h_R_itur_P839).abs().max()),
    }
    summary = {"p838_coefficients_39GHz": coef, "differences": diffs,
               "rain_path_alternative": alt, "rain_rate_needed_for_paper_max": needed}
    with open(OUT / "rain_label_diagnostic.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    pd.set_option("display.width", 250)
    cols = ["city", "case", "rain_mm_h", "h_R_km", "h_s_km", "L_s_km", "gamma_dB_km", "r001", "nu001", "L_E_km",
            "A_rain_dB", "A_rain_ref_R1", "A_rain_itur_R2", "A_rain_gamma_x_Ls_alt", "A_gas_dB", "A_gas_itur_approx",
            "A_cloud_dB", "A_cloud_ref", "EPL_dB"]
    print(table[cols].round(4).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
