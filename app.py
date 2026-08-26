"""
Exoplanet Detector - web frontend
=================================
A one-page web app: enter a TESS star ID, and it detects any transit signal,
classifies it (planet / eclipsing binary / noise), shows the phase-folded light
curve, and explains the decision in plain English.

Setup:
    pip install streamlit lightkurve wotan transitleastsquares scikit-learn shap joblib matplotlib
    # make sure model.pkl (saved from your notebook) is in this folder
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import joblib

# ---------------------------------------------------------------------------
# Pipeline functions (same logic as the notebook, self-contained here)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    bundle = joblib.load("model.pkl")
    return bundle["clf"], bundle["cols"]


def load_light_curve(target):
    import lightkurve as lk
    search = lk.search_lightcurve(target, mission="TESS", author="SPOC", exposure_time=120)
    lc = search[0].download().remove_nans().normalize()
    return lc.time.value, lc.flux.value


def detrend(time, flux, window_length=0.5):
    from wotan import flatten
    flat, trend = flatten(time, flux, window_length=window_length,
                          method="biweight", return_trend=True)
    g = np.isfinite(flat)
    return time[g], flat[g], trend[g]


def detect(time, flat_flux):
    from transitleastsquares import transitleastsquares
    return transitleastsquares(time, flat_flux).power(period_min=0.5, period_max=10.0)


def extract_features(time, flux, period, t0, duration):
    dur_phase = duration / period; half = dur_phase / 2.0
    phase = ((time - t0 + 0.5 * period) % period) / period - 0.5
    out = np.abs(phase) > 3 * half
    base = np.median(flux[out]) if out.sum() else 1.0
    inp = np.abs(phase) < half
    pd_ = max(base - np.median(flux[inp]), 1e-9)
    ins = np.abs(np.abs(phase) - 0.5) < half
    sd = base - np.median(flux[ins]) if ins.sum() else 0.0
    tn = np.round((time - t0) / period).astype(int)
    om = inp & (tn % 2 == 1); em = inp & (tn % 2 == 0)
    do = base - np.median(flux[om]) if om.sum() else pd_
    de = base - np.median(flux[em]) if em.sum() else pd_
    oe = abs(do - de) / pd_
    core = np.abs(phase) < half * 0.3; wing = (np.abs(phase) > half * 0.6) & inp
    if core.sum() and wing.sum():
        vs = 1.0 - ((base - np.median(flux[wing])) / max(base - np.median(flux[core]), 1e-9))
    else:
        vs = 0.0
    return {"primary_depth_ppm": pd_ * 1e6, "secondary_depth_ppm": sd * 1e6,
            "secondary_ratio": sd / pd_, "odd_even_diff": oe, "v_shape": vs}


_MEANING = {"primary_depth_ppm": "transit depth", "secondary_depth_ppm": "a secondary eclipse",
            "secondary_ratio": "presence of a secondary eclipse",
            "odd_even_diff": "difference between alternating transits",
            "v_shape": "a V-shaped (grazing) profile", "sde": "detection significance",
            "snr": "signal-to-noise"}


def explain(clf, columns, row):
    import shap
    pred = clf.predict(row)[0]
    classes = list(clf.classes_)
    base = clf.calibrated_classifiers_[0].estimator if hasattr(clf, "calibrated_classifiers_") else clf
    sv = shap.TreeExplainer(base).shap_values(row)
    idx = classes.index(pred)
    contrib = sv[..., idx][0] if np.ndim(sv) == 3 else sv[idx][0]
    ranked = sorted(zip(columns, contrib), key=lambda t: -t[1])
    reasons = [_MEANING.get(f, f) for f, c in ranked[:2] if c > 0]
    return " and ".join(reasons) if reasons else "the overall signal pattern"


@st.cache_data(show_spinner=False)
def analyze(target):
    """Full pipeline for one star. Cached so repeat lookups are instant."""
    time, flux = load_light_curve(target)
    time_c, flat, trend = detrend(time, flux)
    r = detect(time_c, flat)
    feats = extract_features(time_c, flat, r.period, r.T0, r.duration)
    feats["sde"] = float(r.SDE); feats["snr"] = float(r.snr)
    return {
        "period": float(r.period), "depth_ppm": (1 - r.depth) * 1e6,
        "duration_hr": float(r.duration) * 24, "sde": float(r.SDE), "snr": float(r.snr),
        "feats": feats,
        "folded_phase": np.asarray(r.folded_phase), "folded_y": np.asarray(r.folded_y),
        "model_phase": np.asarray(r.model_folded_phase), "model_y": np.asarray(r.model_folded_model),
    }


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Exoplanet Detector", page_icon="🪐", layout="centered")
st.title("🪐 Exoplanet Transit Detector")
st.write("Enter a TESS star ID. The pipeline detrends the light curve, searches for "
         "transit signals, classifies what it finds, and explains why.")

clf, cols = load_model()

target = st.text_input("TESS star ID", value="TIC 307210830",
                       help="e.g. TIC 307210830 (a confirmed planet host)")

if st.button("Analyze", type="primary"):
    try:
        with st.spinner(f"Downloading and analyzing {target}… (this can take ~30–60s)"):
            res = analyze(target)

        # classify
        Xr = pd.DataFrame([res["feats"]])[cols].fillna(0)
        pred = clf.predict(Xr)[0]
        conf = float(clf.predict_proba(Xr)[0].max())
        reason = explain(clf, cols, Xr)

        # verdict
        nice = {"planet": "🌍 Planet candidate", "eclipsing_binary": "⭐⭐ Eclipsing binary",
                "noise": "🌫️ Noise / no clear transit"}.get(pred, pred)
        if res["sde"] < 7:
            st.warning("No statistically significant transit found (low detection significance).")
        st.subheader(nice)
        st.caption(f"Classified mainly due to {reason}.")
        st.progress(conf, text=f"Confidence: {conf:.0%}")

        # parameters
        c1, c2, c3 = st.columns(3)
        c1.metric("Period", f"{res['period']:.3f} d")
        c2.metric("Depth", f"{res['depth_ppm']:.0f} ppm")
        c3.metric("Duration", f"{res['duration_hr']:.2f} h")
        c1.metric("Significance (SDE)", f"{res['sde']:.1f}")
        c2.metric("SNR", f"{res['snr']:.1f}")

        # phase-folded plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(res["folded_phase"], res["folded_y"], ".", ms=2, alpha=0.4, label="data")
        ax.plot(res["model_phase"], res["model_y"], "-", lw=2, label="transit model")
        ax.set_xlabel("Phase"); ax.set_ylabel("Normalized flux")
        ax.set_title(f"Phase-folded on P = {res['period']:.4f} d")
        ax.legend(loc="lower left")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Could not analyze {target}: {e}\n\nCheck the ID is a valid TESS target "
                 "with 2-minute data (try TIC 307210830).")

st.divider()
st.caption("Built with lightkurve · wotan · Transit Least Squares · scikit-learn · SHAP")