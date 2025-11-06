import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def smooth_mt_df(df, mode="medium"):

    levels = {
        "sharp": 40,
        "medium": 100,
        "smooth": 200
    }

    n = levels.get(mode, 100)

    f_raw = df["frequency"].values
    idx = np.argsort(f_raw)
    f_raw = f_raw[idx]

    f_new = np.logspace(np.log10(f_raw.min()), np.log10(f_raw.max()), n)

    def interp_log(y):
        fn = interp1d(np.log10(f_raw), np.log10(y[idx]), kind="cubic")
        return 10**fn(np.log10(f_new))

    def interp_lin(y):
        fn = interp1d(np.log10(f_raw), y[idx], kind="cubic")
        return fn(np.log10(f_new))

    df_s = pd.DataFrame({
        "frequency": f_new,
        "rho_xy": interp_log(df["rho_xy"].values),
        "rho_yx": interp_log(df["rho_yx"].values),
        "phase_xy_deg": interp_lin(df["phase_xy_deg"].values),
        "phase_yx_deg": interp_lin(df["phase_yx_deg"].values),
    })

    return df_s
