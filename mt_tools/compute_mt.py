import numpy as np
import pandas as pd

MU0 = 4 * np.pi * 1e-7

def compute_rho_phase(Z, freq):
    omega = 2 * np.pi * freq
    rho = (np.abs(Z)**2) / (MU0 * omega)
    phase = np.degrees(np.arctan2(Z.imag, Z.real))
    return rho, phase


def edi_to_dataframe(edi_path):
    from .edi_parser import EDIParser
    parser = EDIParser(edi_path)
    data = parser.parse()

    freq = data["freq"]
    Zxy = data["Zxy"]
    Zyx = data["Zyx"]

    rho_xy, ph_xy = compute_rho_phase(Zxy, freq)
    rho_yx, ph_yx = compute_rho_phase(Zyx, freq)

    df = pd.DataFrame({
        "frequency": freq,
        "rho_xy": rho_xy,
        "rho_yx": rho_yx,
        "phase_xy_deg": ph_xy,
        "phase_yx_deg": ph_yx,
    })

    return df
