import numpy as np
import matplotlib.pyplot as plt

def plot_mt(df):
    f = df["frequency"].values
    rho_xy = df["rho_xy"].values
    rho_yx = df["rho_yx"].values
    phi_xy = df["phase_xy_deg"].values
    phi_yx = df["phase_yx_deg"].values

    idx = np.argsort(f)
    f = f[idx]
    rho_xy = rho_xy[idx]
    rho_yx = rho_yx[idx]
    phi_xy = phi_xy[idx]
    phi_yx = phi_yx[idx]

    plt.figure(figsize=(10,8))

    ax1 = plt.subplot(2,1,1)
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.invert_xaxis()

    ax1.plot(f, rho_xy, 'o-', label="ρxy")
    ax1.plot(f, rho_yx, 'o-', label="ρyx")
    ax1.grid(True, which="both", ls="--")
    ax1.set_ylabel("ρ (Ω·m)")
    ax1.legend()

    ax2 = plt.subplot(2,1,2, sharex=ax1)
    ax2.set_xscale("log"); ax2.invert_xaxis()
    ax2.plot(f, phi_xy, 's-', label="ϕxy")
    ax2.plot(f, phi_yx, 's-', label="ϕyx")
    ax2.set_ylim(0, 45)
    ax2.grid(True, which="both", ls="--")
    ax2.set_ylabel("phase (°)")
    ax2.set_xlabel("frequency (Hz)")
    ax2.legend()

    plt.tight_layout()
    plt.show()
