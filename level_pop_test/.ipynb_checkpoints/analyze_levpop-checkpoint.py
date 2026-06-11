import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

dat = np.genfromtxt('/home/jinlim/RT_JIN/level_pop_test/CLOUDY_QSO_popHe+_pops.dat', skip_header=1)
depth = dat[:, 0]
lev   = dat[:, 1:]

phy = np.genfromtxt('/home/jinlim/RT_JIN/level_pop_test/CLOUDY_QSO_pop.phy', skip_header=1)
Te_phy = phy[:, 1]
ne_phy = phy[:, 3]

rad = np.genfromtxt('/home/jinlim/RT_JIN/level_pop_test/CLOUDY_QSO_pop.rad', skip_header=1)
radius = rad[:, 1]

nz = min(len(depth), len(radius), len(Te_phy))
depth  = depth[:nz]
lev    = lev[:nz]
radius = radius[:nz]
Te = Te_phy[:nz]
ne = ne_phy[:nz]

kpc   = 3.086e21
r_kpc = radius / kpc

def start_idx(n):
    return n * (n - 1) // 2

def n_total(lev_arr, n):
    s = start_idx(n)
    return lev_arr[:, s:s+n].sum(axis=1)

n_max_plot = 10
n_pops = {n: n_total(lev, n) for n in range(1, n_max_plot + 1)}

print("=== He II level populations (peak) ===")
for n in range(1, n_max_plot + 1):
    print("  n={:2d}: {:.3e} cm^-3".format(n, n_pops[n].max()))

s3 = start_idx(3)
n3s = lev[:, s3]
n3p = lev[:, s3 + 1]
n3d = lev[:, s3 + 2]
print("\nn=3 sublevel peaks:")
print("  3s: {:.3e}, 3p: {:.3e}, 3d: {:.3e}".format(n3s.max(), n3p.max(), n3d.max()))

hnu_1640 = 6.626e-27 * 2.998e10 / (1640.4e-8)
A_3d_2p  = 1.033e9
A_3p_2s  = 3.600e8
A_3s_2p  = 2.670e8

emis_pop = (n3d * A_3d_2p + n3p * A_3p_2s + n3s * A_3s_2p) * hnu_1640

with open('/home/jinlim/RT_JIN/level_pop_test/CLOUDY_QSO_pop.emis') as f:
    hdr = f.readline().strip()
print("\n.emis header (partial):", hdr[:400])

emis_raw = np.genfromtxt('/home/jinlim/RT_JIN/level_pop_test/CLOUDY_QSO_pop.emis', skip_header=1)
if emis_raw.ndim == 2:
    emis_raw = emis_raw[:nz]
    emis_cloudy_1640 = emis_raw[:, 10]
    mask = emis_cloudy_1640 > 0
    print("\nCloudy emis(1640) peak  = {:.3e} erg/s/cm3".format(emis_cloudy_1640.max()))
    print("Level-pop emis peak     = {:.3e} erg/s/cm3".format(emis_pop.max()))
    ratio = emis_pop[mask] / emis_cloudy_1640[mask]
    print("Ratio levpop/cloudy at peak zone: {:.3f}".format(emis_pop.argmax()))

# === PLOT 1: level populations n=1..10 vs radius ===
fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

colors = plt.cm.tab10(np.linspace(0, 1, n_max_plot))
ax = axes[0]
for n in range(1, n_max_plot + 1):
    valid = n_pops[n] > 0
    if valid.sum() > 1:
        ax.semilogy(r_kpc[valid], n_pops[n][valid], color=colors[n-1],
                    lw=1.5, label="n={}".format(n))
ax.set_ylabel("n(He II, level n)  [cm$^{-3}$]", fontsize=12)
ax.set_title("He II (He$^+$) Level Populations  [Cloudy zone-by-zone]", fontsize=13)
ax.legend(fontsize=9, ncol=2, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_ylim(1e-30, 1e-2)

# n=3 sublevels (1640A upper state)
ax = axes[1]
s2 = start_idx(2)
n2s = lev[:, s2];  n2p = lev[:, s2+1]
for arr, lbl, ls in [(n2s, "n=2, 2s (lower)", '--'),
                      (n2p, "n=2, 2p (lower)", '--'),
                      (n3s, "n=3, 3s", '-'),
                      (n3p, "n=3, 3p", '-'),
                      (n3d, "n=3, 3d (dominant)", '-')]:
    valid = arr > 0
    if valid.sum() > 1:
        ax.semilogy(r_kpc[valid], arr[valid], lw=1.5, ls=ls, label=lbl)
ax.set_ylabel("n(He II, sublevel)  [cm$^{-3}$]", fontsize=12)
ax.set_xlabel("Radius [kpc]", fontsize=12)
ax.set_title("He II 1640$\\AA$ : n=3 (upper) & n=2 (lower) sublevels", fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/jinlim/RT_JIN/level_pop_test/HeII_level_pops.png', dpi=150, bbox_inches='tight')
print("\nSaved: HeII_level_pops.png")

# === PLOT 2: emissivity from level pops vs Cloudy ===
if emis_raw.ndim == 2:
    fig2, axes2 = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    fig2.suptitle("He II 1640$\\AA$ Emissivity: Level Populations vs Cloudy", fontsize=13)

    ax = axes2[0]
    ax.semilogy(r_kpc, emis_cloudy_1640, 'k-', lw=2, label='Cloudy (direct)')
    valid_p = emis_pop > 0
    ax.semilogy(r_kpc[valid_p], emis_pop[valid_p], 'r--', lw=1.5,
                label='From level pops (n=3 x A x hnu)')
    ax.set_ylabel("$\\varepsilon_{1640}$ [erg/s/cm$^3$]", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes2[1]
    ratio_arr = np.where(emis_cloudy_1640 > 0, emis_pop / emis_cloudy_1640, np.nan)
    ax.plot(r_kpc, ratio_arr, 'b-', lw=1.5)
    ax.axhline(1.0, color='k', ls='--', alpha=0.5)
    ax.set_ylim(0, 3)
    ax.set_ylabel("LevPop / Cloudy", fontsize=12)
    ax.grid(True, alpha=0.3)

    ax = axes2[2]
    ax_te = ax.twinx()
    l1, = ax.semilogy(r_kpc, ne, 'b-', lw=1.5, label='$n_e$')
    l2, = ax_te.semilogy(r_kpc, Te, 'r-', lw=1.5, label='$T_e$')
    ax.set_ylabel("$n_e$ [cm$^{-3}$]", fontsize=12, color='b')
    ax_te.set_ylabel("$T_e$ [K]", fontsize=12, color='r')
    ax.set_xlabel("Radius [kpc]", fontsize=12)
    ax.legend(handles=[l1, l2], fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/jinlim/RT_JIN/level_pop_test/HeII_emis_levpop_vs_cloudy.png',
                dpi=150, bbox_inches='tight')
    print("Saved: HeII_emis_levpop_vs_cloudy.png")
