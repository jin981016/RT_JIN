"""
RT_v1.py — numba-accelerated & vectorised version of RT.py
Run with /opt/anaconda3/bin/python (numba 0.59 + Python 3.11).
See RT_v1_changes.txt for a full description of every change.
"""

import numpy as np
import matplotlib.pyplot as plt
import pyCloudy as pc
import pyneb as pn
from astropy    import constants as const
from astropy.io import ascii
import pandas as pd
import warnings
from numpy import log10, exp
import os
from astropy.constants import h, c
import astropy.units as u
from scipy.interpolate import interp1d
from scipy.integrate import simpson
import matplotlib.gridspec as gridspec

# ── Numba (graceful fallback if not available) ────────────────────────────────
try:
    from numba import njit, prange
    _NUMBA = True
except ImportError:
    _NUMBA = False
    def njit(*a, **kw):          # dummy decorator
        def _d(f): return f
        return _d(a[0]) if (len(a) == 1 and callable(a[0])) else _d
    prange = range

warnings.filterwarnings('ignore')

# ── Physical constants ────────────────────────────────────────────────────────
kpc      = const.kpc.cgs.value
h_ev     = h.to(u.eV * u.s).value
c_cms    = c.to(u.cm / u.s).value
c_kms    = c.to(u.km / u.s).value

Ly_a_K   = 1215.673644609e-8
Ly_a_H   = 1215.668237310e-8
C_IV_K   = 1548.187e-8
C_IV_H   = 1550.772e-8
C_IV_K_A = 1548.187
C_IV_H_A = 1550.772

nu_Lya_K  = c_cms / Ly_a_K
nu_Lya_H  = c_cms / Ly_a_H
nu_CIV_K  = c_cms / C_IV_K
nu_CIV_H  = c_cms / C_IV_H

Ryd_Lya_K = h_ev * nu_Lya_K / 13.6
Ryd_Lya_H = h_ev * nu_Lya_H / 13.6
Ryd_CIV_K = h_ev * nu_CIV_K / 13.6
Ryd_CIV_H = h_ev * nu_CIV_H / 13.6

_KPC_CM = float(const.kpc.cgs.value)   # plain float for numba kernels


# =====================================================
# Numba kernels  (pure float64 arrays, no Python objects)
# =====================================================

if _NUMBA:
    @njit(parallel=True, cache=True)
    def _abel_kernel(Project_R, radius_kpc, emissivity, z, n_pts):
        """
        Abel projection via the substitution  r = sqrt(R^2 + t^2).

        The standard integrand  emis(r)*r/sqrt(r^2-R^2) dr  becomes
        emis(sqrt(R^2+t^2)) dt  after the substitution — the 1/sqrt
        singularity at r=R cancels exactly, so a uniform-grid trapezoid
        rule converges cleanly.  The outer loop over projected radii is
        parallelised with numba.prange.
        """
        N     = len(Project_R)
        SB    = np.zeros(N)
        r_max = radius_kpc[-1]
        fac   = 2.0 / (1.0 + z) ** 4

        for ii in prange(N):
            R       = Project_R[ii]
            t2_max  = r_max * r_max - R * R
            if t2_max <= 0.0:
                continue
            t_max = np.sqrt(t2_max)
            dt    = t_max / (n_pts - 1)

            s = 0.0
            for jj in range(n_pts):
                t  = jj * dt
                r  = np.sqrt(R * R + t * t)
                em = np.interp(r, radius_kpc, emissivity)
                # trapezoidal weights
                w  = 0.5 if (jj == 0 or jj == n_pts - 1) else 1.0
                s += w * em

            SB[ii] = fac * dt * s

        return SB

    @njit(cache=True)
    def _photon_number_kernel(R_rt, origin_SB, CIV_lum, kpc_val):
        """Inner computation for photon_number_SB, JIT-compiled."""
        n   = len(R_rt)
        dR  = R_rt[1] - R_rt[0]

        areas = np.empty(n)
        for ii in range(n):
            R = R_rt[ii]
            if R == 0.0:
                areas[ii] = np.pi * (0.5 * dR) ** 2
            elif ii == n - 1:
                areas[ii] = np.pi * (2.0 * R + 0.5 * dR) * 0.5 * dR
            else:
                areas[ii] = 2.0 * np.pi * R * dR

        num_dis      = origin_SB * areas
        total        = num_dis.sum()
        factor_atom  = CIV_lum / total
        factor_area  = (100.0 * kpc_val) ** 2

        sb = num_dis * factor_atom / (areas * factor_area)
        return num_dis, sb

else:
    # ── Pure-numpy fallbacks (no numba) ──────────────────────────────────────

    def _abel_kernel(Project_R, radius_kpc, emissivity, z, n_pts):
        """Vectorised numpy fallback for the Abel transform."""
        N     = len(Project_R)
        SB    = np.zeros(N)
        r_max = radius_kpc[-1]
        fac   = 2.0 / (1.0 + z) ** 4

        for ii in range(N):
            R      = Project_R[ii]
            t2_max = r_max ** 2 - R ** 2
            if t2_max <= 0.0:
                continue
            t_arr  = np.linspace(0.0, np.sqrt(t2_max), n_pts)
            r_arr  = np.sqrt(R ** 2 + t_arr ** 2)
            em     = np.interp(r_arr, radius_kpc, emissivity)
            SB[ii] = fac * np.trapz(em, t_arr)

        return SB

    def _photon_number_kernel(R_rt, origin_SB, CIV_lum, kpc_val):
        """Vectorised numpy fallback for photon_number_SB."""
        n   = len(R_rt)
        dR  = R_rt[1] - R_rt[0]

        areas       = 2.0 * np.pi * R_rt * dR
        areas[0]    = np.pi * (0.5 * dR) ** 2
        areas[-1]   = np.pi * (2.0 * R_rt[-1] + 0.5 * dR) * 0.5 * dR

        num_dis     = origin_SB * areas
        factor_atom = CIV_lum / num_dis.sum()
        factor_area = (100.0 * kpc_val) ** 2

        sb = num_dis * factor_atom / (areas * factor_area)
        return num_dis, sb


# =====================================================
# 공통 헬퍼 함수
# =====================================================

def calculate_order_and_value(value):
    """주어진 값에 대해 변환된 값과 해당 order 반환"""
    if value == 0:
        return "000", 0
    elif value == 1:
        return int(value * 100), 0
    elif value < 100:
        return int(value * 10), 1
    elif value < 1000:
        return int(value), 2
    return int(value / 10), 3


def _infer_multi_factor_from_column_density_order(Column_density_order):
    try:
        val = float(Column_density_order)
    except Exception:
        raise ValueError(
            "Column_density_order must be convertible to float (e.g. 20.0 or 20.5)")

    frac = abs(val - np.floor(val))
    if np.isclose(frac, 0.0, atol=1e-6):
        return 1.0
    if np.isclose(frac, 0.5, atol=1e-6):
        return 3.2
    raise ValueError("Only xx.0 or xx.5 are supported for Column_density_order")


def resolve_column_density_path(Lumin, metals, Column_density_order,
                                sub_path='CIV/CLOUDY_QSO'):
    cl_order  = _infer_multi_factor_from_column_density_order(Column_density_order)
    _val      = float(Column_density_order)
    _frac     = abs(_val - np.floor(_val))
    _dir      = np.floor(_val) if np.isclose(_frac, 0.5, atol=1e-6) else _val
    dir_str   = f"{_dir:.1f}"
    return os.path.expanduser(
        f'~/CIV_RT_scat_data/CLOUDY_setup/'
        f'Lum_{Lumin}_2/metal_{metals}/N_H_{cl_order}_{dir_str}/{sub_path}'
    )


# =====================================================
# Path_RT Part — Spectrum
# =====================================================

def Spectrum_path(path_RT, Line, atom_num, atom_index, V_out, V_emit, V_rand):
    out,  out_order  = calculate_order_and_value(V_out)
    emit, emit_order = calculate_order_and_value(V_emit)
    ran,  ran_order  = calculate_order_and_value(V_rand)
    return (
        r'{}/N_atom{}0E+{}_Vexp{}E+0{}_Vemit{}E+0{}_tauD000E+00_Vran{}E+0{}'
        .format(path_RT, atom_num, atom_index,
                out, out_order, emit, emit_order, ran, ran_order)
    )


def Spectrum_K_or_H(path_RT, line, atom_num, atom_index, V_out, V_emit, V_rand):
    path_file = Spectrum_path(
        path_RT, line, atom_num, atom_index, V_out, V_emit, V_rand) + 'spec.dat'
    try:
        data = pd.read_csv(path_file, sep=r'\s+', header=None)
        print("v_rand = 11.8 km/s")
    except Exception:
        print('파일을 찾을 수 없습니다.', path_file)
        return None, None, None

    x = data[0].to_numpy()
    try:
        if line in ('h', 'H'):
            return x, data[2].to_numpy(), data[4].to_numpy()
        elif line in ('k', 'K'):
            return x, data[1].to_numpy(), data[3].to_numpy()
        else:
            raise ValueError('invalid line')
    except Exception:
        print('line을 입력하세요 K or H')
        return None, None, None


def f_esc(path_RT, line, atom_num, atom_index, V_out, V_emit, V_rand):
    path_file = Spectrum_path(
        path_RT, line, atom_num, atom_index, V_out, V_emit, V_rand) + '_f_esc.dat'
    try:
        data = pd.read_csv(path_file, sep=r'\s+', header=None)
    except Exception:
        print('파일을 찾을 수 없습니다.', path_file)
        return None

    row = data.iloc[0]
    return {
        'total_esc':  row[0],  'ratio_k_h': row[1],
        'ratio_k':    row[2],  'ratio_h':   row[3],
        'ns_k':       row[4],  'ns_h':      row[5],
        'ns_dust_k':  row[6],  'ns_dust_h': row[7],
        'path_k':     row[8],  'path_h':    row[9],
        'dir_esc_k':  row[10], 'dir_esc_h': row[11],
        'ns_clump_k': row[12], 'ns_clump_h':row[13],
    }


def Spectrum_All(path_RT, line, atom_num, atom_index, V_out, V_emit, V_rand):
    path_file = Spectrum_path(
        path_RT, line, atom_num, atom_index, V_out, V_emit, V_rand) + 'spec_com.dat'
    try:
        data = pd.read_csv(path_file, sep=r'\s+', header=None)
    except Exception:
        print('파일을 찾을 수 없습니다.', path_file)
        return None, None, None
    return data[0].to_numpy(), data[1].to_numpy(), data[2].to_numpy()


def Spectrum_RT(path_RT, line, atom_num, atom_index, V_out, V_emit, V_rand):
    lam, spec_tot, spec_halo = Spectrum_All(
        path_RT, line, atom_num, atom_index, V_out, V_emit, V_rand)
    lam_c = (C_IV_K_A + C_IV_H_A) / 2

    if line in ('k', 'K'):
        mask = lam <= lam_c
    elif line in ('h', 'H'):
        mask = lam > lam_c
    else:
        return lam, spec_tot, spec_halo

    return lam[mask], spec_tot[mask], spec_halo[mask]


# =====================================================
# Spatial distribution — path
# =====================================================

def Spatial_Distribution_path(path_RT, v_out, v_emit, v_rand, geo, atom,
                               Lumin, metals, Column_density_order):
    if v_out == 0:
        expand, vout_order = "000", 0
    elif v_out >= 1000:
        expand, vout_order = int(v_out / 10), 3
    else:
        expand, vout_order = v_out, 2

    emit, emit_order = calculate_order_and_value(v_emit)
    rand, rand_order = calculate_order_and_value(v_rand)
    lum              = int(Lumin * 10)

    if isinstance(geo, str):
        geo_map = {'NEBULA': 2, 'NEB': 2, 'QSO': 3,
                   'CONTINUUM': 4, 'CON': 4, 'TEST': 1}
        geo_upper = geo.upper()
        if geo_upper not in geo_map:
            raise ValueError(f"Invalid geo string: {geo}")
        geo = geo_map[geo_upper]
    elif isinstance(geo, int):
        if geo not in [1, 2, 3, 4]:
            raise ValueError(f"Invalid geo value: {geo}")
    else:
        raise TypeError(f"geo must be string or int, got {type(geo)}")

    metals_str = f"{int(metals * 1000):04d}" if metals < 1 else str(int(metals))
    col        = int(Column_density_order * 10)
    path_tt    = f'{path_RT}/{atom}L{lum}M{metals_str}NH{col}'

    if geo == 1:
        path_rt = (f'{path_tt}/N_atom{geo}00E+10_'
                   f'Vexp000E+00_Vemit100E+00_tauD000E+00_Vran000E+00')
    else:
        path_rt = (f'{path_tt}/N_atom{geo}00E+10_'
                   f'Vexp{expand}E+0{vout_order}_Vemit{emit}E+0{emit_order}_'
                   f'tauD000E+00_Vran{rand}E+0{rand_order}')

    return path_rt


def Surface_Brightness_RT(path_RT, v_out, v_emit, v_rand, geo, atom,
                           Lumin, metals, Column_density_order):
    path_file = Spatial_Distribution_path(
        path_RT, v_out, v_emit, v_rand, geo, atom,
        Lumin, metals, Column_density_order) + 'radi.dat'

    if not os.path.exists(path_file):
        print(f"Warning: RT file not found: {path_file}")
        return None

    try:
        name    = ['radius', 'SB_K', 'SB_H', 'SB_tot', '1', '2', '3']
        data_sp = pd.read_csv(path_file, sep=r'\s+', header=None, names=name)
        rad     = data_sp['radius'].to_numpy()
        SB_t    = data_sp['SB_tot'].to_numpy()
        return rad * 100, rad * 100 * kpc, SB_t
    except Exception as e:
        print(f"Error reading RT file {path_file}: {e}")
        return None


# =====================================================
# SED 계산 함수
# =====================================================

def QSO_SED(path_SED):
    file_w = pd.read_csv(
        os.path.join(path_SED, 'QSO.sed'),
        comment='#', sep=r'\s+', engine='python',
        header=None, names=['Ryd', 'nufnu']
    )
    return file_w['Ryd'].to_numpy(), file_w['nufnu'].to_numpy()


def SED_properties(Lumin, V_emit, metals, Column_density_order):
    path_SED = resolve_column_density_path(Lumin, metals, Column_density_order, 'CIV')
    Ryd, Fnu = QSO_SED(path_SED)
    nu        = Ryd * 13.6 / h_ev
    nuFnu     = Fnu * nu

    # searchsorted is O(log N) — faster than argmin(abs()) which is O(N)
    idx_1ryd          = np.searchsorted(Ryd, 1.0)
    normalized_factor = 10 ** Lumin / nuFnu[idx_1ryd]

    Lnu       = Fnu * normalized_factor
    lambda_cm = c_cms / nu
    lambda_A  = lambda_cm * 1e8
    Llambda   = Lnu * c_cms / lambda_cm ** 2 * 1e-8

    Lc = np.median(Llambda[(lambda_A >= 1450) & (lambda_A <= 1470)])

    peak_mask   = (lambda_A > 1548) & (lambda_A <= 1551)
    peak_idx    = np.where(peak_mask)[0]
    peak_idx    = peak_idx[np.argmax(Llambda[peak_idx])]
    Llambda_max = Llambda[peak_idx]
    lambda_c    = lambda_cm[peak_idx]

    # Vectorised Gaussian (was a lambda + element-wise loop via np.exp)
    del_lam               = V_emit / c_kms * lambda_c
    Gaussian_Fitting_Flux = (Llambda_max - Lc) * np.exp(
        -(lambda_cm - lambda_c) ** 2 / del_lam ** 2) + Lc

    line_mask   = (lambda_A >= 1500) & (lambda_A <= 1600)
    lam_line_J  = lambda_A[line_mask]
    Llam_line_J = Llambda[line_mask]
    idx_J       = np.argsort(lam_line_J)
    lam_line_J  = lam_line_J[idx_J]
    Llam_line_J = Llam_line_J[idx_J]
    y_line_J    = np.maximum((Llam_line_J - Lc) / Lc, 0)
    EW_direct_J = simpson(y_line_J, x=lam_line_J)

    return Lc, EW_direct_J, Gaussian_Fitting_Flux


# =====================================================
# Surface Brightness — CLOUDY  (numba-accelerated Abel)
# =====================================================

def Surface_Brightness_CLOUDY(z, radius_kpc, emissivity, dr, n_pts=512):
    """
    Compute the projected surface brightness via an Abel transform.

    The substitution r = sqrt(R^2 + t^2) removes the integrand singularity
    at r = R, so uniform-grid quadrature (trapezoid) converges accurately.
    The inner loop is executed by a numba @njit(parallel=True) kernel when
    numba is available, otherwise falls back to vectorised numpy.

    Parameters
    ----------
    n_pts : int
        Quadrature points per projected radius (default 512).
        Increase for higher accuracy on steep emissivity profiles.
    """
    Project_R = np.linspace(0, 100, 70) * kpc
    SB        = _abel_kernel(Project_R, radius_kpc, emissivity, float(z), n_pts)
    Lumin     = np.trapz(2.0 * np.pi * Project_R * SB, Project_R)
    return Project_R / kpc, SB / (np.pi * 4.0), Lumin


# =====================================================
# CLOUDY data 로드
# =====================================================

def _read_ele_column(path, ele, col_idx):
    """
    Load one ionic-fraction column from a CLOUDY .ele_X file.
    numpy.loadtxt replaces the original line-by-line Python loop (~10× faster).
    """
    data = np.loadtxt(f'{path}.ele_{ele}', skiprows=1)
    return data[:, col_idx]


def CLOUDY_data_path(Lumin, metals, Column_density_order):
    path_cloudy = resolve_column_density_path(
        Lumin, metals, Column_density_order, 'CIV/CLOUDY_QSO')

    Mod = pc.CloudyModel(path_cloudy)
    Mod.ionic_names

    frac_He = 1.00E-01
    frac_C  = 2.45E-04

    n_H  = Mod.nH
    n_He = n_H * frac_He
    n_C  = n_H * frac_C
    dr   = Mod.dr

    CIV_frac   = _read_ele_column(path_cloudy, 'C',  4)
    CV_frac    = _read_ele_column(path_cloudy, 'C',  5)
    CIII_frac  = _read_ele_column(path_cloudy, 'C',  3)
    HeII_frac  = _read_ele_column(path_cloudy, 'He', 2)
    HeIII_frac = _read_ele_column(path_cloudy, 'He', 3)
    HeI_frac   = _read_ele_column(path_cloudy, 'He', 1)
    HII_frac   = _read_ele_column(path_cloudy, 'H',  2)
    HI_frac    = _read_ele_column(path_cloudy, 'H',  1)

    nden_CIV  = CIV_frac  * n_C
    nden_HeII = HeII_frac * n_He

    CIV_Lum   = (float(Mod.get_emis_vol('C__4_154819A'))
                 + float(Mod.get_emis_vol('C__4_155078A')))
    CIV_emis  = Mod.get_emis('C__4_154819A') + Mod.get_emis('C__4_155078A')
    Lya_Lum   = float(Mod.get_emis_vol('H__1_121567A'))
    Lya_emis  = Mod.get_emis('H__1_121567A')
    HeII_Lum  = float(Mod.get_emis_vol('HE_2_164043A'))
    HeII_emis = Mod.get_emis('HE_2_164043A')

    radius_p_CIV,  SB_CIV,  Lumin_CIV  = Surface_Brightness_CLOUDY(
        0, Mod.radius, CIV_emis,  dr)
    radius_p_HeII, SB_HeII, Lumin_HeII = Surface_Brightness_CLOUDY(
        0, Mod.radius, HeII_emis, dr)
    radius_p_Lya,  SB_Lya,  Lumin_Lya  = Surface_Brightness_CLOUDY(
        0, Mod.radius, Lya_emis,  dr)

    return {
        'radius_p':            radius_p_CIV,
        'SB_CIV':              SB_CIV,
        'Lumin_CIV':           Lumin_CIV,
        'SB_HeII':             SB_HeII,
        'Lumin_HeII':          Lumin_HeII,
        'SB_Lya':              SB_Lya,
        'Lumin_Lya':           Lumin_Lya,
        'radius':              Mod.radius / kpc,
        'radius_kpc':          Mod.radius,
        'frac_CIII':           CIII_frac,
        'frac_CV':             CV_frac,
        'frac_HeI':            HeI_frac,
        'frac_HeIII':          HeIII_frac,
        'frac_HI':             HI_frac,
        'frac_HII':            HII_frac,
        'ne':                  Mod.ne,
        'Te':                  Mod.te,
        'Teff':                Mod.Teff,
        'logU':                Mod.log_U,
        'nH':                  n_H,
        'frac_CIV':            CIV_frac,
        'emis_CIV':            CIV_emis,
        'nden_CIV':            nden_CIV,
        'frac_HeII':           HeII_frac,
        'emis_HeII':           HeII_emis,
        'nden_HeII':           nden_HeII,
        'Lum_CIV':             CIV_Lum,
        'Lum_HeII':            HeII_Lum,
        'Lum_Lya':             Lya_Lum,
        'Column_density_CIV':  np.sum(dr * nden_CIV),
        'Column_density_HeII': np.sum(dr * nden_HeII),
        'Column_density_Lya':  np.sum(dr * n_H),
        'Column_density_H':    np.sum(dr * n_H),
    }


# =====================================================
# photon_number_SB
# =====================================================

def photon_number_SB(radius, origin_SB, CIV_lum):
    try:
        if len(radius) < 2 or len(origin_SB) != len(radius):
            print("Error: Invalid input arrays for photon_number_SB")
            return np.array([0.0]), np.array([0.0])

        R_rt       = (radius / 100.0).astype(np.float64)
        origin_arr = origin_SB.astype(np.float64)
        number_dis, sb = _photon_number_kernel(R_rt, origin_arr,
                                               float(CIV_lum), _KPC_CM)
        return number_dis, sb

    except Exception as e:
        print(f"Error in photon_number_SB: {e}")
        return np.array([0.0]), np.array([0.0])


# =====================================================
# spatial_distribution_CLOUDY
# =====================================================

def spatial_distribution_CLOUDY(atom,Lumin, metals, Column_density_order):
    cloudy_data = CLOUDY_data_path(Lumin, metals, Column_density_order)
    return cloudy_data['radius_p'], cloudy_data[f'SB_{atom}']


# =====================================================
# spatial_distribution_RT
# =====================================================

def spatial_distribution_RT(path_RT, V_out, V_emit, V_rand, geo, atom,
                             Lumin, metals, Column_density_order):
    Lc, EW_direct_J, _ = SED_properties(Lumin, V_emit, metals, Column_density_order)
    cloudy_data         = CLOUDY_data_path(Lumin, metals, Column_density_order)

    CIV_total_flux_QSO = EW_direct_J * Lc
    Lc_total_flux_QSO  = Lc * 30

    geo_upper = geo.upper() if isinstance(geo, str) else ''
    if geo_upper in ('NEBULA', 'TEST', 'NEB', 'NE'):
        CIV_Lumin = cloudy_data[f'Lumin_{atom}']
    elif geo_upper == 'QSO':
        CIV_Lumin = CIV_total_flux_QSO
    elif geo_upper in ('CONTINUUM', 'CON', 'CONT'):
        CIV_Lumin = Lc_total_flux_QSO
    else:
        raise TypeError(f"geo must be string or int, got {type(geo)}")

    result = Surface_Brightness_RT(path_RT, V_out, V_emit, V_rand, geo,
                                   atom, Lumin, metals, Column_density_order)
    radius, _, Num_SB = result
    number_dist, surface_brightness = photon_number_SB(radius, Num_SB, CIV_Lumin)

    return radius[1:-1], Num_SB, number_dist, surface_brightness[1:-1] / (np.pi * 4.0)


# =====================================================
# total_Scattered_CIV
# =====================================================

def total_Scattered_CIV(path_RT, V_out, V_emit, V_rand, Lumin, metals,
                         Column_density_order):
    atom      = "CIV"
    Total_CIV = 0

    for geo in ("NEB", "QSO", "Continuum"):
        radius, _, _, surface_brightness = spatial_distribution_RT(
            path_RT, V_out, V_emit, V_rand, geo, atom,
            Lumin, metals, Column_density_order)
        Total_CIV += surface_brightness

    return radius, Total_CIV


# =====================================================
# sr → arcsec² 변환
# =====================================================

def sr_to_arcsec_2(SB, z):
    """Surface Brightness 단위를 sr → arcsec² 로 변환"""
    return SB * (1.0 + z) ** 4 / 4.255e10


# =====================================================
# CLOUDY 파일 읽기 유틸리티
# =====================================================

def read_file(path_way, atom):
    """CLOUDY 모델 파일에서 luminosity, emissivity, number density 반환"""
    frac_C  = 2.45E-04
    frac_He = 1.00E-01

    Mod = pc.CloudyModel(path_way)
    Mod.ionic_names
    n_H  = Mod.nH
    n_He = n_H * frac_He
    n_C  = n_H * frac_C

    CIV_frac  = _read_ele_column(path_way, 'C',  4)
    HeII_frac = _read_ele_column(path_way, 'He', 2)

    nden_CIV  = CIV_frac  * n_C
    nden_HeII = HeII_frac * n_He

    if atom == 'CIV':
        lum  = (float(Mod.get_emis_vol('C__4_154819A'))
                + float(Mod.get_emis_vol('C__4_155078A')))
        emis = Mod.get_emis('C__4_154819A') + Mod.get_emis('C__4_155078A')
        den  = nden_CIV
    elif atom == 'Lya':
        lum  = float(Mod.get_emis_vol('H__1_121567A'))
        emis = Mod.get_emis('H__1_121567A')
        den  = n_H
    elif atom == 'HeII':
        lum  = float(Mod.get_emis_vol('HE_2_164043A'))
        emis = Mod.get_emis('HE_2_164043A')
        den  = nden_HeII
    else:
        raise ValueError(f"Unknown atom: {atom}. Choose 'CIV', 'Lya', or 'HeII'.")

    return lum, emis, den


def radius_info(path, atom=None):
    """CLOUDY 모델에서 radius [kpc], radius [cm], dr [cm] 반환"""
    Mod = pc.CloudyModel(path)
    return Mod.radius / kpc, Mod.radius, Mod.dr


def make_data_file(save_path, path, atom, Lumin, metals, Column_density_order):
    """CLOUDY 결과를 읽어 RT 입력용 텍스트 파일과 폴더를 생성합니다."""
    mode = "WO"

    lum_val, emis, den       = read_file(path, atom)
    radius_R, radius_kpc, dr = radius_info(path, atom)

    tt         = pd.DataFrame(np.column_stack((radius_R, emis, den)))
    lum_int    = int(Lumin * 10)
    col_int    = int(Column_density_order * 10)
    metals_str = f"{int(metals * 1000):04d}" if metals < 1 else str(int(metals))

    folder_name = f'{atom}L{lum_int}M{metals_str}NH{col_int}'
    folder_path = f'{save_path}{folder_name}'

    try:
        os.makedirs(folder_path, exist_ok=True)
        print(f"Created folder: {folder_path}")
    except Exception as e:
        print(f"Error creating folder {folder_path}: {e}")

    tt.to_csv(f'{save_path}{folder_name}.txt', sep='\t', index=False, header=False)

    file2       = f'{folder_path}/radius_emis_number_density.txt'
    header_text = (
        f"# ATOM = {atom}, log(nuLnu) = {Lumin} [erg/s] at 1 Ryd, "
        f"log(N_H) = {Column_density_order} [1/cm2], "
        f"metallicity = {metals} Z_sun, {mode} line transfer\n"
        f"# (1) radius [kpc] (2) emissivity [erg/s/cm3/sr]"
        f" (3) number_density [1/cm3]\n"
    )
    with open(file2, 'w') as f:
        f.write(header_text)
    tt.to_csv(file2, sep='\t', index=False, header=False, mode='a')
    print("make data file!")


def resolve_column_density_path_new(Lumin, metals, Column_density_order,
                                sub_path='CIV/CLOUDY_QSO'):
    cl_order  = _infer_multi_factor_from_column_density_order(Column_density_order)
    _val      = float(Column_density_order)
    _frac     = abs(_val - np.floor(_val))
    _dir      = np.floor(_val) if np.isclose(_frac, 0.5, atol=1e-6) else _val
    dir_str   = f"{_dir:.1f}"
    return os.path.expanduser(
        f'/home/jinlim/Make_CLOUDY_data/CLOUDY_17_Jun_8_2026/'
        f'Lum_{Lumin}_2/metal_{metals}/N_H_{cl_order}_{dir_str}/{sub_path}'
    )

def CLOUDY_data_path_new(Lumin, metals, Column_density_order):
    path_cloudy = resolve_column_density_path_new(
        Lumin, metals, Column_density_order, 'CIV/CLOUDY_QSO')

    Mod = pc.CloudyModel(path_cloudy)
    Mod.ionic_names

    frac_He = 1.00E-01
    frac_C  = 2.45E-04

    n_H  = Mod.nH
    n_He = n_H * frac_He
    n_C  = n_H * frac_C
    dr   = Mod.dr

    CIV_frac   = _read_ele_column(path_cloudy, 'C',  4)
    CV_frac    = _read_ele_column(path_cloudy, 'C',  5)
    CIII_frac  = _read_ele_column(path_cloudy, 'C',  3)
    HeII_frac  = _read_ele_column(path_cloudy, 'He', 2)
    HeIII_frac = _read_ele_column(path_cloudy, 'He', 3)
    HeI_frac   = _read_ele_column(path_cloudy, 'He', 1)
    HII_frac   = _read_ele_column(path_cloudy, 'H',  2)
    HI_frac    = _read_ele_column(path_cloudy, 'H',  1)

    nden_CIV  = CIV_frac  * n_C
    nden_HeII = HeII_frac * n_He

    CIV_Lum   = (float(Mod.get_emis_vol('C__4_154819A'))
                 + float(Mod.get_emis_vol('C__4_155078A')))
    CIV_emis  = Mod.get_emis('C__4_154819A') + Mod.get_emis('C__4_155078A')
    Lya_Lum   = float(Mod.get_emis_vol('H__1_121567A'))
    Lya_emis  = Mod.get_emis('H__1_121567A')
    HeII_Lum  = float(Mod.get_emis_vol('HE_2_164043A'))
    HeII_emis = Mod.get_emis('HE_2_164043A')

    radius_p_CIV,  SB_CIV,  Lumin_CIV  = Surface_Brightness_CLOUDY(
        0, Mod.radius, CIV_emis,  dr)
    radius_p_HeII, SB_HeII, Lumin_HeII = Surface_Brightness_CLOUDY(
        0, Mod.radius, HeII_emis, dr)
    radius_p_Lya,  SB_Lya,  Lumin_Lya  = Surface_Brightness_CLOUDY(
        0, Mod.radius, Lya_emis,  dr)

    return {
        'radius_p':            radius_p_CIV,
        'SB_CIV':              SB_CIV,
        'Lumin_CIV':           Lumin_CIV,
        'SB_HeII':             SB_HeII,
        'Lumin_HeII':          Lumin_HeII,
        'SB_Lya':              SB_Lya,
        'Lumin_Lya':           Lumin_Lya,
        'radius':              Mod.radius / kpc,
        'radius_kpc':          Mod.radius,
        'frac_CIII':           CIII_frac,
        'frac_CV':             CV_frac,
        'frac_HeI':            HeI_frac,
        'frac_HeIII':          HeIII_frac,
        'frac_HI':             HI_frac,
        'frac_HII':            HII_frac,
        'ne':                  Mod.ne,
        'Te':                  Mod.te,
        'Teff':                Mod.Teff,
        'logU':                Mod.log_U,
        'nH':                  n_H,
        'frac_CIV':            CIV_frac,
        'emis_CIV':            CIV_emis,
        'nden_CIV':            nden_CIV,
        'frac_HeII':           HeII_frac,
        'emis_HeII':           HeII_emis,
        'nden_HeII':           nden_HeII,
        'Lum_CIV':             CIV_Lum,
        'Lum_HeII':            HeII_Lum,
        'Lum_Lya':             Lya_Lum,
        'Column_density_CIV':  np.sum(dr * nden_CIV),
        'Column_density_HeII': np.sum(dr * nden_HeII),
        'Column_density_Lya':  np.sum(dr * n_H),
        'Column_density_H':    np.sum(dr * n_H),
    }

