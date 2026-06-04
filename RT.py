import numpy as np                                                                        
import matplotlib.pyplot as plt
import pyCloudy as pc
import pyneb as pn
from astropy    import constants as const
from astropy.io import ascii
import pandas as pd
from scipy import interpolate
import warnings
from scipy.integrate import quad, IntegrationWarning
import scipy.integrate as integrate
from numpy import log10, exp
import os
from astropy.constants import h, c
import astropy.units as u
from scipy import special
from scipy.interpolate import interp1d
from scipy.integrate import simpson
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore', category=IntegrationWarning)

kpc = const.kpc.cgs.value
h_ev = h.to(u.eV *u.s).value
c_cms = c.to(u.cm/u.s).value
c_kms = c.to(u.km/u.s).value
Ly_a_K = 1215.673644609e-8
Ly_a_H = 1215.668237310e-8
C_IV_K = 1548.187e-8 # cm
C_IV_H = 1550.772e-8 # cm 
C_IV_K_A = 1548.187 # Angstrom
C_IV_H_A = 1550.772 # Angstrom

nu_Lya_K = c_cms / Ly_a_K
nu_Lya_H = c_cms / Ly_a_H

nu_CIV_K = c_cms / C_IV_K 
nu_CIV_H = c_cms / C_IV_H

Ryd_Lya_K = h_ev * nu_Lya_K / 13.6 
Ryd_Lya_H = h_ev * nu_Lya_H / 13.6 

Ryd_CIV_K = h_ev * nu_CIV_K / 13.6 
Ryd_CIV_H = h_ev * nu_CIV_H / 13.6 


# =====================================================
# 공통 헬퍼 함수
# =====================================================

def calculate_order_and_value(value):
    """ 주어진 값에 대해 변환된 값과 해당 order 반환 """
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
        raise ValueError("Column_density_order must be convertible to float (e.g., 20.0 or 20.5)")

    fractional_part = abs(val - np.floor(val))
    if np.isclose(fractional_part, 0.0, atol=1e-6):
        return 1.0
    if np.isclose(fractional_part, 0.5, atol=1e-6):
        return 3.2
    raise ValueError("Only xx.0 or xx.5 are supported for Column_density_order to infer multi_factor")


def resolve_column_density_path(Lumin, metals, Column_density_order, sub_path='CIV/CLOUDY_QSO'):
    """
    Column_density_order (e.g. 20.0 or 20.5) 로부터
    ~/CLOUDY_setup/Lum_{Lumin}_2/metal_{metals}/N_H_{cl_order}_{dir_order_str}/{sub_path}
    경로를 생성해 반환합니다.

    - xx.0  → cl_order=1.0,  dir_order_str='xx.0'
    - xx.5  → cl_order=3.2,  dir_order_str='xx.0'  (floor 적용)
    """
    cl_order = _infer_multi_factor_from_column_density_order(Column_density_order)
    try:
        _val = float(Column_density_order)
    except Exception:
        _val = Column_density_order

    if isinstance(_val, (float, int)):
        _frac = abs(_val - np.floor(_val))
        _dir_order = np.floor(_val) if np.isclose(_frac, 0.5, atol=1e-6) else _val
        dir_order_str = f"{_dir_order:.1f}"
    else:
        dir_order_str = str(Column_density_order)

    return os.path.expanduser(
        f'~/CLOUDY_setup/Lum_{Lumin}_2/metal_{metals}/N_H_{cl_order}_{dir_order_str}/{sub_path}'
    )


# =====================================================
# Path_RT Part — Spectrum
# =====================================================

def Spectrum_path(path_RT, Line, atom_num, atom_index, V_out, V_emit, V_rand):
    out,  out_order  = calculate_order_and_value(V_out)
    emit, emit_order = calculate_order_and_value(V_emit)
    ran,  ran_order  = calculate_order_and_value(V_rand)
    path_real = (
        r'{}/N_atom{}0E+{}_Vexp{}E+0{}_Vemit{}E+0{}_tauD000E+00_Vran{}E+0{}'
        .format(path_RT, atom_num, atom_index,
                out, out_order, emit, emit_order, ran, ran_order)
    )
    return path_real


def Spectrum_K_or_H(path_RT, line, atom_num, atom_index, V_out, V_emit, V_rand):
    path_sp = Spectrum_path(path_RT, line, atom_num, atom_index, V_out, V_emit, V_rand)
    path_RT_Spectrum = f'{path_sp}spec.dat'
    try:
        data = pd.read_csv(path_RT_Spectrum, sep=r'\s+', header=None)
        print("v_rand = 11.8 km/s")
    except Exception:
        print('파일을 찾을 수 없습니다.', path_RT_Spectrum)
        return None, None, None

    x = data[0].to_numpy()

    try:
        if line in ('h', 'H'):
            spec_tot  = data[2].to_numpy()
            spec_scat = data[4].to_numpy()
        elif line in ('k', 'K'):
            spec_tot  = data[1].to_numpy()
            spec_scat = data[3].to_numpy()
        else:
            raise ValueError('invalid line')
        return x, spec_tot, spec_scat
    except Exception:
        print('line을 입력하세요 K or H')
        return None, None, None


def f_esc(path_RT, line, atom_num, atom_index, V_out, V_emit, V_rand):
    path_sp = Spectrum_path(path_RT, line, atom_num, atom_index, V_out, V_emit, V_rand)
    path_RT_Spectrum = f'{path_sp}_f_esc.dat'
    try:
        data = pd.read_csv(path_RT_Spectrum, sep=r'\s+', header=None)
    except Exception:
        print('파일을 찾을 수 없습니다.', path_RT_Spectrum)
        return None

    return {
        'total_esc':  data.iloc[0, 0],
        'ratio_k_h':  data.iloc[0, 1],
        'ratio_k':    data.iloc[0, 2],
        'ratio_h':    data.iloc[0, 3],
        'ns_k':       data.iloc[0, 4],
        'ns_h':       data.iloc[0, 5],
        'ns_dust_k':  data.iloc[0, 6],
        'ns_dust_h':  data.iloc[0, 7],
        'path_k':     data.iloc[0, 8],
        'path_h':     data.iloc[0, 9],
        'dir_esc_k':  data.iloc[0, 10],
        'dir_esc_h':  data.iloc[0, 11],
        'ns_clump_k': data.iloc[0, 12],
        'ns_clump_h': data.iloc[0, 13],
    }


def Spectrum_All(path_RT, line, atom_num, atom_index, V_out, V_emit, V_rand):
    path_sp = Spectrum_path(path_RT, line, atom_num, atom_index, V_out, V_emit, V_rand)
    path_RT_Spectrum = f'{path_sp}spec_com.dat'
    try:
        data = pd.read_csv(path_RT_Spectrum, sep=r'\s+', header=None)
    except Exception:
        print('파일을 찾을 수 없습니다.', path_RT_Spectrum)
        return None, None, None

    lam      = data[0].to_numpy()
    spec_tot = data[1].to_numpy()
    spec_sc  = data[2].to_numpy()
    return lam, spec_tot, spec_sc


def Spectrum_RT(path_RT, line, atom_num, atom_index, V_out, V_emit, V_rand):
    lam, spec_tot, spec_halo = Spectrum_All(path_RT, line, atom_num, atom_index, V_out, V_emit, V_rand)
    lam_c = (C_IV_K_A + C_IV_H_A) / 2

    if line in ('k', 'K'):
        ioc = np.where(lam <= lam_c)[0]
    elif line in ('h', 'H'):
        ioc = np.where(lam > lam_c)[0]
    else:
        return lam, spec_tot, spec_halo

    return lam[ioc], spec_tot[ioc], spec_halo[ioc]


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

    lum = int(Lumin * 10)

    # geo 처리
    if isinstance(geo, str):
        geo_upper = geo.upper()
        geo_map = {'NEBULA': 2, 'NEB': 2, 'QSO': 3,
                   'CONTINUUM': 4, 'CON': 4, 'TEST': 1}
        if geo_upper not in geo_map:
            raise ValueError(f"Invalid geo string: {geo}")
        geo = geo_map[geo_upper]
    elif isinstance(geo, int):
        if geo not in [1, 2, 3, 4]:
            raise ValueError(f"Invalid geo value: {geo}")
    else:
        raise TypeError(f"geo must be string or int, got {type(geo)}")

    # metals 문자열 처리
    if metals < 1:
        metals_str = f"{int(metals * 1000):04d}"
    else:
        metals_str = str(int(metals))

    col = int(Column_density_order * 10)

    path_tt = f'{path_RT}/{atom}L{lum}M{metals_str}NH{col}'

    if geo == 1:
        path_rt = (f'{path_tt}/N_atom{geo}00E+10_'
                   f'Vexp000E+00_Vemit100E+00_'
                   f'tauD000E+00_Vran000E+00')
    else:
        path_rt = (f'{path_tt}/N_atom{geo}00E+10_'
                   f'Vexp{expand}E+0{vout_order}_Vemit{emit}E+0{emit_order}_'
                   f'tauD000E+00_Vran{rand}E+0{rand_order}')

    return path_rt


def Surface_Brightness_RT(path_RT, v_out, v_emit, v_rand, geo, atom,
                           Lumin, metals, Column_density_order):
    path_sb = Spatial_Distribution_path(path_RT, v_out, v_emit, v_rand, geo,
                                        atom, Lumin, metals, Column_density_order)
    path_Spatial_Distribution = f'{path_sb}radi.dat'

    if not os.path.exists(path_Spatial_Distribution):
        print(f"Warning: RT file not found: {path_Spatial_Distribution}")
        return None

    try:
        name = ['radius', 'SB_K', 'SB_H', 'SB_tot', '1', '2', '3']
        data_sp = pd.read_csv(path_Spatial_Distribution, sep=r'\s+',
                              header=None, names=name)
        rad  = data_sp['radius'].to_numpy()
        SB_t = data_sp['SB_tot'].to_numpy()
        return rad * 100, rad * 100 * kpc, SB_t
    except Exception as e:
        print(f"Error reading RT file {path_Spatial_Distribution}: {e}")
        return None


# =====================================================
# SED 계산 함수
# =====================================================

def QSO_SED(path_SED):
    file_name_w = os.path.join(path_SED, 'QSO.sed')
    file_w = pd.read_csv(
        file_name_w,
        comment='#',
        sep=r'\s+',
        engine='python',
        header=None,
        names=['Ryd', 'nufnu']
    )
    Ryd_w  = file_w['Ryd'].to_numpy()
    nufnu_w = file_w['nufnu'].to_numpy()
    return Ryd_w, nufnu_w


def SED_properties(Lumin, V_emit, metals, Column_density_order):
    # ← 수정: _resolve_column_density_path 사용
    path_SED = _resolve_column_density_path(Lumin, metals, Column_density_order, 'CIV')

    Ryd, Fnu = QSO_SED(path_SED)
    nu   = Ryd * 13.6 / h_ev
    nuFnu = Fnu * nu

    idx_1ryd = np.argmin(np.abs(Ryd - 1.0))
    normalized_factor = 10**Lumin / nuFnu[idx_1ryd]

    Lnu    = Fnu * normalized_factor
    nuLnu  = Lnu * nu

    lambda_cm = c_cms / nu
    lambda_A  = lambda_cm * 1e8
    Llambda   = Lnu * c_cms / lambda_cm**2 * 1e-8

    cont_mask = (lambda_A >= 1450) & (lambda_A <= 1470)
    Lc = np.median(Llambda[cont_mask])

    peak_mask    = (lambda_A > 1548) & (lambda_A <= 1551)
    peak_indices = np.where(peak_mask)[0]
    peak_idx     = peak_indices[np.argmax(Llambda[peak_indices])]
    Llambda_max  = Llambda[peak_idx]
    lambda_c     = lambda_cm[peak_idx]

    def Flux(Fc, F_max, lam, lam_c, V_emit):
        del_lam = V_emit / c_kms * lam_c
        return (F_max - Fc) * np.exp(-(lam - lam_c)**2 / del_lam**2) + Fc

    Gaussian_Fitting_Flux = Flux(Lc, Llambda_max, lambda_cm, lambda_c, V_emit)

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
# Surface Brightness — CLOUDY
# =====================================================

def Surface_Brightness_CLOUDY(z, radius_kpc, emissivity, dr):
    r_min, r_max = radius_kpc.min(), radius_kpc.max()
    Project_R = np.linspace(0, 100, 70) * kpc
    N = len(Project_R)
    surface_brightness = np.zeros(N)

    emis_interp = interp1d(radius_kpc, emissivity, bounds_error=False, fill_value=0)

    for ii, R in enumerate(Project_R):
        def integrand(r):
            if r < R:
                return 0
            emis = emis_interp(r)
            return emis * r / np.sqrt(r**2 - R**2)
        surface_brightness[ii], _ = quad(integrand, R, r_max)
        surface_brightness[ii] *= 2 / (1 + z)**4

    dR    = Project_R[1] - Project_R[0]
    Lumin = np.trapz(2 * np.pi * Project_R * surface_brightness, Project_R)

    return Project_R / kpc, surface_brightness / (np.pi * 4), Lumin


# =====================================================
# CLOUDY data 로드
# =====================================================

def CLOUDY_data_path(Lumin, metals, Column_density_order):
    # ← 수정: _resolve_column_density_path 사용
    path_cloudy = _resolve_column_density_path(Lumin, metals, Column_density_order, 'CIV/CLOUDY_QSO')

    Mod = pc.CloudyModel(path_cloudy)
    Mod.ionic_names
    N_H = sum(Mod.dr * Mod.nH)

    frac_He = 1.00E-01
    frac_C  = 2.45E-04
    frac_O  = 4.90E-04
    frac_N  = 8.51E-05
    frac_Mg = 3.47E-05

    N_HI  = sum(Mod.dr * Mod.nH * Mod.get_ionic('H', 0))
    N_HII = sum(Mod.dr * Mod.nH * Mod.get_ionic('H', 1))
    N_CIV = frac_C * sum(Mod.dr * Mod.nH * Mod.get_ionic('C', 3))

    num = len(Mod.nH)

    def _read_ele_column(path_cloudy, ele, col_idx):
        frac = np.zeros(num)
        with open(f'{path_cloudy}.ele_{ele}', 'r') as f:
            f.readline()
            for i, line in enumerate(f):
                frac[i] = float(line.strip().split()[col_idx])
        return frac

    CIV_frac  = _read_ele_column(path_cloudy, 'C',  4)
    CV_frac   = _read_ele_column(path_cloudy, 'C',  5)
    CIII_frac = _read_ele_column(path_cloudy, 'C',  3)
    HeII_frac = _read_ele_column(path_cloudy, 'He', 2)
    HeIII_frac= _read_ele_column(path_cloudy, 'He', 3)
    HeI_frac  = _read_ele_column(path_cloudy, 'He', 1)
    HII_frac  = _read_ele_column(path_cloudy, 'H',  2)
    HI_frac   = _read_ele_column(path_cloudy, 'H',  1)

    radius     = Mod.radius / kpc
    radius_kpc = Mod.radius
    dr         = Mod.dr
    n_H        = Mod.nH
    n_e        = Mod.ne
    n_He       = n_H * frac_He
    n_C        = n_H * frac_C
    nden_CIV   = CIV_frac  * n_C
    nden_HeII  = HeII_frac * n_He

    CIV_Lum   = float(Mod.get_emis_vol('C__4_154819A')) + float(Mod.get_emis_vol('C__4_155078A'))
    CIV_emis  = Mod.get_emis('C__4_154819A') + Mod.get_emis('C__4_155078A')
    Lya_Lum   = float(Mod.get_emis_vol('H__1_121567A'))
    Lya_emis  = Mod.get_emis('H__1_121567A')
    HeII_Lum  = float(Mod.get_emis_vol('HE_2_164043A'))
    HeII_emis = Mod.get_emis('HE_2_164043A')

    CIV_Column_density  = np.sum(dr * nden_CIV)
    HeII_Column_density = np.sum(dr * nden_HeII)
    Lya_Column_density  = np.sum(dr * n_H)
    H_Column_density    = np.sum(dr * n_H)

    radius_p_CIV,  SB_CIV,  Lumin_CIV  = Surface_Brightness_CLOUDY(0, radius_kpc, CIV_emis,  dr)
    radius_p_HeII, SB_HeII, Lumin_HeII = Surface_Brightness_CLOUDY(0, radius_kpc, HeII_emis, dr)
    radius_p_Lya,  SB_Lya,  Lumin_Lya  = Surface_Brightness_CLOUDY(0, radius_kpc, Lya_emis,  dr)

    return {
        'radius_p':             radius_p_CIV,
        'SB_CIV':               SB_CIV,
        'Lumin_CIV':            Lumin_CIV,
        'SB_HeII':              SB_HeII,
        'Lumin_HeII':           Lumin_HeII,
        'SB_Lya':               SB_Lya,
        'Lumin_Lya':            Lumin_Lya,
        'radius':               radius,
        'radius_kpc':           radius_kpc,
        'frac_CIII':            CIII_frac,
        'frac_CV':              CV_frac,
        'frac_HeI':             HeI_frac,
        'frac_HeIII':           HeIII_frac,
        'frac_HI':              HI_frac,
        'frac_HII':             HII_frac,
        'ne':                   n_e,
        'Te':                   Mod.te,
        'Teff':                 Mod.Teff,
        'logU':                 Mod.log_U,
        'nH':                   n_H,
        'frac_CIV':             CIV_frac,
        'emis_CIV':             CIV_emis,
        'nden_CIV':             nden_CIV,
        'frac_HeII':            HeII_frac,
        'emis_HeII':            HeII_emis,
        'nden_HeII':            nden_HeII,
        'Lum_CIV':              CIV_Lum,
        'Lum_HeII':             HeII_Lum,
        'Lum_Lya':              Lya_Lum,
        'Column_density_CIV':   CIV_Column_density,
        'Column_density_HeII':  HeII_Column_density,
        'Column_density_Lya':   Lya_Column_density,
        'Column_density_H':     H_Column_density,
    }


# =====================================================
# photon_number_SB
# =====================================================

def photon_number_SB(radius, origin_SB, CIV_lum):
    try:
        if len(radius) < 2 or len(origin_SB) != len(radius):
            print("Error: Invalid input arrays for photon_number_SB")
            return np.array([0]), np.array([0])

        R_rt = radius / 100
        dR   = R_rt[1] - R_rt[0]
        number_dis          = np.zeros(len(R_rt))
        surface_brightness_RT = np.zeros(len(R_rt))

        for ii, R in enumerate(R_rt):
            if R == 0:
                area = np.pi * (0.5 * dR)**2
            elif R == R_rt[-1]:
                area = np.pi * (2 * R + 0.5 * dR) * 0.5 * dR
            else:
                area = 2 * np.pi * R * dR
            number_dis[ii] = origin_SB[ii] * area

        total_Number = np.sum(number_dis)
        if total_Number == 0:
            print("Warning: Total number is zero")
            return number_dis, surface_brightness_RT

        factor_atom = CIV_lum / total_Number

        for ii, R in enumerate(R_rt):
            if R == 0:
                area = np.pi * (0.5 * dR)**2
            elif R == R_rt[-1]:
                area = np.pi * (2 * R + 0.5 * dR) * 0.5 * dR
            else:
                area = 2 * np.pi * R * dR
            factor_area = (100 * kpc)**2
            surface_brightness_RT[ii] = number_dis[ii] * factor_atom / (area * factor_area)

        return number_dis, surface_brightness_RT

    except Exception as e:
        print(f"Error in photon_number_SB: {e}")
        return np.array([0]), np.array([0])


# =====================================================
# spatial_distribution_CLOUDY
# =====================================================

def spatial_distribution_CLOUDY(path_RT, V_out, V_emit, V_rand, geo, atom,
                                 Lumin, metals, Column_density_order):
    # ← 수정: _resolve_column_density_path 사용
    path_SED = _resolve_column_density_path(Lumin, metals, Column_density_order, 'CIV')
    Lc, EW_direct_J, _ = SED_properties(Lumin, V_emit, metals, Column_density_order)
    cloudy_data = CLOUDY_data_path(Lumin, metals, Column_density_order)

    radius             = cloudy_data['radius_p']
    intrinsic_brightness = cloudy_data[f'SB_{atom}']

    return radius, intrinsic_brightness


# =====================================================
# spatial_distribution_RT
# =====================================================

def spatial_distribution_RT(path_RT, V_out, V_emit, V_rand, geo, atom,
                             Lumin, metals, Column_density_order):
    # ← 수정: _resolve_column_density_path 사용
    path_SED = _resolve_column_density_path(Lumin, metals, Column_density_order, 'CIV')

    Lc, EW_direct_J, _ = SED_properties(Lumin, V_emit, metals, Column_density_order)
    cloudy_data = CLOUDY_data_path(Lumin, metals, Column_density_order)

    CIV_total_flux_QSO = EW_direct_J * Lc
    Lc_total_flux_QSO  = Lc * 30  #(1600 - 1500)

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
    radius  = result[0]
    Num_SB  = result[2]

    number_dist, surface_brightness = photon_number_SB(radius, Num_SB, CIV_Lumin)

    return radius[1:-1], Num_SB, number_dist, surface_brightness[1:-1] / (np.pi * 4)


# =====================================================
# total_Scattered_CIV
# =====================================================

def total_Scattered_CIV(path_RT, V_out, V_emit, V_rand, Lumin, metals, Column_density_order):
    atom      = "CIV"
    Total_CIV = 0

    for geo in ("NEB", "QSO", "Continuum"):
        radius, Num_SB, number_dist, surface_brightness = spatial_distribution_RT(
            path_RT, V_out, V_emit, V_rand, geo, atom, Lumin, metals, Column_density_order
        )
        Total_CIV += surface_brightness

    return radius, Total_CIV


# =====================================================
# sr → arcsec² 변환
# =====================================================

def sr_to_arcsec_2(SB, z):
    """Surface Brightness 단위를 sr → arcsec² 로 변환"""
    SB_arc = SB * (1 + z)**4 / 4.255e10
    return SB_arc


# =====================================================
# CLOUDY 파일 읽기 유틸리티
# =====================================================
 
def read_file(path_way, atom):
    """CLOUDY 모델 파일에서 luminosity, emissivity, number density 반환"""
    frac_He = 1.00E-01
    frac_C  = 2.45E-04
    frac_O  = 4.90E-04
    frac_N  = 8.51E-05
    frac_Mg = 3.47E-05
 
    Mod = pc.CloudyModel(path_way)
    Mod.ionic_names
    num = len(Mod.nH)
 
    CIV_frac  = _read_ele_column(path_way, 'C',  4, num)
    HeII_frac = _read_ele_column(path_way, 'He', 2, num)
 
    n_H       = Mod.nH
    n_He      = n_H * frac_He
    n_C       = n_H * frac_C
    nden_CIV  = CIV_frac  * n_C
    nden_HeII = HeII_frac * n_He
 
    if atom == 'CIV':
        Cloudy_Lum  = (float(Mod.get_emis_vol('C__4_154819A'))
                       + float(Mod.get_emis_vol('C__4_155078A')))
        Cloudy_emis = (Mod.get_emis('C__4_154819A')
                       + Mod.get_emis('C__4_155078A'))
        Cloudy_den  = nden_CIV
    elif atom == 'Lya':
        Cloudy_Lum  = float(Mod.get_emis_vol('H__1_121567A'))
        Cloudy_emis = Mod.get_emis('H__1_121567A')
        Cloudy_den  = n_H
    elif atom == 'HeII':
        Cloudy_Lum  = float(Mod.get_emis_vol('HE_2_164043A'))
        Cloudy_emis = Mod.get_emis('HE_2_164043A')
        Cloudy_den  = nden_HeII
    else:
        raise ValueError(f"Unknown atom: {atom}. Choose 'CIV', 'Lya', or 'HeII'.")
 
    return Cloudy_Lum, Cloudy_emis, Cloudy_den
 
 
def _read_ele_column(path_way, ele, col_idx, num):
    """
    CLOUDY .ele_{ele} 파일에서 특정 열(col_idx)을 읽어 numpy array로 반환.
    (CLOUDY_data_path 및 read_file 공용 헬퍼)
    """
    frac = np.zeros(num)
    with open(f'{path_way}.ele_{ele}', 'r') as f:
        f.readline()  # header skip
        for i, line in enumerate(f):
            frac[i] = float(line.strip().split()[col_idx])
    return frac
 
 
def radius_info(path, atom=None):
    """CLOUDY 모델에서 radius [kpc], radius [cm], dr [cm] 반환"""
    Mod = pc.CloudyModel(path)
    radius     = Mod.radius / kpc
    radius_kpc = Mod.radius
    dr         = Mod.dr
    return radius, radius_kpc, dr
 
 
def make_data_file(path, atom, Lumin, metals, Column_density_order):
    """
    CLOUDY 결과를 읽어 RT 입력용 텍스트 파일과 폴더를 생성합니다.
 
    Parameters
    ----------
    path   : str  — CLOUDY 모델 파일 경로 (확장자 제외)
    atom   : str  — 'CIV', 'Lya', 'HeII'
    Lumin  : float — log(nuLnu) at 1 Ryd [erg/s]
    metals : float — metallicity [Z_sun]
    Column_density_order : float — e.g. 20.0 or 20.5
    """
    mode = "WO"
 
    lum_val, emis, den = read_file(path, atom)
    radius_R, radius_kpc, dr = radius_info(path, atom)
 
    tt = pd.DataFrame(np.column_stack((radius_R, emis, den)))
 
    # 폴더/파일명 구성 — Column_density_order 처리 통일
    lum_int = int(Lumin * 10)
    col_int = int(Column_density_order * 10)
 
    if metals < 1:
        metals_str = f"{int(metals * 1000):04d}"
    else:
        metals_str = str(int(metals))
 
    folder_name = f'{atom}L{lum_int}M{metals_str}NH{col_int}'
    folder_path = f'/home/jin/RT/RT_Run_data/{folder_name}'
 
    try:
        os.makedirs(folder_path, exist_ok=True)
        print(f"Created folder: {folder_path}")
    except Exception as e:
        print(f"Error creating folder {folder_path}: {e}")
 
    # 메인 텍스트 파일 저장 (RT 입력용)
    tt.to_csv(
        f'/home/jin/RT/RT_Run_data/{folder_name}.txt',
        sep='\t', index=False, header=False
    )
 
    # 상세 정보 파일 저장
    file2 = f'{folder_path}/radius_emis_number_density.txt'
    header_text = (
        f"# ATOM = {atom}, log(nuLnu) = {Lumin} [erg/s] at 1 Ryd, "
        f"log(N_H) = {Column_density_order} [1/cm2], "
        f"metallicity = {metals} Z_sun, {mode} line transfer\n"
        f"# (1) radius [kpc] (2) emissivity [erg/s/cm3/sr] (3) number_density [1/cm3]\n"
    )
    with open(file2, "w") as f:
        f.write(header_text)
    tt.to_csv(file2, sep="\t", index=False, header=False, mode="a")
 
    print("make data file!")
    return
