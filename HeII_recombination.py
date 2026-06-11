"""
HeII_recombination.py
=====================
He II 1640Å 재결합 방출선 emissivity 계산 및 Cloudy 결과 비교.

두 가지 방법:
  1. Cloudy   : pyCloudy로 모델 파일에서 직접 읽음 (HE_2_164043A)
  2. Case B   : PyNeb (Storey & Hummer 1995) 테이블 + 반경별 Te, ne, nden_HeIII

재결합 반응: He³⁺ + e⁻ → He²⁺* → He²⁺ + hν
따라서 rate ∝ n(He³⁺) × n_e  (재결합하는 쪽은 He³⁺)

사용법:
    python3 HeII_recombination.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pyCloudy as pc
import pyneb as pn
from astropy import constants as const
import warnings

warnings.filterwarnings('ignore')

# ── constants ─────────────────────────────────────────────────────────────────
kpc = const.kpc.cgs.value

# ── PyNeb He II RecAtom (Storey & Hummer 1995, T range: 500–30000 K) ─────────
HE2 = pn.RecAtom('He', 2)
HE2_T_MAX = 30000.0   # K  — 테이블 상한


# ── helper: Cloudy .ele 파일에서 이온 분율 읽기 ───────────────────────────────
def _read_ele_column(path, element, ion_stage):
    """element 의 ion_stage 번째 이온 분율 배열을 반환한다."""
    ele_file = f"{path}.ele_{element}"
    data = np.genfromtxt(ele_file)
    return data[:, ion_stage]   # col 0: depth, col 1: neutral, col 2: 1st ion, …


# ── helper: Cloudy 경로 ───────────────────────────────────────────────────────
def cloudy_path(Lumin, metals, N_H_order):
    """
    예: Lumin=44.0, metals=1.0, N_H_order=22.0
    → ~/CIV_RT_scat_data/CLOUDY_setup/Lum_44.0_2/metal_1.0/N_H_1_22.0/CIV/CLOUDY_QSO
    """
    from RT_v1 import resolve_column_density_path
    return resolve_column_density_path(Lumin, metals, N_H_order, 'CIV/CLOUDY_QSO')


# ── Case B emissivity profile ─────────────────────────────────────────────────
def caseB_HeII_emissivity(Te, ne, nden_HeIII, wave=1640):
    """
    반경별 He II Case B emissivity 계산 [erg/s/cm³].

    Parameters
    ----------
    Te         : array, 전자온도 [K]
    ne         : array, 전자밀도 [cm⁻³]
    nden_HeIII : array, He³⁺ 수밀도 [cm⁻³]  (= HeIII_frac × n_He)
                 재결합 반응: He³⁺ + e⁻ → He²⁺* + hν
                 rate ∝ n(He³⁺) × n_e
    wave       : float, 파장 [Å] — 기본값 1640 (n=3→2)

    Returns
    -------
    emis      : array [erg/s/cm³],  nan인 구간은 T > 30000 K 초과 zone
    alpha_eff : array [cm³/s],      effective recombination coefficient
                j_λ = h*ν × α_eff  (PyNeb 반환값 / h*ν)

    Notes
    -----
    PyNeb getEmissivity(T, ne, wave, product=False) 반환값:
      j_λ = h*ν × α_eff^B(T, ne)  [erg·s⁻¹·cm³]
    따라서
      ε(r) [erg/s/cm³] = j_λ(Te(r), ne(r)) × ne(r) × nden_HeIII(r)

    Case B 조건:
      He II Lyman 계열 (n→1, 특히 303.78 Å) 광자는 광학적으로 두꺼운
      nebula에서 재흡수된다고 가정. 관측되는 방출은 Balmer 이상 계열.
    """
    Te         = np.asarray(Te,         dtype=float)
    ne         = np.asarray(ne,         dtype=float)
    nden_HeIII = np.asarray(nden_HeIII, dtype=float)

    # T 범위 체크
    out_of_range = Te > HE2_T_MAX
    if out_of_range.any():
        n_bad = out_of_range.sum()
        print(f"[경고] {n_bad}개 zone의 Te > {HE2_T_MAX:.0f} K → NaN 처리됨 "
              f"(T_max={Te[out_of_range].max():.0f} K)")

    # PyNeb: product=False → 각 zone에 대해 pairwise 계산
    # j_pyneb = h*ν × α_eff^B(T, ne)  [erg·s⁻¹·cm³]
    j_pyneb = HE2.getEmissivity(Te, ne, wave=wave, product=False)

    hnu       = 6.626e-27 * 3e10 / (wave * 1e-8)      # erg
    alpha_eff = j_pyneb / hnu                          # cm³/s
    emis      = j_pyneb * ne * nden_HeIII              # erg/s/cm³

    return emis, alpha_eff


# ── 전체 비교 함수 ─────────────────────────────────────────────────────────────
def compare_HeII(Lumin=44.0, metals=1.0, Column_density_order=22.0,
                 wave=1640, save_fig=None):
    """
    Cloudy 결과와 PyNeb Case B 결과를 반경 profile로 비교.

    Parameters
    ----------
    Lumin, metals, Column_density_order : Cloudy 모델 파라미터
    wave       : He II 파장 [Å], 기본 1640 (n=3→2)
    save_fig   : 파일 경로 지정 시 저장 (None이면 화면 표시)
    """
    path = cloudy_path(Lumin, metals, Column_density_order)
    print(f"Cloudy 모델 경로: {path}")

    Mod = pc.CloudyModel(path)

    # ── 기본 물리량 (반경별 profile) ──────────────────────────────────────────
    frac_He = 1.00e-01
    nH      = Mod.nH                           # [cm⁻³]
    n_He    = nH * frac_He                     # [cm⁻³]
    Te      = Mod.te                           # [K]
    ne      = Mod.ne                           # [cm⁻³]
    dr      = Mod.dr                           # [cm]
    radius_kpc = Mod.radius / kpc              # [kpc] (Cloudy radius는 cm 단위)
    radius_cm  = Mod.radius                    # [cm]

    # He²⁺, He³⁺ 수밀도
    HeII_frac  = _read_ele_column(path, 'He', 2)
    HeIII_frac = _read_ele_column(path, 'He', 3)
    nden_HeII  = HeII_frac  * n_He             # [cm⁻³]
    nden_HeIII = HeIII_frac * n_He             # [cm⁻³]  ← 재결합하는 쪽

    # ── Cloudy 직접 계산 emissivity ───────────────────────────────────────────
    emis_cloudy = Mod.get_emis('HE_2_164043A') # [erg/s/cm³]

    # ── PyNeb Case B emissivity ───────────────────────────────────────────────
    emis_pyneb, alpha_eff = caseB_HeII_emissivity(Te, ne, nden_HeIII, wave=wave)

    # ── 비율 계산 ─────────────────────────────────────────────────────────────
    ratio = np.where(emis_cloudy > 0, emis_pyneb / emis_cloudy, np.nan)

    # ── 출력 요약 ─────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  He II {wave}Å  emissivity 비교  "
          f"(Lum={Lumin}, Z={metals}, logNH={Column_density_order})")
    print(f"{'='*55}")
    print(f"{'항목':<30}{'Cloudy':>12}{'Case B':>12}")
    print(f"{'-'*55}")

    valid = np.isfinite(emis_pyneb) & (emis_cloudy > 0)
    int_c = np.trapz(emis_cloudy[valid], radius_cm[valid])
    int_p = np.trapz(emis_pyneb[valid],  radius_cm[valid])
    print(f"{'∫ emis dr [erg/s/cm²]':<30}{int_c:>12.3e}{int_p:>12.3e}")
    print(f"{'ratio (CaseB / Cloudy)':<30}{'':>12}{int_p/int_c:>12.3f}")
    print(f"{'T_e 범위 [K]':<30}{Te.min():>12.0f} – {Te.max():.0f}")
    print(f"{'n_e 범위 [cm⁻³]':<30}{ne.min():>12.2e} – {ne.max():.2e}")
    print(f"{'n(He²⁺) 범위 [cm⁻³]':<30}"
          f"{nden_HeII.min():>12.2e} – {nden_HeII.max():.2e}")
    print(f"{'n(He³⁺) 범위 [cm⁻³]':<30}"
          f"{nden_HeIII.min():>12.2e} – {nden_HeIII.max():.2e}")
    print(f"{'α_eff(1640) @ T=10⁴K':<30}"
          f"{'':>12}{HE2.getEmissivity(1e4, 1e2, wave=wave):>12.3e}  [erg·s⁻¹·cm³]")

    # ── 플롯 ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    fig.suptitle(
        f"He II {wave}Å  |  Lum={Lumin}, Z={metals}, logNH={Column_density_order}",
        fontsize=14)

    # (1) Emissivity 비교
    ax = axes[0]
    ax.plot(radius_kpc, emis_cloudy, 'k-',  lw=2,   label='Cloudy (full RT)')
    ax.plot(radius_kpc, emis_pyneb,  'r--', lw=1.5, label='Case B (PyNeb, Storey & Hummer 1995)')
    ax.set_yscale('log')
    ax.set_ylabel(r'$\varepsilon$ [erg s$^{-1}$ cm$^{-3}$]', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # (2) Ratio
    ax = axes[1]
    ax.plot(radius_kpc, ratio, 'b-', lw=1.5)
    ax.axhline(1.0, color='k', ls='--', alpha=0.5)
    ax.set_ylabel('Case B / Cloudy', fontsize=12)
    ax.set_ylim(0, 3)
    ax.grid(True, alpha=0.3)

    # (3) Te, ne profile
    ax = axes[2]
    ax_ne = ax.twinx()
    l1, = ax.plot(radius_kpc, Te,   'r-',  lw=1.5, label=r'$T_e$ [K]')
    l2, = ax_ne.plot(radius_kpc, ne, 'b-', lw=1.5, label=r'$n_e$ [cm$^{-3}$]')
    ax.set_ylabel(r'$T_e$ [K]', fontsize=12, color='r')
    ax_ne.set_ylabel(r'$n_e$ [cm$^{-3}$]', fontsize=12, color='b')
    ax.set_xlabel('Radius [kpc]', fontsize=12)
    ax.legend(handles=[l1, l2], fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_fig:
        plt.savefig(save_fig, dpi=150, bbox_inches='tight')
        print(f"\n그림 저장: {save_fig}")
    else:
        plt.show()

    return {
        'radius_kpc':  radius_kpc,
        'Te':          Te,
        'ne':          ne,
        'nden_HeII':   nden_HeII,
        'nden_HeIII':  nden_HeIII,
        'emis_cloudy': emis_cloudy,
        'emis_caseB':  emis_pyneb,
        'alpha_eff':   alpha_eff,
        'ratio':       ratio,
    }


# ── 단일 zone 계산 예시 (빠른 확인용) ─────────────────────────────────────────
def quick_check(Te=1e4, ne=1e2, nden_HeIII=1.0, wave=1640):
    """
    단일 조건에서 Case B emissivity를 계산하는 간단한 예시.

    emissivity = j_λ(T, n_e) × n_e × n(He³⁺)
    j_λ = h*ν × α_eff^B  [erg·s⁻¹·cm³]
    """
    j     = HE2.getEmissivity(Te, ne, wave=wave)
    hnu   = 6.626e-27 * 3e10 / (wave * 1e-8)   # erg
    alpha = j / hnu                              # cm³/s
    emis  = j * ne * nden_HeIII
    print(f"\n[quick_check] He II {wave}Å at T={Te:.0e} K, ne={ne:.0e} cm⁻³")
    print(f"  j_λ  = h*ν × α_eff = {j:.3e} erg·s⁻¹·cm³")
    print(f"  α_eff^B             = {alpha:.3e} cm³/s")
    print(f"  ε [n(He³⁺)=ne=1]   = {j:.3e} erg/s/cm³")
    return j


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # 기본 파라미터로 실행
    Lumin  = 44.0
    metals = 1.0
    N_H    = 22.0

    quick_check(Te=1e4, ne=1e2)

    result = compare_HeII(
        Lumin=Lumin,
        metals=metals,
        Column_density_order=N_H,
        wave=1640,
        save_fig=os.path.expanduser(
            f'~/RT_JIN/HeII_recomb_L{Lumin}_Z{metals}_N{N_H}.png')
    )
