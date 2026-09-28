from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import curve_fit

from ics_core import crop_center


# 1. MODELE THEORIQUE RICS
def rics_scan_term(xi, eta, D, w0_um, pixel_size_um, tau_p, tau_l):
    """S(xi,eta) : terme du a la structure temporelle du balayage.

    Convention d'axes (heritee de ics_core.spatial_acf, image.shape=(M,N)) :
        xi  = lag le long de l'AXE 0 (lignes/rangees)    -> axe LENT  (tau_l)
        eta = lag le long de l'AXE 1 (colonnes)          -> axe RAPIDE (tau_p)
    """
    denom = 1.0 + ((4.0 * D * (tau_l * np.abs(xi) + tau_p * np.abs(eta))) / w0_um ** 2)
    # num = (pixel_size_um ** 2 / w0_um ** 2) * (xi ** 2 + eta ** 2)
    num = (pixel_size_um * np.abs(xi)/ w0_um)**2 + (np.abs(eta) * pixel_size_um / w0_um)**2
    return np.exp(-num / denom)


def rics_diffusion_term(xi, eta, D, N, w0_um, wz_um, tau_p, tau_l, gamma=0.3535):
    """G(xi,eta) : terme de diffusion classique (identique a l'ICS/FCS),
    mais dont l'argument temporel depend maintenant du LAG (xi,eta) via
    la structure du balayage : t = tau_l*|xi| + tau_p*|eta|
    (xi = axe lent/lignes, eta = axe rapide/colonnes -- cf. rics_scan_term)."""
    arg = 4.0 * D * (tau_l * np.abs(xi) + tau_p * np.abs(eta))
    term_xy = (1.0 + (arg / w0_um ** 2)) ** -1
    term_z = (1.0 + (arg / wz_um ** 2)) ** -0.5
    return (gamma / N) * term_xy * term_z


def rics_model(xi, eta, D, N, w0_um, pixel_size_um, tau_p, tau_l, wz_um=None, gamma=0.3535, ginf=0.0):
    """Fonction de correlation RICS complete : G_s(xi,eta) = S*G + g_inf."""
    if wz_um is None:
        wz_um = 3.0 * w0_um
    S = rics_scan_term(xi, eta, D, w0_um, pixel_size_um, tau_p, tau_l)
    G = rics_diffusion_term(xi, eta, D, N, w0_um, wz_um, tau_p, tau_l, gamma)
    return S * G + ginf


# 2. AJUSTEMENT : on fixe les parametres INSTRUMENTAUX (connus/calibres :
#    w0, wz, taille de pixel, tau_p, tau_l) et on ajuste D et N (et optionnellement un offset g_inf).
@dataclass
class RICSFitResult:
    D: float
    N: float
    ginf: float
    perr: np.ndarray = field(repr=False)
    pcov: np.ndarray = field(repr=False)
    n_points: int = 0
    fixed_params: dict = field(default_factory=dict, repr=False)


def fit_rics(G: np.ndarray, xi: np.ndarray, eta: np.ndarray, w0_um: float, pixel_size_um: float, tau_p: float,
            tau_l: float, wz_um: Optional[float] = None, gamma: float = 0.3535, overlap: Optional[np.ndarray] = None,
            fit_half_width: Optional[int] = None, min_overlap_frac: float = 0.0, fit_ginf: bool = True,
            exclude_zero_lag: bool = True, D_guess: float = 1.0, N_guess: Optional[float] = None) -> RICSFitResult:
    """
    Ajuste G(xi,eta) (mesuree par ics_core.spatial_acf, moyennee sur la pile d'images) au modele RICS
    complet pour en extraire D et N.

    Les parametres instrumentaux (w0_um, pixel_size_um, tau_p, tau_l, wz_um, gamma) sont maintenus FIXES 
    (comme lors d'un vrai ajustement RICS : ils sont normalement calibres independamment, cf. Brown et
    al. 2008, section "Solution preparation and system calibration").
    """
    if wz_um is None:
        wz_um = 3.0 * w0_um

    if fit_half_width is not None:
        G, xi, eta, overlap = crop_center(G, xi, eta, overlap, fit_half_width)

    XI, ETA = np.meshgrid(xi, eta, indexing="ij")
    keep = np.ones_like(G, dtype=bool)
    if overlap is not None and min_overlap_frac > 0:
        keep &= overlap >= (min_overlap_frac * overlap.max())
    if exclude_zero_lag:
        keep &= ~((XI == 0) & (ETA == 0))

    xdata = np.vstack([XI[keep].ravel(), ETA[keep].ravel()])
    ydata = G[keep].ravel()

    if N_guess is None:
        ci, cj = np.where((XI == 0) & (ETA == 0))
        g0_center = float(G[ci, cj][0]) if len(ci) else float(np.nanmax(ydata))
        N_guess = max(gamma / max(g0_center, 1e-6), 1e-3)

    def _model_fit(coords, D, N, ginf):
        xi_, eta_ = coords
        return rics_model(xi_, eta_, D, N, w0_um, pixel_size_um, tau_p, tau_l,
                            wz_um=wz_um, gamma=gamma, ginf=ginf)

    def _model_fit_no_offset(coords, D, N):
        xi_, eta_ = coords
        return rics_model(xi_, eta_, D, N, w0_um, pixel_size_um, tau_p, tau_l,
                            wz_um=wz_um, gamma=gamma, ginf=0.0)

    bounds_lo = [1e-4, 1e-6]
    bounds_hi = [1e4, 1e6]
    if fit_ginf:
        p0 = [D_guess, N_guess, 0.0]
        popt, pcov = curve_fit(_model_fit, xdata, ydata, p0=p0, maxfev=40000,
                                bounds=(bounds_lo + [-1.0], bounds_hi + [1.0]))
        D_fit, N_fit, ginf_fit = popt
    else:
        p0 = [D_guess, N_guess]
        popt, pcov = curve_fit(_model_fit_no_offset, xdata, ydata, p0=p0, maxfev=40000,
                                bounds=(bounds_lo, bounds_hi))
        D_fit, N_fit = popt
        ginf_fit = 0.0

    perr = np.sqrt(np.diag(pcov))
    fixed = dict(w0_um=w0_um, wz_um=wz_um, pixel_size_um=pixel_size_um,
                tau_p=tau_p, tau_l=tau_l, gamma=gamma)
    return RICSFitResult(D=D_fit, N=N_fit, ginf=ginf_fit, perr=perr, pcov=pcov,
                        n_points=xdata.shape[1], fixed_params=fixed)
