
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import tifffile
from scipy.optimize import curve_fit


# 1. ENTREES / SORTIES
def load_stack(folder: str | Path, pattern: str = "*.tif*") -> tuple[np.ndarray, list[str]]:
    """
    Charge toutes les images .tif d'un dossier comme une pile 3D (n_images, H, W).

    Parameters
    ----------
    folder : dossier contenant les images .tif (ex: ".../Density/Case 1")
    pattern : motif de recherche des fichiers (par defaut "*.tif*", couvre .tif et .tiff)

    Returns
    -------
    stack : ndarray (n_images, H, W), dtype float64
    filenames : liste des noms de fichiers, dans l'ordre de la pile
    """
    folder = Path(folder)
    files = sorted(folder.glob(pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"Aucune image .tif trouvee dans {folder}")
    imgs = [tifffile.imread(str(f)).astype(np.float64) for f in files]
    shapes = {im.shape for im in imgs}
    if len(shapes) > 1:
        raise ValueError(f"Les images de {folder} n'ont pas toutes la meme taille : {shapes}")
    stack = np.stack(imgs, axis=0)
    return stack, [f.name for f in files]


# 2. FONCTION D'AUTOCORRELATION SPATIALE (ICS), AVEC CORRECTION DE BORD
def spatial_acf(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcule la fonction d'autocorrelation spatiale normalisee d'une image 2D :

        G(xi, eta) = < dI(x,y) * dI(x+xi, y+eta) >_(x,y commun)  /  <I>^2

    avec dI = I - <I>.

    Returns
    -------
    G       : ndarray (2M, 2N) -- ACF, lag (0,0) au centre (apres fftshift)
    overlap : ndarray (2M, 2N) -- nombre de pixels superposes pour chaque lag
    xi      : ndarray (2M,)    -- valeurs de decalage en x (en pixels)
    eta     : ndarray (2N,)    -- valeurs de decalage en y (en pixels)
    """
    image = np.asarray(image, dtype=np.float64)
    M, N = image.shape
    mu = image.mean()
    dI = image - mu

    # Taille de padding = 2x (>= 2*dim - 1) pour eviter tout wrap-around
    Mp, Np = 2 * M, 2 * N

    # numerateur : somme( dI(x,y) * dI(x+xi, y+eta) ) sur le recouvrement
    F = np.fft.fft2(dI, s=(Mp, Np))
    numerator = np.fft.ifft2(F * np.conj(F)).real

    # nombre de pixels communs pour chaque lag (correlation d'un masque de 1)
    mask = np.ones((M, N), dtype=np.float64)
    Fm = np.fft.fft2(mask, s=(Mp, Np))
    overlap = np.fft.ifft2(Fm * np.conj(Fm)).real
    overlap = np.clip(np.round(overlap), 1, None)  # evite division par 0

    G = numerator / (overlap * mu ** 2)

    # recentrer le lag (0,0) au milieu du tableau
    G = np.fft.fftshift(G)
    overlap = np.fft.fftshift(overlap)

    xi = np.arange(-M, M)   # longueur 2M, correspond a l'axe 0 (lignes -> x)
    eta = np.arange(-N, N)  # longueur 2N, correspond a l'axe 1 (colonnes -> y)

    return G, overlap, xi, eta


def average_acf_over_stack(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcule G(xi,eta) pour chaque image d'une pile puis moyenne le tout
    (comme dans Wiseman/Kolin : moyenner plusieurs ACF individuelles
    ameliore le rapport signal/bruit et fait disparaitre les pics de
    correlation aleatoires, cf. Fig. 4 de Kolin & Wiseman 2007).

    Returns
    -------
    Gavg        : ACF moyenne
    Gstack      : ACF individuelles (n_images, 2M, 2N) -- utile pour stats
    overlap     : nombre de pixels communs (identique pour chaque image de meme taille)
    xi, eta     : axes de decalage
    """
    Gs = []
    overlap = xi = eta = None
    for img in stack:
        G, overlap, xi, eta = spatial_acf(img)
        Gs.append(G)
    Gstack = np.stack(Gs, axis=0)
    Gavg = Gstack.mean(axis=0)
    return Gavg, Gstack, overlap, xi, eta


# 3. FENETRAGE CENTRAL + AJUSTEMENT GAUSSIEN 2D
def crop_center(G: np.ndarray, xi: np.ndarray, eta: np.ndarray, overlap: Optional[np.ndarray], half_width: int):
    """
    Extrait une fenetre carree +-half_width pixels autour du lag (0,0).
    """
    ci = int(np.where(xi == 0)[0][0])
    cj = int(np.where(eta == 0)[0][0])
    half_width = int(min(half_width, ci, len(xi) - 1 - ci, cj, len(eta) - 1 - cj))
    half_width = max(half_width, 1)
    sl_i = slice(ci - half_width, ci + half_width + 1)
    sl_j = slice(cj - half_width, cj + half_width + 1)
    Gc = G[sl_i, sl_j]
    xic = xi[sl_i]
    etac = eta[sl_j]
    ovc = overlap[sl_i, sl_j] if overlap is not None else None
    return Gc, xic, etac, ovc


def _gaussian2d(coords, g0, omega, ginf):
    xi, eta = coords
    return g0 * np.exp(-(xi ** 2 + eta ** 2) / omega ** 2) + ginf


@dataclass
class ACFFitResult:
    g0: float
    omega_pix: float
    ginf: float
    perr: np.ndarray = field(repr=False)          # incertitudes (1 sigma) sur [g0, omega, ginf]
    pcov: np.ndarray = field(repr=False)
    n_points: int = 0


def fit_gaussian_acf(G: np.ndarray, xi: np.ndarray, eta: np.ndarray,
                    overlap: Optional[np.ndarray] = None, fit_half_width: Optional[int] = None,
                    exclude_zero_lag: bool = False, min_overlap_frac: float = 0.0) -> ACFFitResult:
    """
    Ajuste G(xi, eta) a une gaussienne 2D isotrope :
        G(xi,eta) = g(0,0) * exp( -(xi^2 + eta^2) / omega^2 ) + g_infini

    Parameters
    ----------
    fit_half_width : ne garder que les lags dans +-fit_half_width pixels
    exclude_zero_lag : si True, exclut le point (0,0) de l'ajustement.
    min_overlap_frac : fraction minimale (0-1) du recouvrement maximal
        requise pour qu'un point de lag soit inclus dans l'ajustement.
    """
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

    # valeur au centre comme estimation initiale de g(0,0)
    ci, cj = np.where((XI == 0) & (ETA == 0))
    g0_guess = float(G[ci, cj][0]) if len(ci) else float(np.nanmax(ydata))
    p0 = [max(g0_guess, 1e-6), 3.0, 0.0]

    try:
        popt, pcov = curve_fit(_gaussian2d, xdata, ydata, p0=p0, maxfev=20000)
    except RuntimeError as e:
        raise RuntimeError(f"L'ajustement gaussien n'a pas converge : {e}")

    perr = np.sqrt(np.diag(pcov))
    return ACFFitResult(g0=popt[0], omega_pix=popt[1], ginf=popt[2],
                        perr=perr, pcov=pcov, n_points=xdata.shape[1])


# 4. QUANTITES DERIVEES (densite de particules)
def density_from_fit(fit: ACFFitResult, pixel_size_um: float) -> dict:
    """
    Convertit les parametres d'ajustement de l'ACF en grandeurs physiques.

    Theorie (Petersen 1993 ; Kolin & Wiseman 2007, Eq. 3-5) :
        g(0,0) = 1 / <n_p>              <n_p> = nb moyen de particules
                                                independantes par "beam area"
        Beam area = pi * omega0^2       (omega0 en unites physiques)
        Densite de particules (CD)  =  <n_p> / (pi * omega0^2)
    """
    omega_um = fit.omega_pix * pixel_size_um
    n_per_BA = 1.0 / fit.g0
    beam_area_um2 = np.pi * omega_um ** 2
    density_um2 = n_per_BA / beam_area_um2
    return dict(omega_um=omega_um, n_per_beam_area=n_per_BA, beam_area_um2=beam_area_um2,
                density_per_um2=density_um2,)