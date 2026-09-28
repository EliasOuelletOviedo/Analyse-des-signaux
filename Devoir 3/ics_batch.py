from __future__ import annotations
from pathlib import Path
from typing import Optional
import re

import numpy as np
import pandas as pd

from ics_core import (load_stack, average_acf_over_stack, fit_gaussian_acf, density_from_fit, ACFFitResult)

# Categories utilisees par la partie A (ICS). "Diffusion" est pour RICS (partie B).
ICS_CATEGORIES = ["density", "image size", "imagesize", "noise", "pixel size", "pixelsize"]


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def discover_dataset(base_dir: str | Path) -> dict[str, dict[str, Path]]:
    """
    Parcourt base_dir et renvoie un dictionnaire :
        {"density": {"Case 1": Path(...), "Case 2": Path(...), ...},
        "image size": {"10 BAs": Path(...), ...},
        "noise": {...},
        "pixel size": {...},
        "diffusion": {...},   # present mais non utilise en partie A
        }
    Seuls les dossiers contenant directement des fichiers .tif sont retenus
    comme "cas".
    """
    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(f"Le dossier de base {base} n'existe pas.")

    dataset: dict[str, dict[str, Path]] = {}
    for category_dir in sorted(base.iterdir()):
        if not category_dir.is_dir():
            continue
        cases: dict[str, Path] = {}
        for case_dir in sorted(category_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            if list(case_dir.glob("*.tif")) or list(case_dir.glob("*.tiff")):
                cases[case_dir.name] = case_dir
        if cases:
            dataset[_norm(category_dir.name)] = cases
    return dataset


def analyze_condition(folder: str | Path, pixel_size_um: float, fit_half_width: int = 15,
                        min_overlap_frac: float = 0.3, exclude_zero_lag: bool = False) -> dict:
    """
    Analyse ICS complete d'un dossier "cas" (pile de N images repetees
    d'une meme condition de simulation).

    Retourne un dictionnaire contenant :
        - Gavg, Gstack, overlap, xi, eta  : ACF moyenne / individuelles
        - fit_avg      : ACFFitResult sur l'ACF moyennee (meilleure estimation)
        - fit_each     : liste d'ACFFitResult, un par image (pour les stats)
        - g0_mean/std, omega_um_mean/std, density_mean/std : statistiques inter-images (repetitions), utiles pour les barres d'erreur
        - n_frames, image_shape
    """
    stack, filenames = load_stack(folder)
    Gavg, Gstack, overlap, xi, eta = average_acf_over_stack(stack)

    fit_avg = fit_gaussian_acf(Gavg, xi, eta, overlap, fit_half_width=fit_half_width,
                                min_overlap_frac=min_overlap_frac, exclude_zero_lag=exclude_zero_lag)
    dens_avg = density_from_fit(fit_avg, pixel_size_um)

    fit_each: list[ACFFitResult] = []
    dens_each: list[dict] = []
    for G in Gstack:
        try:
            f = fit_gaussian_acf(G, xi, eta, overlap, fit_half_width=fit_half_width,
                                min_overlap_frac=min_overlap_frac, exclude_zero_lag=exclude_zero_lag)
            fit_each.append(f)
            dens_each.append(density_from_fit(f, pixel_size_um))
        except RuntimeError:
            continue

    g0_arr = np.array([f.g0 for f in fit_each])
    omega_arr = np.array([d["omega_um"] for d in dens_each])
    density_arr = np.array([d["density_per_um2"] for d in dens_each])
    nBA_arr = np.array([d["n_per_beam_area"] for d in dens_each])

    return dict(
        folder=Path(folder), filenames=filenames,
        stack=stack, Gavg=Gavg, Gstack=Gstack, overlap=overlap, xi=xi, eta=eta,
        fit_avg=fit_avg, dens_avg=dens_avg,
        fit_each=fit_each, dens_each=dens_each,
        g0_mean=g0_arr.mean(), g0_std=g0_arr.std(ddof=1) if len(g0_arr) > 1 else 0.0,
        omega_um_mean=omega_arr.mean(), omega_um_std=omega_arr.std(ddof=1) if len(omega_arr) > 1 else 0.0,
        density_mean=density_arr.mean(), density_std=density_arr.std(ddof=1) if len(density_arr) > 1 else 0.0,
        n_per_BA_mean=nBA_arr.mean(), n_per_BA_std=nBA_arr.std(ddof=1) if len(nBA_arr) > 1 else 0.0,
        n_frames=stack.shape[0], image_shape=stack.shape[1:],
        pixel_size_um=pixel_size_um,
    )


def analyze_category(cases: dict[str, Path], pixel_size_um: float | dict[str, float] = 0.05,
                    **kwargs) -> dict[str, dict]:
    """
    Applique analyze_condition() a tous les "cas" d'une categorie
    (ex : toutes les densites, toutes les tailles d'image...).

    pixel_size_um peut etre soit une valeur unique (utilisee pour tous
    les cas), soit un dict {case_name: pixel_size_um} -- utile pour la
    categorie "pixel size" ou la taille de pixel change d'un cas a l'autre.
    """
    results = {}
    for case_name, folder in cases.items():
        ps = pixel_size_um[case_name] if isinstance(pixel_size_um, dict) else pixel_size_um
        results[case_name] = analyze_condition(folder, pixel_size_um=ps, **kwargs)
    return results


def summary_table(results: dict[str, dict], nominal: Optional[dict[str, float]] = None,
                nominal_key: str = "n_per_BA_mean") -> pd.DataFrame:
    """
    Construit un tableau recapitulatif (pandas DataFrame) pour une categorie
    analysee avec analyze_category().

    nominal : dict optionnel {case_name: valeur_nominale} pour calculer
            une erreur relative (%) par rapport a la valeur attendue.
    """
    rows = []
    for case_name, r in results.items():
        row = dict(
            cas=case_name,
            n_images=r["n_frames"],
            taille_image=f"{r['image_shape'][0]}x{r['image_shape'][1]}",
            pixel_size_um=r["pixel_size_um"],
            g0=r["fit_avg"].g0,
            omega_um=r["dens_avg"]["omega_um"],
            n_par_BA=r["dens_avg"]["n_per_beam_area"],
            n_par_BA_std_repet=r["n_per_BA_std"],
            densite_um2=r["dens_avg"]["density_per_um2"],
            densite_std_repet=r["density_std"],
        )
        if nominal is not None and case_name in nominal:
            nom = nominal[case_name]
            row["nominal"] = nom
            row["erreur_%"] = 100.0 * (row[nominal_key.replace("_mean", "") if False else "n_par_BA"] - nom) / nom
        rows.append(row)
    df = pd.DataFrame(rows)
    return df