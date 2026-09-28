from __future__ import annotations
from pathlib import Path
import re

import numpy as np
import pandas as pd

from ics_core import load_stack, average_acf_over_stack
from rics_core import fit_rics, RICSFitResult


def extract_D_from_case_name(name: str) -> float:
    """Extrait la valeur numerique de D d'un nom de dossier tel que 'D=0.1'."""
    m = re.search(r"[\d.]+", name)
    return float(m.group()) if m else np.nan


def analyze_diffusion_condition(folder: str | Path, w0_um: float, pixel_size_um: float,
                                tau_p: float, tau_l: float, wz_um: float | None = None,
                                fit_half_width: int = 15, min_overlap_frac: float = 0.3,
                                fit_ginf: bool = False, D_guess: float = 1.0) -> dict:
    """
    Analyse RICS complete d'un dossier "cas" (une valeur de D, N images
    repetees). Calcule G(xi,eta) (moyennee + individuelles) puis ajuste
    le modele RICS complet (S*G) pour en extraire D et N.
    """
    stack, filenames = load_stack(folder)
    Gavg, Gstack, overlap, xi, eta = average_acf_over_stack(stack)

    hw = min(fit_half_width, stack.shape[1] // 2 - 1, stack.shape[2] // 2 - 1)

    fit_avg = fit_rics(Gavg, xi, eta, w0_um=w0_um, pixel_size_um=pixel_size_um,
                        tau_p=tau_p, tau_l=tau_l, wz_um=wz_um, overlap=overlap,
                        fit_half_width=hw, min_overlap_frac=min_overlap_frac,
                        fit_ginf=fit_ginf, D_guess=D_guess)

    fit_each: list[RICSFitResult] = []
    for G in Gstack:
        try:
            f = fit_rics(G, xi, eta, w0_um=w0_um, pixel_size_um=pixel_size_um,
                        tau_p=tau_p, tau_l=tau_l, wz_um=wz_um, overlap=overlap,
                        fit_half_width=hw, min_overlap_frac=min_overlap_frac,
                        fit_ginf=fit_ginf, D_guess=D_guess)
            fit_each.append(f)
        except RuntimeError:
            continue

    D_arr = np.array([f.D for f in fit_each])
    N_arr = np.array([f.N for f in fit_each])

    return dict(
        folder=Path(folder), filenames=filenames, stack=stack,
        Gavg=Gavg, Gstack=Gstack, overlap=overlap, xi=xi, eta=eta,
        fit_avg=fit_avg, fit_each=fit_each,
        D_mean=D_arr.mean() if len(D_arr) else np.nan,
        D_std=D_arr.std(ddof=1) if len(D_arr) > 1 else 0.0,
        N_mean=N_arr.mean() if len(N_arr) else np.nan,
        N_std=N_arr.std(ddof=1) if len(N_arr) > 1 else 0.0,
        n_frames=stack.shape[0], image_shape=stack.shape[1:],
        w0_um=w0_um, pixel_size_um=pixel_size_um, tau_p=tau_p, tau_l=tau_l,
    )


def analyze_diffusion_category(cases: dict[str, Path], w0_um: float, pixel_size_um: float,
                                tau_p: float, tau_l: float, **kwargs) -> dict[str, dict]:
    """Applique analyze_diffusion_condition() a tous les cas D=... d'une categorie."""
    results = {}
    for case_name, folder in cases.items():
        # meilleure estimation initiale de D = valeur nominale lue du nom de dossier
        D_guess = extract_D_from_case_name(case_name)
        if not np.isfinite(D_guess) or D_guess <= 0:
            D_guess = 1.0
        results[case_name] = analyze_diffusion_condition(
            folder, w0_um=w0_um, pixel_size_um=pixel_size_um,
            tau_p=tau_p, tau_l=tau_l, D_guess=D_guess, **kwargs)
    return results


def rics_summary_table(results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for case_name, r in results.items():
        D_nominal = extract_D_from_case_name(case_name)
        D_fit = r["fit_avg"].D
        row = {
            "cas": case_name,
            "D_nominal_um2_s": D_nominal,
            "D_fit_um2_s": D_fit,
            "D_fit_std_repet": r["D_std"],
            "erreur_%": 100.0 * (D_fit - D_nominal) / D_nominal if D_nominal else np.nan,
            "N_fit": r["fit_avg"].N,
            "n_images": r["n_frames"],
            "taille_image": f"{r['image_shape'][0]}x{r['image_shape'][1]}",
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    if "D_nominal_um2_s" in df:
        df = df.sort_values("D_nominal_um2_s").reset_index(drop=True)
    return df