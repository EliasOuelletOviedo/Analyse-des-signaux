from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from ics_batch import discover_dataset
from rics_batch import analyze_diffusion_category, rics_summary_table
from rics_plots import (
    plot_rics_case_overview, plot_diffusion_effect,
    plot_microscope_params_effect, plot_rics_schema,
)


# CONFIGURATION -- A ADAPTER SELON LES DONNEES REELLES
BASE_DIR = Path("simulations")
OUTPUT_DIR = Path("resultats_RICS")

# --- parametres instrumentaux fixes (calibres independamment, cf. Brown
#     et al. 2008 "Solution preparation and system calibration") ---------
W0_UM = 0.2                 # rayon 1/e^2 (xy) de la PSF (um) -- meme valeur que partie A
WZ_UM = None                # rayon 1/e^2 (z) ; None -> 3*W0_UM (gaussienne 3D, Brown 2008)
PIXEL_SIZE_UM = 0.05        # taille de pixel nominale (um/pixel) -- meme valeur que partie A

# !! A VERIFIER/CORRIGER selon le simulateur !!
# Valeurs par defaut plausibles = scan "medium" de Brown et al. 2008 (Table 3 / Methods)
TAU_P_S = 8e-6               # pixel dwell time (s)  (8 us -- scan "slow", recommande)
TAU_L_S = 2.0e-3             # line time (s)         (~2 ms, ordre de grandeur typique)

FIT_HALF_WIDTH = 15
MIN_OVERLAP_FRAC = 0.3
FIT_GINF = False              # ajuster ou non un offset g_infini (defaut : non, cf. modele "propre")


# PIPELINE PRINCIPAL
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base_dir = BASE_DIR
    dataset = discover_dataset(base_dir) if base_dir.exists() else {}
    print("Cas de diffusion trouves :", list(dataset["diffusion"].keys()))

    # Schema conceptuel (independant des donnees)
    plot_rics_schema(tau_p_s=TAU_P_S, tau_l_s=TAU_L_S,
                    savepath=OUTPUT_DIR / "00_schema_balayage_raster.png")

    # Etude theorique de l'effet des parametres du microscope (independante
    # des donnees -- repond directement a la question de discussion)
    plot_microscope_params_effect(
        D_um2_s=10.0, D_fast_um2_s=100.0, w0_um=W0_UM, pixel_size_um=PIXEL_SIZE_UM,
        tau_p_ref=TAU_P_S, tau_l_ref=TAU_L_S,
        savepath=OUTPUT_DIR / "00_effet_theorique_parametres_microscope.png",
    )

    # B.1 Analyse RICS de chaque cas D = ...
    print("\n=== Analyse RICS : effet du coefficient de diffusion D ===")
    res = analyze_diffusion_category(
        dataset["diffusion"], w0_um=W0_UM, pixel_size_um=PIXEL_SIZE_UM,
        tau_p=TAU_P_S, tau_l=TAU_L_S, wz_um=WZ_UM,
        fit_half_width=FIT_HALF_WIDTH, min_overlap_frac=MIN_OVERLAP_FRAC,
        fit_ginf=FIT_GINF,
    )

    for case, r in res.items():
        plot_rics_case_overview(
            r, f"RICS - {case}",
            savepath=OUTPUT_DIR / f"01_rics_overview_{case.replace('=', '_')}.png")

    plot_diffusion_effect(res, savepath=OUTPUT_DIR / "01_rics_diffusion_effect_summary.png")

    df = rics_summary_table(res)
    df.to_csv(OUTPUT_DIR / "01_rics_diffusion_summary.csv", index=False)
    print(df.to_string(index=False))

    print(f"\nTermine. Figures et tableaux ecrits dans : {OUTPUT_DIR.resolve()}")
    return df


if __name__ == "__main__":
    main()