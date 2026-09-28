from __future__ import annotations
from pathlib import Path
from scipy.stats import norm

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ics_core import crop_center, _gaussian2d


# Style global sobre (lisible en niveaux de gris / impression)-
plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
plt.style.use('https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-basic.mplstyle')


# 1. Vue d'ensemble d'un cas unique : image + ACF 2D + surface 3D + profils
def plot_case_overview(result: dict, title: str, savepath: str | Path | None = None, half_width_display: int = 15):
    """
    result : sortie de ics_batch.analyze_condition()
    """
    Gavg, xi, eta = result["Gavg"], result["xi"], result["eta"]
    overlap = result["overlap"]
    fit = result["fit_avg"]
    ps = result["pixel_size_um"]

    Gc, xic, etac, _ = crop_center(Gavg, xi, eta, overlap, half_width_display)

    fig = plt.figure(figsize=(13, 8.5))
    fig.suptitle(title, fontsize=22, fontweight="bold")

    # (a) image representative (1ere image de la pile)
    ax1 = fig.add_subplot(2, 3, 1)
    im0 = result["stack"][0]
    ax1.imshow(im0, cmap="inferno")
    ax1.set_title("Image simulee (frame 1)", fontsize=17, pad=13)
    ax1.set_xlabel("x (pixels)"); ax1.set_ylabel("y (pixels)")
    ax1.text(-0.35, 1.1, "A", transform=ax1.transAxes, fontsize=16, fontweight="bold", va="bottom", ha="right")

    # (b) ACF 2D (fenetre centrale)
    ax2 = fig.add_subplot(2, 3, 2)
    extent = [etac.min(), etac.max(), xic.max(), xic.min()]
    im2 = ax2.imshow(Gc, cmap="inferno", extent=extent)
    ax2.set_title(r"$G(\xi,\eta)$ (moyenne des images)", fontsize=17, pad=18)
    ax2.set_xlabel(r"$\eta$ (pixels)"); ax2.set_ylabel(r"$\xi$ (pixels)")
    ax2.text(-0.35, 1.1, "B", transform=ax2.transAxes, fontsize=16, fontweight="bold", va="bottom", ha="right")
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    # (c) surface 3D de l'ACF (comme Fig. 2B de Kolin & Wiseman 2007)
    ax3 = fig.add_subplot(2, 3, 3, projection="3d")
    XI, ETA = np.meshgrid(xic, etac, indexing="ij")
    ax3.plot_surface(XI, ETA, Gc, cmap="inferno", linewidth=0, antialiased=True)
    ax3.set_title(r"Surface $G(\xi,\eta)$", fontsize=17, pad=13)
    ax3.set_xlabel(r"$\xi$"); ax3.set_ylabel(r"$\eta$"); ax3.set_zlabel("G")
    ax2.text(-0.2, 1.1, "C", transform=ax3.transAxes, fontsize=16, fontweight="bold", va="bottom", ha="right")

    # (d) profil horizontal (eta=0) avec ajustement
    ax4 = fig.add_subplot(2, 3, 4)
    j0 = int(np.where(etac == 0)[0][0])
    prof_h = Gc[:, j0]
    fit_h = _gaussian2d((xic, np.zeros_like(xic)), fit.g0, fit.omega_pix, fit.ginf)
    ax4.plot(xic, fit_h, "-", color='#c5093efd', lw=2, label="ajustement gaussien")
    ax4.plot(xic, prof_h, "o", color='black', ms=4, label="données")
    ax4.axhline(fit.ginf, color="gray", ls="--", lw=1, label=r"$g_\infty$")
    ax4.set_xlabel(r"$\xi$ (pixels)"); ax4.set_ylabel(r"$G(\xi, 0)$")
    ax4.set_title("Profil horizontal", fontsize=17, pad=13)
    ax4.legend(fontsize=8)
    ax4.text(-0.23, 1.1, "D", transform=ax4.transAxes, fontsize=16, fontweight="bold", va="bottom", ha="right")

    # (e) profil vertical (xi=0) avec ajustement
    ax5 = fig.add_subplot(2, 3, 5)
    i0 = int(np.where(xic == 0)[0][0])
    prof_v = Gc[i0, :]
    fit_v = _gaussian2d((np.zeros_like(etac), etac), fit.g0, fit.omega_pix, fit.ginf)
    ax5.plot(etac, fit_v, "-", lw=2, color="#c5093efd", label="ajustement gaussien")
    ax5.plot(etac, prof_v, "o", ms=4, color="black", label="données")
    ax5.set_xlabel(r"$\eta$ (pixels)"); ax5.set_ylabel(r"$G(0, \eta)$")
    ax5.set_title("Profil vertical", fontsize=17, pad=13)
    ax5.legend(fontsize=8)
    ax5.text(-0.23, 1.1, "E", transform=ax5.transAxes, fontsize=16, fontweight="bold", va="bottom", ha="right")

    # # (f) texte resultats numeriques
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")

    dens = result["dens_avg"]

    data = [
        [r"$g(0,0)$", f"{fit.g0:.4f} ± {result['g0_std']:.4f}"],
        [r"$\omega_0$ (pixels)", f"{fit.omega_pix:.3f}"],
        [r"$\omega_0$ ($\mu$m)", f"{dens['omega_um']:.4f} ± {result['omega_um_std']:.4f}"],
        [r"$g_\infty$", f"{fit.ginf:.5f}"],
        ["", ""],
        [r"$\langle n_p\rangle$ (part./BA)",
        f"{dens['n_per_beam_area']:.2f} ± {result['n_per_BA_std']:.2f}"],
        ["Densité (part./µm²)",
        f"{dens['density_per_um2']:.1f} ± {result['density_std']:.1f}"],
        ["", ""],
        ["Taille image", str(result["image_shape"])],
        ["Pixel size", f"{ps} µm/pixel"],
        ["N images moyennées", str(result["n_frames"])],
    ]

    table = ax6.table(cellText=data, colLabels=["Grandeur", "Valeur"], cellLoc="left", colLoc="left", loc="center", colWidths=[0.62, 0.38])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.55)

    # Retirer les bordures
    for cell in table.get_celld().values():
        cell.set_linewidth(0)

    ax6.set_title("Résultats", fontsize=17, pad=13)
    ax6.text(-0.1, 1.1, "F", transform=ax6.transAxes, fontsize=16, fontweight="bold", va="bottom", ha="right")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return fig

# 2. Effet de la densite de particules
def plot_density_effect(results: dict[str, dict], nominal: dict[str, float] | None = None,
                        savepath: str | Path | None = None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    color = ["#7b6ce9ff", "#3d2bc4ff", "#0d044dff"]
    cases = list(results.keys())

    # (a) profils horizontaux superposes
    ax = axes[0]
    for s, case in enumerate(cases):
        r = results[case]
        Gc, xic, etac, _ = crop_center(r["Gavg"], r["xi"], r["eta"], r["overlap"], 15)
        j0 = int(np.where(etac == 0)[0][0])
        ax.plot(xic, Gc[:, j0], "-", ms=3, color=color[s], label=case, alpha=0.9)
    ax.set_xlabel(r"$\xi$ (pixels)"); ax.set_ylabel(r"$G(\xi,0)$")
    ax.set_title("Effet de la densite sur la forme de G", fontsize=16, pad=13)
    ax.legend(fontsize=13)
    ax.text(-0.16, 1.05, "A", transform=ax.transAxes, fontsize=18, fontweight="bold", va="bottom", ha="right")

    # (b) densite moyenne par cas, barre d'erreur = ecart-type
    ax = axes[1]
    means = np.array([results[c]["dens_avg"]["n_per_beam_area"] for c in cases])
    stds = np.array([results[c]["n_per_BA_std"] for c in cases])
    x = np.arange(len(cases))
    ax.bar(x, means, yerr=stds, capsize=5, color=color[:len(cases)], alpha=0.85, error_kw=dict(lw=1.5, ecolor="black"))
    if nominal:
        nom = [nominal.get(c, np.nan) for c in cases]
        ax.plot(x, nom, "s--", color="gray", label="nominale")
        ax.legend(fontsize=13)
    ax.set_xticks(x); ax.set_xticklabels(cases, rotation=20)
    ax.set_ylabel(r"$n_{par}$ / aire de faisceau")
    ax.set_title("Densité mesurée", fontsize=16, pad=13)
    ax.text(-0.16, 1.05, "B", transform=ax.transAxes, fontsize=18, fontweight="bold", va="bottom", ha="right")

    # (c) precision relative (ecart-type / moyenne) par cas
    ax = axes[2]
    rel_precision = 100 * stds / means
    ax.bar(x, rel_precision, color=color[:len(cases)], alpha=0.85)
    for xi_, v in zip(x, rel_precision):
        ax.text(xi_, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(cases, rotation=20)
    ax.set_ylabel("Écart-type relatif (%)")
    ax.set_title("Précision de la mesure de densité", fontsize=16, pad=13)
    ax.text(-0.16, 1.05, "C", transform=ax.transAxes, fontsize=18, fontweight="bold", va="bottom", ha="right")

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return fig

# 3. Effet de la taille de l'image (nombre de "beam areas")
def _extract_number(label: str) -> float:
    import re
    m = re.search(r"[\d.]+", label)
    return float(m.group()) if m else np.nan

def plot_imagesize_effect(results: dict[str, dict], nominal_n_per_BA: float | None = None, savepath: str | Path | None = None):
    cases = sorted(results.keys(), key=_extract_number)
    n_BAs = np.array([_extract_number(c) for c in cases])
    color = ["#aea4f4ff", "#7b6ce9ff", "#3d2bc4ff", "#0d044dff", "#030018ff"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) ACF (profil horizontal) pour chaque taille -> qualite de la decroissance
    ax = axes[0]
    for s, case in enumerate(cases):
        r = results[case]
        Gc, xic, etac, _ = crop_center(r["Gavg"], r["xi"], r["eta"], r["overlap"], 15)
        j0 = int(np.where(etac == 0)[0][0])
        ax.plot(xic, Gc[:, j0], "-", color=color[s], ms=3, label=case)
    # ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(r"$\xi$ (pixels)"); ax.set_ylabel(r"$G(\xi,0)$")
    ax.set_title(r"Effet de la taille d'image sur $G(\xi,0)$", fontsize=16, pad=13)
    ax.legend(fontsize=13)
    ax.text(-0.16, 1.05, "A", transform=ax.transAxes, fontsize=18, fontweight="bold", va="bottom", ha="right")

    # (b) precision de g(0,0) vs taille, avec points colores + tendance theorique 1/sqrt(N)
    ax = axes[1]
    g0_std = np.array([results[c]["g0_std"] for c in cases])
    g0_mean = np.array([results[c]["g0_mean"] for c in cases])
    cv = 100 * g0_std / g0_mean
    for s, (n, c) in enumerate(zip(n_BAs, cv)):
        ax.plot(n, c, "o", ms=9, color=color[s], zorder=3)
    ax.plot(n_BAs, cv, "", color="lightgray", lw=1.2, zorder=1)
    # reference theorique : CV ~ 1/sqrt(N), normalisee au premier point
    ref = cv[0] * np.sqrt(n_BAs[0] / n_BAs)
    ax.plot(n_BAs, ref, "k--", lw=1, label=r"tendance $\propto 1/\sqrt{N}$", zorder=2)
    ax.set_xlabel("Taille de l'image (nb. beam areas)")
    # ax.set_ylabel("Coefficient de variation de g(0,0) (%)")
    ax.set_ylabel(r"CV[$g(0,0)$] [%]")
    ax.set_title("Precision vs taille d'image", fontsize=16, pad=13)
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.legend(fontsize=11)
    ax.text(-0.16, 1.05, "B", transform=ax.transAxes, fontsize=18, fontweight="bold", va="bottom", ha="right")

    # (c) exactitude : densite mesuree (+/- ecart-type) vs taille, en barres
    ax = axes[2]
    dens_mean = np.array([results[c]["n_per_BA_mean"] for c in cases])
    dens_std = np.array([results[c]["n_per_BA_std"] for c in cases])
    x = np.arange(len(cases))
    ax.bar(x, dens_mean, yerr=dens_std, capsize=4, color=color[:len(cases)], alpha=0.85, error_kw=dict(lw=1.5, ecolor="black"))
    if nominal_n_per_BA is not None:
        ax.axhline(nominal_n_per_BA, color="gray", ls="--", lw=1.2, label="Valeur nominale")
        ax.legend(fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(cases, rotation=20)
    ax.set_ylabel("Densité [particules/BA]")
    ax.set_title("Exactitude vs taille d'image", fontsize=16, pad=13)
    ax.text(-0.16, 1.05, "C", transform=ax.transAxes, fontsize=18, fontweight="bold", va="bottom", ha="right")

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return fig

# 4. Effet du bruit du detecteur (shot noise)
def plot_noise_effect(results: dict[str, dict], nominal_n_per_BA: float = 10.0,
                        savepath: str | Path | None = None):
    cases = list(results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    color = ["#7b6ce9ff", "#3d2bc4ff", "#0d044dff"]
    color2 = ["#bc4675ff", "#a01c51ff", "#4d0421ff"]
    x = np.arange(len(cases))
    width = 0.35

    # (a) profils horizontaux zoomes pres du centre -> pic de shot noise au lag (0,0)
    ax = axes[0]
    for s, case in enumerate(cases):
        r = results[case]
        Gc, xic, etac, _ = crop_center(r["Gavg"], r["xi"], r["eta"], r["overlap"], 6)
        j0 = int(np.where(etac == 0)[0][0])
        ax.plot(xic, Gc[:, j0], "-", color=color[s], ms=4, label=case)
    ax.set_xlabel(r"$\xi$ (pixels)"); ax.set_ylabel(r"$G(\xi,0)$")
    ax.set_title(r"Effet du bruit sur $G(\xi,0)$", fontsize=16, pad=13)
    ax.legend(fontsize=13)
    ax.text(-0.16, 1.05, "A", transform=ax.transAxes, fontsize=18, fontweight="bold", va="bottom", ha="right")

    # (b) g(0,0) avec vs sans le point (0,0) inclus dans l'ajustement
    from ics_core import fit_gaussian_acf
    g0_incl = np.array([results[c]["fit_avg"].g0 for c in cases])
    g0_incl_std = np.array([results[c]["g0_std"] for c in cases])
    g0_excl = []
    for c in cases:
        r = results[c]
        f = fit_gaussian_acf(r["Gavg"], r["xi"], r["eta"], r["overlap"],
                                fit_half_width=15, min_overlap_frac=0.3, exclude_zero_lag=True)
        g0_excl.append(f.g0)
    g0_excl = np.array(g0_excl)

    ax = axes[1]
    ax.bar(x - width/2, g0_incl, width, yerr=g0_incl_std, capsize=4, alpha=0.85,
            color=color[:len(cases)], label="(0,0) inclus",
            error_kw=dict(lw=1.3, ecolor="black"))
    ax.bar(x + width/2, g0_excl, width, color=color2[:len(cases)], alpha=0.85, label="(0,0) exclu")
    ax.set_xticks(x); ax.set_xticklabels(cases, rotation=20)
    ax.set_ylabel(r"$g(0,0)$ ajusté")
    ax.set_title("Biais du shot noise sur g(0,0)", fontsize=16, pad=13)
    ax.legend(fontsize=13)
    ax.text(-0.16, 1.05, "B", transform=ax.transAxes, fontsize=18, fontweight="bold", va="bottom", ha="right")

    # (c) densite mesuree vs niveau de bruit (inclus vs exclu), meme mise en forme que B
    dens_incl = np.array([results[c]["dens_avg"]["n_per_beam_area"] for c in cases])
    dens_incl_std = np.array([results[c]["n_per_BA_std"] for c in cases])
    k = dens_incl * g0_incl          # facteur de conversion densite = k / g0, deduit du cas inclus
    dens_excl = k / g0_excl

    ax = axes[2]
    ax.bar(x - width/2, dens_incl, width, yerr=dens_incl_std, capsize=4,
            color=color[:len(cases)], alpha=0.85, label="(0,0) inclus",
            error_kw=dict(lw=1.3, ecolor="black"))
    ax.bar(x + width/2, dens_excl, width, color=color2[:len(cases)], alpha=0.85, label="(0,0) exclu")
    ax.axhline(nominal_n_per_BA, color="gray", ls="--", lw=1.2, label="valeur nominale (10)")
    ax.set_xticks(x); ax.set_xticklabels(cases, rotation=20)
    ax.set_ylabel("Densité [particules/BA]")
    ax.set_title("Densité estimée vs niveau de bruit", fontsize=16, pad=13)
    ax.legend(fontsize=13)
    ax.text(-0.16, 1.05, "C", transform=ax.transAxes, fontsize=18, fontweight="bold", va="bottom", ha="right")

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return fig

# 5. Effet de la taille du pixel (sous-echantillonnage de la PSF)
def plot_pixelsize_effect(results: dict[str, dict], nominal_omega_um: float | None = None,
                        nominal_n_per_BA: float | None = None,
                        savepath: str | Path | None = None):
    cases = sorted(results.keys(), key=lambda c: results[c]["pixel_size_um"])
    px = np.array([results[c]["pixel_size_um"] for c in cases])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    color = ["#a8a0ebff", "#6557cfff", "#3d2bc4ff", "#0d044dff"]

    # (a) ACF en unites physiques (µm) -> compare la largeur reelle
    #     marqueurs conserves pour visualiser le nombre de points echantillonnant le pic
    ax = axes[0]
    for s, case in enumerate(cases):
        r = results[case]
        ps = r["pixel_size_um"]
        Gc, xic, etac, _ = crop_center(r["Gavg"], r["xi"], r["eta"], r["overlap"],
                                        min(15, r["Gavg"].shape[0] // 2 - 1))
        j0 = int(np.where(etac == 0)[0][0])
        ax.plot(xic * ps, Gc[:, j0], "-", color=color[s], ms=4, label=f"{ps} µm/pix")
    # if nominal_omega_um is not None:
    #     ax.axvspan(-nominal_omega_um, nominal_omega_um, color="gray", alpha=0.1, label=r"rayon PSF ($\omega_0$)")
    ax.set_xlabel(r"$\xi$ (µm)"); ax.set_ylabel(r"$G(\xi,0)$")
    ax.set_title(r"Effet de la taille des pixels sur $G(\xi,0)$", fontsize=16, pad=13)
    ax.legend(fontsize=13)
    ax.text(-0.16, 1.05, "A", transform=ax.transAxes, fontsize=18, fontweight="bold", va="bottom", ha="right")

    # (b) omega mesure (µm) vs ratio pixel/PSF -> montre le seuil de sous-echantillonnage
    ax = axes[1]
    omega_mean = np.array([results[c]["omega_um_mean"] for c in cases])
    omega_std = np.array([results[c]["omega_um_std"] for c in cases])
    if nominal_omega_um is not None:
        ratio = px / nominal_omega_um
        for s, (rv, om, oe) in enumerate(zip(ratio, omega_mean, omega_std)):
            ax.errorbar(rv, om, yerr=oe, fmt="o", ms=6, capsize=4, color=color[s], zorder=3)
        ax.plot(ratio, omega_mean, "-", color="lightgray", lw=1.2, zorder=1)
        ax.axhline(nominal_omega_um, color="gray", ls="--", lw=1.2, label=r"$\omega_0$ nominal")
        ax.axvline(0.5, color="red", ls=":", lw=1.5, label="Nyquist\n(2 px/rayon PSF)")
        ax.set_xlabel(r"Taille pixel / rayon PSF")
    else:
        for s, (pv, om, oe) in enumerate(zip(px, omega_mean, omega_std)):
            ax.errorbar(pv, om, yerr=oe, fmt="o", ms=6, capsize=4, color=color[s], zorder=3)
        ax.plot(px, omega_mean, "-", color="lightgray", lw=1.2, zorder=1)
        ax.set_xlabel("Taille du pixel [µm]")
    ax.set_ylabel(r"$\omega_0$ [µm]")
    ax.set_title("Biais de $\\omega_0$ vs sous-echantillonnage", fontsize=16, pad=13)
    ax.legend(fontsize=13)
    ax.text(-0.16, 1.05, "B", transform=ax.transAxes, fontsize=18, fontweight="bold", va="bottom", ha="right")

    # (c) densite mesuree vs ratio pixel/PSF -> exactitude en fonction du sous-echantillonnage
    ax = axes[2]
    dens_mean = np.array([results[c]["n_per_BA_mean"] for c in cases])
    dens_std = np.array([results[c]["n_per_BA_std"] for c in cases])
    if nominal_omega_um is not None:
        for s, (rv, dm, de) in enumerate(zip(ratio, dens_mean, dens_std)):
            ax.errorbar(rv, dm, yerr=de, fmt="o", ms=9, capsize=4, color=color[s], zorder=3)
        ax.plot(ratio, dens_mean, "-", color="lightgray", lw=1.2, zorder=1)
        ax.axvline(0.5, color="red", ls=":", lw=1.5, label="Nyquist\n(2 px/rayon PSF)")
        ax.set_xlabel(r"Taille pixel / rayon PSF")
    else:
        for s, (pv, dm, de) in enumerate(zip(px, dens_mean, dens_std)):
            ax.errorbar(pv, dm, yerr=de, fmt="o", ms=9, capsize=4, color=color[s], zorder=3)
        ax.plot(px, dens_mean, "-", color="lightgray", lw=1.2, zorder=1)
        ax.set_xlabel("Taille du pixel (µm)")
    if nominal_n_per_BA is not None:
        ax.axhline(nominal_n_per_BA, color="gray", ls="--", lw=1.2, label="densité nominale")
    ax.set_ylabel("Densité [particules/BA]")
    ax.set_title("Exactitude de la densité vs taille de pixel", fontsize=16, pad=13)
    ax.legend(fontsize=13)
    # ax.set_yscale("log")
    ax.text(-0.16, 1.05, "C", transform=ax.transAxes, fontsize=18, fontweight="bold", va="bottom", ha="right")

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return fig


# 6. Schema conceptuel (pour expliquer les parametres de simulation)
def plot_schema(psf_omega_um: float = 0.2, pixel_size_um: float = 0.05,
                savepath: str | Path | None = None):
    """
    Figure schematique (sans donnees reelles) illustrant :
        (a) le profil gaussien de la PSF et sa relation au "beam area"
        (b) la grille de pixels superposee a la PSF (echantillonnage)
        (c) le principe de la correlation "decaler-et-multiplier"
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # (a) profil PSF 1D + rayon 1/e^2
    ax = axes[0]
    r = np.linspace(-3 * psf_omega_um, 3 * psf_omega_um, 400)
    psf = np.exp(-2 * r ** 2 / psf_omega_um ** 2)  # profil d'intensite (1/e^2 a r=omega)
    ax.plot(r, psf, lw=2)
    ax.axvline(psf_omega_um, color="tab:red", ls="--")
    ax.axvline(-psf_omega_um, color="tab:red", ls="--")
    ax.axhline(np.exp(-2), color="gray", ls=":")
    ax.annotate(r"$\omega_0$", xy=(psf_omega_um, 0.05), color="tab:red")
    ax.set_xlabel("position (µm)"); ax.set_ylabel("Intensite normalisee")
    ax.set_title(f"PSF gaussienne (profil radial)\n$\\omega_0$ = {psf_omega_um} µm (rayon 1/e$^2$)")

    # (b) PSF 2D (beam area) superposee a la grille de pixels
    ax = axes[1]
    extent_um = 6 * psf_omega_um
    x = np.linspace(-extent_um/2, extent_um/2, 300)
    y = np.linspace(-extent_um/2, extent_um/2, 300)
    X, Y = np.meshgrid(x, y)
    PSF2D = np.exp(-2 * (X**2 + Y**2) / psf_omega_um**2)
    ax.imshow(PSF2D, extent=[x.min(), x.max(), y.min(), y.max()], cmap="inferno", origin="lower")
    circle = plt.Circle((0, 0), psf_omega_um, fill=False, color="cyan", lw=2)
    ax.add_patch(circle)
    # grille de pixels
    n_pix = int(extent_um / pixel_size_um)
    ticks = np.arange(-n_pix//2, n_pix//2 + 1) * pixel_size_um
    for t in ticks:
        ax.axvline(t, color="white", lw=0.3, alpha=0.5)
        ax.axhline(t, color="white", lw=0.3, alpha=0.5)
    ax.set_xlim(x.min(), x.max()); ax.set_ylim(y.min(), y.max())
    ax.set_title(f"PSF 2D + grille de pixels\ntaille pixel = {pixel_size_um} µm  "
                f"(beam area = $\\pi\\omega_0^2$ = {np.pi*psf_omega_um**2:.3f} µm$^2$)")
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")

    # (c) principe de la correlation : image originale vs image decalee (xi,eta)
    ax = axes[2]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.add_patch(plt.Rectangle((1, 1), 6, 6, fill=False, lw=2, color="tab:blue"))
    ax.add_patch(plt.Rectangle((3, 3), 6, 6, fill=False, lw=2, color="tab:red", ls="--"))
    ax.annotate("Image originale\nI(x,y)", (1.2, 7.3), color="tab:blue", fontsize=9)
    ax.annotate("Image decalee\nI(x+$\\xi$, y+$\\eta$)", (5.3, 9.3), color="tab:red", fontsize=9)
    # zone de recouvrement
    ax.add_patch(plt.Rectangle((3, 3), 4, 4, facecolor="gray", alpha=0.35))
    ax.annotate("pixels\ncommuns\n(recouvrement)", (4.0, 4.6), fontsize=8, ha="center")
    ax.annotate("", xy=(3, 3), xytext=(1, 1),
                arrowprops=dict(arrowstyle="->", color="black"))
    ax.text(1.6, 1.7, r"($\xi,\eta$)", fontsize=10)
    ax.axis("off")
    ax.set_title("Principe de G($\\xi,\\eta$) :\nne moyenner que sur le recouvrement")

    fig.suptitle("Schema des parametres de la simulation ICS", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return fig