from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from ics_core import crop_center
from rics_core import rics_model
from rics_batch import extract_D_from_case_name

plt.rcParams.update({
    "figure.dpi": 110,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
})
plt.style.use('https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-basic.mplstyle')


# 1. Vue d'ensemble d'un cas RICS unique
def plot_rics_case_overview(result: dict, title: str, savepath: str | Path | None = None,
                            half_width_display: int = 15):
    Gavg, xi, eta = result["Gavg"], result["xi"], result["eta"]
    overlap = result["overlap"]
    fit = result["fit_avg"]
    hw = min(half_width_display, Gavg.shape[0] // 2 - 1, Gavg.shape[1] // 2 - 1)
    Gc, xic, etac, _ = crop_center(Gavg, xi, eta, overlap, hw)

    fig = plt.figure(figsize=(13, 8.5))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(result["stack"][0], cmap="inferno")
    ax1.set_title("Image simulee (frame 1)")
    ax1.set_xlabel("x = colonnes (axe rapide)"); ax1.set_ylabel("y = lignes (axe lent)")

    ax2 = fig.add_subplot(2, 3, 2)
    extent = [etac.min(), etac.max(), xic.max(), xic.min()]
    im2 = ax2.imshow(Gc, cmap="inferno", extent=extent, aspect="auto")
    ax2.set_title(r"$G_s(\xi,\eta)$ mesuree")
    ax2.set_xlabel(r"$\eta$ (colonnes, rapide)"); ax2.set_ylabel(r"$\xi$ (lignes, lent)")
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    ax3 = fig.add_subplot(2, 3, 3, projection="3d")
    XI, ETA = np.meshgrid(xic, etac, indexing="ij")
    ax3.plot_surface(XI, ETA, Gc, cmap="inferno", linewidth=0, antialiased=True)
    ax3.set_title(r"Surface $G_s(\xi,\eta)$")
    ax3.set_xlabel(r"$\xi$ (lent)"); ax3.set_ylabel(r"$\eta$ (rapide)"); ax3.set_zlabel("G")

    fp = fit.fixed_params

    # (d) profil axe RAPIDE (eta, xi=0) avec le modele RICS complet
    ax4 = fig.add_subplot(2, 3, 4)
    j0 = int(np.where(etac == 0)[0][0])
    prof_fast = Gc[np.where(xic == 0)[0][0], :]
    model_fast = rics_model(np.zeros_like(etac, dtype=float), etac.astype(float),
                            fit.D, fit.N, fp["w0_um"], fp["pixel_size_um"],
                            fp["tau_p"], fp["tau_l"], wz_um=fp["wz_um"],
                            gamma=fp["gamma"], ginf=fit.ginf)
    ax4.plot(etac, prof_fast, "o", ms=4, label="donnees")
    ax4.plot(etac, model_fast, "-", lw=2, label="modele RICS ajuste")
    j0 = int(np.where(etac == 0)[0][0])
    ax4.plot(0, prof_fast[j0], "x", ms=12, mew=2.5, color="red", label="(0,0) exclu (bruit de photon)")
    ax4.set_xlabel(r"$\eta$ (pixels, axe RAPIDE)"); ax4.set_ylabel(r"$G_s(0,\eta)$")
    ax4.set_title("Profil axe rapide (colonnes)")
    ax4.legend(fontsize=8)

    # (e) profil axe LENT (xi, eta=0)
    ax5 = fig.add_subplot(2, 3, 5)
    i0 = int(np.where(xic == 0)[0][0])
    prof_slow = Gc[:, np.where(etac == 0)[0][0]]
    model_slow = rics_model(xic.astype(float), np.zeros_like(xic, dtype=float),
                            fit.D, fit.N, fp["w0_um"], fp["pixel_size_um"],
                            fp["tau_p"], fp["tau_l"], wz_um=fp["wz_um"],
                            gamma=fp["gamma"], ginf=fit.ginf)
    ax5.plot(xic, prof_slow, "o", ms=4, color="tab:orange", label="donnees")
    ax5.plot(xic, model_slow, "-", lw=2, color="tab:red", label="modele RICS ajuste")
    i0 = int(np.where(xic == 0)[0][0])
    ax5.plot(0, prof_slow[i0], "x", ms=12, mew=2.5, color="red",
            label="(0,0) exclu (bruit de photon)")
    ax5.set_xlabel(r"$\xi$ (pixels, axe LENT)"); ax5.set_ylabel(r"$G_s(\xi,0)$")
    ax5.set_title("Profil axe lent (lignes)")
    ax5.legend(fontsize=8)

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")
    D_nom = extract_D_from_case_name(Path(result["folder"]).name)
    txt = (
        f"D ajuste      = {fit.D:.3f} µm²/s\n"
        f"D nominal     = {D_nom:.3f} µm²/s\n"
        f"erreur        = {100*(fit.D-D_nom)/D_nom:+.1f} %\n\n"
        f"N ajuste      = {fit.N:.3f}\n"
        f"g_infini      = {fit.ginf:.5f}\n\n"
        f"--- parametres fixes (instrument) ---\n"
        f"w0            = {fp['w0_um']} µm\n"
        f"wz            = {fp['wz_um']} µm\n"
        f"pixel size    = {fp['pixel_size_um']} µm/pix\n"
        f"tau_p (pixel) = {fp['tau_p']*1e6:.2f} µs\n"
        f"tau_l (ligne) = {fp['tau_l']*1e3:.3f} ms\n\n"
        f"Taille image = {result['image_shape']}\n"
        f"N images moyennees = {result['n_frames']}"
    )
    ax6.text(0.02, 0.98, txt, va="top", ha="left", fontsize=9.5, family="monospace")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return fig


# 2. Effet du coefficient de diffusion D
def plot_diffusion_effect(results: dict[str, dict], savepath: str | Path | None = None):
    cases = sorted(results.keys(), key=extract_D_from_case_name)
    D_nom = [extract_D_from_case_name(c) for c in cases]

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.5))

    # (a) profils axe RAPIDE superposes -> etirement pour D eleve.
    # Le point (0,0) (pic de bruit de photon, cf. plot_rics_case_overview)
    # est retire de la courbe (il ne fait pas partie du modele physique) et
    # marque separement par un "x" pour rester visible sans ecraser l'echelle.
    ax = axes[0]
    for c in cases:
        r = results[c]
        hw = min(15, r["Gavg"].shape[0]//2-1, r["Gavg"].shape[1]//2-1)
        Gc, xic, etac, _ = crop_center(r["Gavg"], r["xi"], r["eta"], r["overlap"], hw)
        i0 = int(np.where(xic == 0)[0][0])
        j0 = int(np.where(etac == 0)[0][0])
        prof = Gc[i0, :].copy()
        zero_val = prof[j0]
        prof_plot = prof.copy(); prof_plot[j0] = np.nan
        line, = ax.plot(etac, prof_plot, "-o", ms=3, label=f"{c}")
        ax.plot(0, zero_val, "x", ms=8, mew=2, color=line.get_color())
    ax.set_yscale("symlog", linthresh=0.5)
    ax.set_xlabel(r"$\eta$ (pixels, axe rapide)"); ax.set_ylabel(r"$G_s(0,\eta)$ (echelle symlog)")
    ax.set_title("Axe RAPIDE : D etire/aplatit G\n(x = point (0,0), exclu, bruit de photon)")
    ax.legend(fontsize=7)

    # (b) profils axe LENT superposes -> decorrelation quasi immediate
    ax = axes[1]
    for c in cases:
        r = results[c]
        hw = min(15, r["Gavg"].shape[0]//2-1, r["Gavg"].shape[1]//2-1)
        Gc, xic, etac, _ = crop_center(r["Gavg"], r["xi"], r["eta"], r["overlap"], hw)
        j0 = int(np.where(etac == 0)[0][0])
        i0 = int(np.where(xic == 0)[0][0])
        prof = Gc[:, j0].copy()
        zero_val = prof[i0]
        prof_plot = prof.copy(); prof_plot[i0] = np.nan
        line, = ax.plot(xic, prof_plot, "-o", ms=3, label=f"{c}")
        ax.plot(0, zero_val, "x", ms=8, mew=2, color=line.get_color())
    ax.set_yscale("symlog", linthresh=0.5)
    ax.set_xlabel(r"$\xi$ (pixels, axe lent)"); ax.set_ylabel(r"$G_s(\xi,0)$ (echelle symlog)")
    ax.set_title("Axe LENT : decorrelation rapide meme a D modere\n(x = point (0,0), exclu, bruit de photon)")
    ax.legend(fontsize=7)

    # (c) exactitude : D ajuste vs D nominal (log-log, y=x)
    ax = axes[2]
    D_fit = [results[c]["fit_avg"].D for c in cases]
    D_fit_std = [results[c]["D_std"] for c in cases]
    ax.errorbar(D_nom, D_fit, yerr=D_fit_std, fmt="o", capsize=4, ms=7)
    lims = [min(D_nom)*0.5, max(D_nom)*2]
    ax.plot(lims, lims, "k--", lw=1, label="y = x (attendu)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$D$ nominal (µm²/s)"); ax.set_ylabel(r"$D$ ajuste (µm²/s)")
    ax.set_title("Exactitude de l'ajustement RICS")
    ax.legend(fontsize=8)

    # (d) ELONGATION le long de l'axe rapide : ratio (largeur 1/e axe rapide) /
    # (largeur 1/e axe lent), calcule a partir du MODELE ajuste (pas des
    # donnees brutes bruitees). C'est cette anisotropie -- G etiree le long
    # de l'axe rapide et quasi ponctuelle le long de l'axe lent -- qui est
    # la signature caracteristique d'une diffusion rapide en RICS (cf.
    # Digman et al. 2005a ; discussion dans le README).
    ax = axes[3]
    from rics_core import rics_model
    lag = np.linspace(0, 40, 2000)

    def width_1e(fit, fast_axis):
        fp = fit.fixed_params
        if fast_axis:
            g = rics_model(np.zeros_like(lag), lag, fit.D, fit.N, fp["w0_um"],
                            fp["pixel_size_um"], fp["tau_p"], fp["tau_l"],
                            wz_um=fp["wz_um"], gamma=fp["gamma"])
        else:
            g = rics_model(lag, np.zeros_like(lag), fit.D, fit.N, fp["w0_um"],
                            fp["pixel_size_um"], fp["tau_p"], fp["tau_l"],
                            wz_um=fp["wz_um"], gamma=fp["gamma"])
        g0 = g[0]
        below = np.where(g <= g0 / np.e)[0]
        return lag[below[0]] if len(below) else lag[-1]

    w_fast, w_slow = [], []
    for c in cases:
        fit = results[c]["fit_avg"]
        w_fast.append(width_1e(fit, True))
        w_slow.append(width_1e(fit, False))
    w_fast, w_slow = np.array(w_fast), np.array(w_slow)
    ratio = w_fast / np.maximum(w_slow, 1e-6)

    ax.plot(D_nom, w_fast, "o-", label="largeur 1/e, axe rapide")
    ax.plot(D_nom, w_slow, "s-", label="largeur 1/e, axe lent")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$D$ (µm²/s, valeur ajustee)")
    ax.set_ylabel("Largeur 1/e (pixels)")
    ax.set_title("Anisotropie de G (axe rapide vs lent) vs D\n(a partir du modele ajuste)")
    ax.legend(fontsize=7)
    ax2 = ax.twinx()
    ax2.plot(D_nom, ratio, "^--", color="tab:green", label="ratio rapide/lent (elongation)")
    ax2.set_ylabel("ratio largeur rapide/lent", color="tab:green")
    ax2.tick_params(axis="y", colors="tab:green")
    ax2.set_yscale("log")

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return fig


# 3. Effet THEORIQUE des parametres du microscope (tau_p, tau_l, pixel size)
def plot_microscope_params_effect(D_um2_s: float = 10.0, D_fast_um2_s: float = 100.0, N: float = 1.0,
                                w0_um: float = 0.2, pixel_size_um: float = 0.05, tau_p_ref: float = 8e-6,
                                tau_l_ref: float = 2e-3, savepath: str | Path | None = None):
    """
    Etude theorique (modele RICS seul, aucune image requise) :
    (a) effet de tau_p (temps de pixel) sur le profil de l'axe RAPIDE,
        a tau_l fixe -- utilise D_fast_um2_s (diffusion rapide, ou l'effet
        de tau_p est visible ; a D lente, 4*D*tau_p/w0^2 << 1 et l'axe
        rapide est insensible a tau_p, cf. panneau (d))
    (b) effet de tau_l (temps de ligne) sur le profil de l'axe LENT,
        a tau_p et D fixes
    (c) effet de la taille du pixel sur l'axe rapide (a tau_p, tau_l, D fixes)
    (d) "carte" D vs tau_p montrant le regime ou l'axe rapide est sensible
        a D (utile pour choisir tau_p en fonction du D attendu)
    """
    lag = np.linspace(0, 20, 200)

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.5))

    # (a) effet de tau_p sur l'axe rapide (D rapide pour que l'effet soit visible)
    ax = axes[0]
    color = ["#aea4f4ff", "#7b6ce9ff", "#3d2bc4ff", "#0d044dff", "#010004ff"]
    s = 0
    for tau_p in [1e-6, 8e-6, 30e-6, 80e-6, 200e-6]:
        g = rics_model(np.zeros_like(lag), lag, D_fast_um2_s, N, w0_um, pixel_size_um,
                        tau_p, tau_l_ref)
        ax.plot(lag, g / g.max(), color=color[s], label=f"$\\tau_p$={tau_p*1e6:.0f} µs")
        s += 1
    ax.set_xlabel(r"$\eta$ (pixels, axe rapide)"); ax.set_ylabel("G normalise")
    ax.set_title(f"Effet de $\\tau_p$ sur l'axe rapide\n(D={D_fast_um2_s} µm²/s -- diffusion RAPIDE --, $\\tau_l$={tau_l_ref*1e3:.2f} ms fixe)")
    ax.legend(fontsize=7)
    ax.text(-0.16, 1.05, "A", transform=ax.transAxes, fontsize=18, fontweight="bold", va="bottom", ha="right")

    # (b) effet de tau_l sur l'axe lent
    ax = axes[1]
    s = 0
    for tau_l in [0.2e-3, 1e-3, 2e-3, 5e-3, 10e-3]:
        g = rics_model(lag, np.zeros_like(lag), D_um2_s, N, w0_um, pixel_size_um,
                        tau_p_ref, tau_l)
        ax.plot(lag, g / g.max(), color=color[s], label=f"$\\tau_l$={tau_l*1e3:.1f} ms")
        s += 1
    ax.set_xlabel(r"$\xi$ (pixels, axe lent)"); ax.set_ylabel("G normalise")
    ax.set_title(f"Effet de $\\tau_l$ sur l'axe lent\n(D={D_um2_s} µm²/s, $\\tau_p$={tau_p_ref*1e6:.0f} µs fixe)")
    ax.legend(fontsize=7)
    ax.text(-0.16, 1.05, "B", transform=ax.transAxes, fontsize=18, fontweight="bold", va="bottom", ha="right")

    # (c) effet de la taille de pixel sur l'axe rapide
    ax = axes[2]
    s = 0
    for ps in [0.02, 0.05, 0.1, 0.2]:
        g = rics_model(np.zeros_like(lag), lag, D_um2_s, N, w0_um, ps,
                        tau_p_ref, tau_l_ref)
        ax.plot(lag, g / g.max(), color=color[s], label=f"pixel={ps} µm")
        s += 1
    ax.set_xlabel(r"$\eta$ (pixels, axe rapide)"); ax.set_ylabel("G normalise")
    ax.set_title("Effet de la taille de pixel\n(axe rapide, D, $\\tau_p$, $\\tau_l$ fixes)")
    ax.legend(fontsize=7)
    ax.text(-0.16, 1.05, "C", transform=ax.transAxes, fontsize=18, fontweight="bold", va="bottom", ha="right")

    # (d) carte D vs tau_p : parametre sans dimension 4*D*tau_p/w0^2
    #     (indique le regime ou l'axe rapide devient sensible a D)
    ax = axes[3]
    D_range = np.logspace(-1, 2, 60)     # 0.1 a 100 um2/s (comme l'enonce)
    taup_range = np.logspace(-6, -4, 60)  # 1 a 100 us
    Dg, Tg = np.meshgrid(D_range, taup_range, indexing="xy")
    dimless = 4 * Dg * Tg / w0_um**2
    im = ax.pcolormesh(D_range, taup_range*1e6, dimless, shading="auto",
                        norm=plt.matplotlib.colors.LogNorm())
    plt.colorbar(im, ax=ax, label=r"$4D\tau_p/\omega_0^2$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("D (µm²/s)"); ax.set_ylabel(r"$\tau_p$ (µs)")
    ax.set_title("Regime sensible sur l'axe rapide\n($4D\\tau_p/\\omega_0^2 \\gtrsim 1$)")
    cs = ax.contour(D_range, taup_range*1e6, dimless, levels=[1.0], colors="white")
    ax.clabel(cs, fmt="=1")
    ax.text(-0.16, 1.05, "D", transform=ax.transAxes, fontsize=18, fontweight="bold", va="bottom", ha="right")

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return fig


# 4. Schema de la structure temporelle du balayage raster
def plot_rics_schema(tau_p_s: float = 8e-6, tau_l_s: float = 2e-3, n_pix_row: int = 6,
                    n_rows: int = 3, savepath: str | Path | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) grille de pixels avec le temps d'acquisition de chacun -- schematique,
    # PAS a l'echelle (tau_l est ~250x plus grand que tau_p, un trace a l'echelle
    # serait illisible sur un seul axe lineaire)
    ax = axes[0]
    for row in range(n_rows):
        for col in range(n_pix_row):
            t = row * tau_l_s + col * tau_p_s
            rect = plt.Rectangle((col, n_rows - 1 - row), 1, 1,
                                facecolor=plt.cm.inferno(0.15 + 0.7 * col / n_pix_row),
                                edgecolor="white", lw=1.5)
            ax.add_patch(rect)
            label = f"{col}$\\tau_p$" if row == 0 else f"{row}$\\tau_l$" + (f"+{col}$\\tau_p$" if col else "")
            ax.text(col + 0.5, n_rows - 1 - row + 0.5, label, ha="center", va="center",
                    fontsize=7.5, color="white")
    ax.set_xlim(0, n_pix_row); ax.set_ylim(0, n_rows)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("axe RAPIDE (colonnes, pas de temps = " + r"$\tau_p$" +
                f" = {tau_p_s*1e6:.1f} µs)")
    ax.set_ylabel("axe LENT\n(lignes, pas = " + r"$\tau_l$" + f" = {tau_l_s*1e3:.2f} ms)")
    ax.set_title("Temps d'acquisition de chaque pixel (schematique, pas a l'echelle) :\n"
                r"$t(x,y) = y\cdot\tau_l + x\cdot\tau_p$")
    ax.set_aspect("equal")

    # (b) echelle logarithmique des temps caracteristiques -- ceci EST a l'echelle
    # (cf. Fig. 9 de Kolin & Wiseman 2007 / Table 1 de Brown et al. 2008)
    ax = axes[1]
    times = {
        r"$\tau_p$ (pixel)": tau_p_s,
        r"$\tau_l$ (ligne)": tau_l_s,
        "temps de trame\n(~ $H\\times\\tau_l$)": tau_l_s * 128,
    }
    ax.set_xscale("log")
    ax.set_xlim(1e-7, 1e2)
    ax.set_yticks([])
    for i, (label, t) in enumerate(times.items()):
        ax.axvline(t, color=f"C{i}", lw=2)
        ax.annotate(f"{label}\n{t:.2g} s", (t, 0.55 + 0.15*i), color=f"C{i}",
                    ha="center", fontsize=9)
    ax.annotate("diffusion rapide\n(GFP en solution)\n~1 ms", (1e-3, 0.1), fontsize=8,
                color="gray", ha="center")
    ax.annotate("diffusion lente\n(recepteur membranaire)\n~0.1-1 s", (0.3, 0.1), fontsize=8,
                color="gray", ha="center")
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Temps caracteristique (s, echelle log)")
    ax.set_title("Echelles de temps sondees par RICS\n(cf. Table 1, Brown et al. 2008)")

    fig.suptitle("Structure temporelle du balayage raster (RICS)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return fig