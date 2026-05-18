"""
generate_report.py
Génère un rapport PDF médical professionnel pour CardioAI.
Usage : importé dans app.py, appelé via generate_pdf_report(...)
"""

import io
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, HRFlowable, PageBreak, KeepTogether
)
from reportlab.pdfgen import canvas as rl_canvas


# ── Palette couleurs ────────────────────────────────────────────────
RED    = colors.HexColor("#C0392B")
DKRED  = colors.HexColor("#922B21")
GREEN  = colors.HexColor("#27AE60")
BLUE   = colors.HexColor("#2980B9")
AMBER  = colors.HexColor("#F39C12")
GREY   = colors.HexColor("#7F8C8D")
LGREY  = colors.HexColor("#ECF0F1")
DKGREY = colors.HexColor("#2C3E50")
WHITE  = colors.white
BLACK  = colors.black

RISK_COLOR = {
    "Faible":  "#27AE60",
    "Modéré":  "#F39C12",
    "Élevé":   "#C0392B",
}


# ── Styles ──────────────────────────────────────────────────────────
def make_styles():
    base = getSampleStyleSheet()
    custom = {}

    custom["title"] = ParagraphStyle(
        "title", parent=base["Normal"],
        fontSize=22, fontName="Helvetica-Bold",
        textColor=WHITE, alignment=TA_CENTER, spaceAfter=4
    )
    custom["subtitle"] = ParagraphStyle(
        "subtitle", parent=base["Normal"],
        fontSize=10, fontName="Helvetica",
        textColor=colors.HexColor("#FADADD"), alignment=TA_CENTER
    )
    custom["section"] = ParagraphStyle(
        "section", parent=base["Normal"],
        fontSize=13, fontName="Helvetica-Bold",
        textColor=DKGREY, spaceBefore=14, spaceAfter=6,
        borderPad=4
    )
    custom["body"] = ParagraphStyle(
        "body", parent=base["Normal"],
        fontSize=9, fontName="Helvetica",
        textColor=DKGREY, spaceAfter=4, leading=14
    )
    custom["small"] = ParagraphStyle(
        "small", parent=base["Normal"],
        fontSize=7.5, fontName="Helvetica",
        textColor=GREY, spaceAfter=2
    )
    custom["label"] = ParagraphStyle(
        "label", parent=base["Normal"],
        fontSize=8, fontName="Helvetica-Bold",
        textColor=GREY, spaceAfter=1
    )
    custom["value"] = ParagraphStyle(
        "value", parent=base["Normal"],
        fontSize=11, fontName="Helvetica-Bold",
        textColor=DKGREY, spaceAfter=2
    )
    custom["result_ok"] = ParagraphStyle(
        "result_ok", parent=base["Normal"],
        fontSize=16, fontName="Helvetica-Bold",
        textColor=GREEN, alignment=TA_CENTER
    )
    custom["result_bad"] = ParagraphStyle(
        "result_bad", parent=base["Normal"],
        fontSize=16, fontName="Helvetica-Bold",
        textColor=RED, alignment=TA_CENTER
    )
    custom["center"] = ParagraphStyle(
        "center", parent=base["Normal"],
        fontSize=9, fontName="Helvetica",
        textColor=DKGREY, alignment=TA_CENTER
    )
    custom["footer"] = ParagraphStyle(
        "footer", parent=base["Normal"],
        fontSize=7, fontName="Helvetica",
        textColor=GREY, alignment=TA_CENTER
    )
    return custom


# ── Graphique matplotlib → image bytes ──────────────────────────────
def fig_to_image(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf


def make_gauge(probability, prediction):
    """Jauge semi-circulaire du niveau de risque."""
    fig, ax = plt.subplots(figsize=(4, 2.2), facecolor="white")
    ax.set_facecolor("white")

    theta = np.linspace(np.pi, 0, 300)
    # Fond arc
    r_out, r_in = 1.0, 0.55
    for i in range(len(theta) - 1):
        t1, t2 = theta[i], theta[i + 1]
        frac = i / (len(theta) - 1)
        c = plt.cm.RdYlGn_r(frac)
        ax.fill_between(
            [r_in * np.cos(t1), r_out * np.cos(t1),
             r_out * np.cos(t2), r_in * np.cos(t2)],
            [r_in * np.sin(t1), r_out * np.sin(t1),
             r_out * np.sin(t2), r_in * np.sin(t2)],
            color=c, linewidth=0
        )

    # Aiguille
    angle = np.pi * (1 - probability)
    needle_x = [0, 0.8 * np.cos(angle)]
    needle_y = [0, 0.8 * np.sin(angle)]
    ax.plot(needle_x, needle_y, color="#2C3E50", linewidth=3, zorder=5)
    ax.add_patch(plt.Circle((0, 0), 0.06, color="#2C3E50", zorder=6))

    # Texte centre
    risk_lvl = "Faible" if probability < 0.33 else ("Modéré" if probability < 0.66 else "Élevé")
    risk_col = RISK_COLOR[risk_lvl]
    ax.text(0, -0.18, f"{probability:.0%}", ha="center", va="center",
            fontsize=16, fontweight="bold", color=risk_col)
    ax.text(0, -0.38, f"Risque {risk_lvl}", ha="center", va="center",
            fontsize=8, color=risk_col)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.5, 1.1)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    return fig_to_image(fig)


def make_bar_metrics(metrics_dict, best_name):
    """Barres horizontales des métriques des modèles."""
    fig, ax = plt.subplots(figsize=(6.5, 3.2), facecolor="white")
    ax.set_facecolor("white")

    models = list(metrics_dict.keys())
    auc_vals = [metrics_dict[m]["AUC"] for m in models]
    colors_bar = ["#C0392B" if m == best_name else "#AEB6BF" for m in models]

    bars = ax.barh(models, auc_vals, color=colors_bar, height=0.55, edgecolor="none")
    for bar, val in zip(bars, auc_vals):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8,
                color="#C0392B" if val == max(auc_vals) else "#7F8C8D",
                fontweight="bold" if val == max(auc_vals) else "normal")

    ax.set_xlim(0, 1.12)
    ax.set_xlabel("AUC-ROC", fontsize=8, color="#7F8C8D")
    ax.tick_params(axis="both", labelsize=8, colors="#7F8C8D")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#ECF0F1")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))
    ax.axvline(0.5, color="#ECF0F1", linewidth=1, linestyle="--")
    ax.set_title("AUC-ROC par modèle", fontsize=9, color="#2C3E50", pad=8)
    fig.tight_layout(pad=0.5)
    return fig_to_image(fig)


def make_pie_chart(n_sain, n_malade):
    """Camembert Sain / Malade."""
    fig, ax = plt.subplots(figsize=(3.2, 2.8), facecolor="white")
    ax.set_facecolor("white")
    vals   = [n_sain, n_malade]
    labels = [f"Sains\n{n_sain}", f"Malades\n{n_malade}"]
    cols   = ["#27AE60", "#C0392B"]
    wedges, texts, autotexts = ax.pie(
        vals, labels=labels, colors=cols, autopct="%1.1f%%",
        startangle=90, pctdistance=0.75,
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2)
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_color("white")
        at.set_fontweight("bold")
    for t in texts:
        t.set_fontsize(8)
        t.set_color("#2C3E50")
    ax.set_title("Répartition du dataset", fontsize=9, color="#2C3E50", pad=6)
    fig.tight_layout(pad=0.2)
    return fig_to_image(fig)


def make_age_hist(df_live):
    """Histogramme âge coloré par target."""
    fig, ax = plt.subplots(figsize=(6.5, 2.8), facecolor="white")
    ax.set_facecolor("white")
    bins = range(int(df_live["age"].min()) - 1, int(df_live["age"].max()) + 2, 3)
    ax.hist(df_live[df_live["target"] == 0]["age"], bins=bins,
            color="#27AE60", alpha=0.75, label="Sain", edgecolor="white")
    ax.hist(df_live[df_live["target"] == 1]["age"], bins=bins,
            color="#C0392B", alpha=0.75, label="Malade", edgecolor="white")
    ax.set_xlabel("Âge (années)", fontsize=8, color="#7F8C8D")
    ax.set_ylabel("Fréquence", fontsize=8, color="#7F8C8D")
    ax.tick_params(labelsize=7.5, colors="#7F8C8D")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#ECF0F1")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Distribution de l'âge par diagnostic", fontsize=9, color="#2C3E50", pad=6)
    fig.tight_layout(pad=0.4)
    return fig_to_image(fig)


# ── Numéro de page (canvas callback) ────────────────────────────────
class NumberedCanvas(rl_canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        n = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_page_number(n)
            super().showPage()
        super().save()

    def _draw_page_number(self, total):
        w, h = A4
        self.setFont("Helvetica", 7)
        self.setFillColor(GREY)
        self.drawRightString(w - 1.5 * cm, 0.8 * cm,
                             f"Page {self._pageNumber} / {total}")
        self.drawString(1.5 * cm, 0.8 * cm,
                        "CardioAI — Rapport confidentiel — IFOAD")
        self.setStrokeColor(LGREY)
        self.setLineWidth(0.5)
        self.line(1.5 * cm, 1.1 * cm, w - 1.5 * cm, 1.1 * cm)


# ── Fonction principale ──────────────────────────────────────────────
def generate_pdf_report(
    patient_data: dict,
    prediction: int,
    probability: float,
    model_name: str,
    metrics_dict: dict,
    best_model_name: str,
    df_live: pd.DataFrame,
    history: list,
) -> bytes:
    """
    Génère un rapport PDF médical et retourne les bytes.

    patient_data  : dict des valeurs saisies (age, sex, chol, ...)
    prediction    : 0 ou 1
    probability   : float entre 0 et 1
    model_name    : nom du modèle utilisé
    metrics_dict  : {nom_modele: {Accuracy, Precision, Recall, F1, AUC}}
    best_model_name : meilleur modèle selon AUC
    df_live       : DataFrame complet (original + nouveaux)
    history       : liste de dicts historique prédictions
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=1.5 * cm, rightMargin=1.5 * cm,
        topMargin=1.8 * cm, bottomMargin=2 * cm,
        title="Rapport CardioAI", author="CardioAI — IFOAD"
    )

    S = make_styles()
    story = []
    W = A4[0] - 3 * cm   # largeur utile

    now = datetime.datetime.now()
    risk_lvl = "Faible" if probability < 0.33 else ("Modéré" if probability < 0.66 else "Élevé")
    diag_txt = "Maladie cardiaque détectée" if prediction == 1 else "Aucune maladie détectée"
    diag_col = RED if prediction == 1 else GREEN

    # ════════════════════════════════════════════════════
    # PAGE 1 — EN-TÊTE + RÉSULTAT PATIENT
    # ════════════════════════════════════════════════════

    # Bandeau rouge header
    header_data = [[
        Paragraph("❤ CardioAI", S["title"]),
    ]]
    header_table = Table(header_data, colWidths=[W])
    header_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), RED),
        ("ROUNDEDCORNERS", (0, 0), (-1, -1), [10, 10, 10, 10]),
        ("TOPPADDING",  (0, 0), (-1, -1), 18),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(header_table)

    sub_data = [[
        Paragraph("Rapport de prédiction médicale — Intelligence Artificielle", S["subtitle"]),
    ]]
    sub_table = Table(sub_data, colWidths=[W])
    sub_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), DKRED),
        ("TOPPADDING",  (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(sub_table)
    story.append(Spacer(1, 0.4 * cm))

    # Infos rapport
    info_data = [
        [Paragraph("Date du rapport", S["label"]),
         Paragraph("Établissement", S["label"]),
         Paragraph("Modèle utilisé", S["label"]),
         Paragraph("Référence", S["label"])],
        [Paragraph(now.strftime("%d/%m/%Y %H:%M"), S["value"]),
         Paragraph("IFOAD · Dr Arthur Sawadogo", S["value"]),
         Paragraph(model_name, S["value"]),
         Paragraph(f"RPT-{now.strftime('%Y%m%d%H%M%S')}", S["value"])],
    ]
    info_table = Table(info_data, colWidths=[W / 4] * 4)
    info_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), LGREY),
        ("BACKGROUND",  (0, 1), (-1, 1), WHITE),
        ("BOX",         (0, 0), (-1, -1), 0.5, colors.HexColor("#E0E0E0")),
        ("INNERGRID",   (0, 0), (-1, -1), 0.3, colors.HexColor("#E0E0E0")),
        ("TOPPADDING",  (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.5 * cm))

    # ── Section Données patient ──────────────────────────
    story.append(Paragraph("📋 Données du patient", S["section"]))
    story.append(HRFlowable(width=W, thickness=1.5, color=RED, spaceAfter=6))

    sex_label = "Homme" if patient_data.get("sex", 1) == 1 else "Femme"
    cp_labels  = {0:"Angine typique", 1:"Angine atypique", 2:"Non-anginale", 3:"Asymptomatique"}
    thal_labels = {3:"Normal", 6:"Défaut fixé", 7:"Défaut réversible"}

    pat_rows = [
        [Paragraph("Paramètre", S["label"]), Paragraph("Valeur", S["label"]),
         Paragraph("Paramètre", S["label"]), Paragraph("Valeur", S["label"])],
        [Paragraph("Âge", S["body"]),
         Paragraph(f"{patient_data.get('age','—')} ans", S["value"]),
         Paragraph("Sexe", S["body"]),
         Paragraph(sex_label, S["value"])],
        [Paragraph("Pression artérielle", S["body"]),
         Paragraph(f"{patient_data.get('trestbps','—')} mm Hg", S["value"]),
         Paragraph("Cholestérol", S["body"]),
         Paragraph(f"{patient_data.get('chol','—')} mg/dl", S["value"])],
        [Paragraph("Fréquence cardiaque max", S["body"]),
         Paragraph(f"{patient_data.get('thalach','—')} bpm", S["value"]),
         Paragraph("Dépression ST (Oldpeak)", S["body"]),
         Paragraph(str(patient_data.get("oldpeak", "—")), S["value"])],
        [Paragraph("Douleur thoracique", S["body"]),
         Paragraph(cp_labels.get(int(patient_data.get("cp", 0)), "—"), S["value"]),
         Paragraph("Thalassémie", S["body"]),
         Paragraph(thal_labels.get(int(patient_data.get("thal", 3)), "—"), S["value"])],
        [Paragraph("Glycémie à jeun > 120", S["body"]),
         Paragraph("Oui" if patient_data.get("fbs", 0) == 1 else "Non", S["value"]),
         Paragraph("Angine à l'exercice", S["body"]),
         Paragraph("Oui" if patient_data.get("exang", 0) == 1 else "Non", S["value"])],
        [Paragraph("Vaisseaux majeurs (ca)", S["body"]),
         Paragraph(str(patient_data.get("ca", "—")), S["value"]),
         Paragraph("Pente ST (slope)", S["body"]),
         Paragraph(str(patient_data.get("slope", "—")), S["value"])],
    ]
    pat_table = Table(pat_rows, colWidths=[W * 0.28, W * 0.22, W * 0.28, W * 0.22])
    pat_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), LGREY),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0), 8),
        ("TEXTCOLOR",   (0, 0), (-1, 0), GREY),
        ("BOX",         (0, 0), (-1, -1), 0.5, colors.HexColor("#E0E0E0")),
        ("INNERGRID",   (0, 0), (-1, -1), 0.3, colors.HexColor("#F0F0F0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [WHITE, colors.HexColor("#FAFAFA")]),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(pat_table)
    story.append(Spacer(1, 0.5 * cm))

    # ── Section Résultat ────────────────────────────────
    story.append(Paragraph("🎯 Résultat de l'analyse IA", S["section"]))
    story.append(HRFlowable(width=W, thickness=1.5, color=RED, spaceAfter=6))

    # Jauge + résultat côte à côte
    gauge_img = make_gauge(probability, prediction)
    gauge = Image(gauge_img, width=6 * cm, height=3.3 * cm)

    result_color_hex = "#C0392B" if prediction == 1 else "#27AE60"
    result_bg = colors.HexColor("#FFF5F5") if prediction == 1 else colors.HexColor("#F0FFF4")
    result_border = RED if prediction == 1 else GREEN

    icon = "⚠" if prediction == 1 else "✓"
    result_content = [
        [Paragraph(f"{icon}  {diag_txt}", S["result_bad"] if prediction == 1 else S["result_ok"])],
        [Spacer(1, 0.2 * cm)],
        [Paragraph(f"Probabilité de risque : <b>{probability:.1%}</b>", S["body"])],
        [Paragraph(f"Niveau de risque : <b>{risk_lvl}</b>", S["body"])],
        [Paragraph(f"Modèle utilisé : <b>{model_name}</b>", S["body"])],
        [Spacer(1, 0.2 * cm)],
        [Paragraph(
            "⚠ Ce résultat est fourni à titre indicatif uniquement. "
            "Consultez un cardiologue pour tout diagnostic médical.",
            S["small"]
        )],
    ]
    result_tbl_inner = Table(result_content, colWidths=[W * 0.58])
    result_tbl_inner.setStyle(TableStyle([
        ("TOPPADDING",    (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
    ]))

    combined = Table([[gauge, result_tbl_inner]],
                     colWidths=[W * 0.35, W * 0.65])
    combined.setStyle(TableStyle([
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND",  (1, 0), (1, 0), result_bg),
        ("BOX",         (0, 0), (-1, -1), 1.5, result_border),
        ("LINEBEFORE",  (1, 0), (1, 0), 3, result_border),
        ("TOPPADDING",  (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("ROUNDEDCORNERS", (0, 0), (-1, -1), [8, 8, 8, 8]),
    ]))
    story.append(combined)
    story.append(Spacer(1, 0.5 * cm))

    # ── Interprétation médicale ──────────────────────────
    story.append(Paragraph("📖 Interprétation médicale", S["section"]))
    story.append(HRFlowable(width=W, thickness=1.5, color=RED, spaceAfter=6))

    chol_val  = patient_data.get("chol", 0)
    tbp_val   = patient_data.get("trestbps", 0)
    age_val   = patient_data.get("age", 0)
    thalach_v = patient_data.get("thalach", 0)

    interp_lines = []
    if chol_val > 240:
        interp_lines.append(f"• Cholestérol élevé ({chol_val} mg/dl > 240) : facteur de risque cardiovasculaire significatif.")
    elif chol_val > 200:
        interp_lines.append(f"• Cholestérol limite ({chol_val} mg/dl) : surveillance recommandée.")
    else:
        interp_lines.append(f"• Cholestérol normal ({chol_val} mg/dl).")

    if tbp_val > 140:
        interp_lines.append(f"• Hypertension artérielle ({tbp_val} mm Hg) : facteur de risque important.")
    elif tbp_val > 120:
        interp_lines.append(f"• Pression artérielle légèrement élevée ({tbp_val} mm Hg) : à surveiller.")
    else:
        interp_lines.append(f"• Pression artérielle normale ({tbp_val} mm Hg).")

    if age_val >= 60:
        interp_lines.append(f"• Âge ({age_val} ans) : profil à risque cardiovasculaire augmenté après 60 ans.")
    elif age_val >= 45:
        interp_lines.append(f"• Âge ({age_val} ans) : vigilance recommandée dès 45 ans.")

    if thalach_v < 100:
        interp_lines.append(f"• Fréquence cardiaque max faible ({thalach_v} bpm) : peut indiquer une capacité cardiaque réduite.")

    if patient_data.get("exang", 0) == 1:
        interp_lines.append("• Angine induite par l'exercice : signe clinique important à investiguer.")

    if not interp_lines:
        interp_lines.append("• Les paramètres cliniques semblent dans des ranges normaux.")

    for line in interp_lines:
        story.append(Paragraph(line, S["body"]))
    story.append(Spacer(1, 0.3 * cm))

    # ════════════════════════════════════════════════════
    # PAGE 2 — ANALYSE DU DATASET + MODÈLES
    # ════════════════════════════════════════════════════
    story.append(PageBreak())

    story.append(Paragraph("📊 Analyse du dataset", S["section"]))
    story.append(HRFlowable(width=W, thickness=1.5, color=RED, spaceAfter=8))

    n_total  = len(df_live)
    n_mal    = int(df_live["target"].sum())
    n_sain   = n_total - n_mal
    age_moy  = df_live["age"].mean()
    chol_moy = df_live["chol"].mean()
    pct_h    = (df_live["sex"] == 1).mean() * 100

    # KPI cards
    kpi_data = [[
        Paragraph(f"<b>{n_total}</b><br/>Patients total", S["center"]),
        Paragraph(f"<b>{n_mal}</b><br/>Malades ({n_mal/n_total:.1%})", S["center"]),
        Paragraph(f"<b>{n_sain}</b><br/>Sains ({n_sain/n_total:.1%})", S["center"]),
        Paragraph(f"<b>{age_moy:.1f} ans</b><br/>Âge moyen", S["center"]),
        Paragraph(f"<b>{chol_moy:.0f}</b><br/>Cholest. moy.", S["center"]),
        Paragraph(f"<b>{pct_h:.0f}%</b><br/>Hommes", S["center"]),
    ]]
    kpi_table = Table(kpi_data, colWidths=[W / 6] * 6)
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (0, 0), colors.HexColor("#EBF5FB")),
        ("BACKGROUND",  (1, 0), (1, 0), colors.HexColor("#FDEDEC")),
        ("BACKGROUND",  (2, 0), (2, 0), colors.HexColor("#EAFAF1")),
        ("BACKGROUND",  (3, 0), (3, 0), colors.HexColor("#FEF9E7")),
        ("BACKGROUND",  (4, 0), (4, 0), colors.HexColor("#F5EEF8")),
        ("BACKGROUND",  (5, 0), (5, 0), colors.HexColor("#EBF5FB")),
        ("BOX",         (0, 0), (-1, -1), 0.5, colors.HexColor("#E0E0E0")),
        ("INNERGRID",   (0, 0), (-1, -1), 0.3, colors.HexColor("#E0E0E0")),
        ("TOPPADDING",  (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 0.4 * cm))

    # Graphiques côte à côte
    pie_img = make_pie_chart(n_sain, n_mal)
    hist_img = make_age_hist(df_live)
    pie  = Image(pie_img,  width=W * 0.37, height=6 * cm)
    hist = Image(hist_img, width=W * 0.60, height=6 * cm)
    charts_row = Table([[pie, hist]], colWidths=[W * 0.38, W * 0.62])
    charts_row.setStyle(TableStyle([
        ("VALIGN",  (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(charts_row)
    story.append(Spacer(1, 0.5 * cm))

    # ── Section Comparaison modèles ──────────────────────
    story.append(Paragraph("🤖 Comparaison des 6 algorithmes", S["section"]))
    story.append(HRFlowable(width=W, thickness=1.5, color=RED, spaceAfter=8))

    # Tableau métriques
    header_row = [
        Paragraph("Modèle", S["label"]),
        Paragraph("Accuracy", S["label"]),
        Paragraph("Précision", S["label"]),
        Paragraph("Rappel", S["label"]),
        Paragraph("F1-Score", S["label"]),
        Paragraph("AUC-ROC", S["label"]),
    ]
    met_rows = [header_row]
    for mname, mvals in metrics_dict.items():
        is_best = (mname == best_model_name)
        style = ParagraphStyle("bold", parent=S["body"],
                               fontName="Helvetica-Bold" if is_best else "Helvetica",
                               textColor=RED if is_best else DKGREY)
        badge = " ★" if is_best else ""
        met_rows.append([
            Paragraph(mname + badge, style),
            Paragraph(f"{mvals.get('Accuracy', 0):.4f}", style),
            Paragraph(f"{mvals.get('Precision', 0):.4f}", style),
            Paragraph(f"{mvals.get('Recall', 0):.4f}", style),
            Paragraph(f"{mvals.get('F1', 0):.4f}", style),
            Paragraph(f"{mvals.get('AUC', 0):.4f}", style),
        ])

    met_table = Table(met_rows, colWidths=[W * 0.30, W * 0.14, W * 0.14,
                                            W * 0.14, W * 0.14, W * 0.14])
    row_bgs = [LGREY] + [
        colors.HexColor("#FFF5F5") if list(metrics_dict.keys())[i] == best_model_name
        else (WHITE if i % 2 == 0 else colors.HexColor("#FAFAFA"))
        for i in range(len(metrics_dict))
    ]
    met_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), LGREY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#FFF5F5")
          if list(metrics_dict.keys())[i % len(metrics_dict)] == best_model_name
          else (WHITE if i % 2 == 0 else colors.HexColor("#FAFAFA"))
          for i in range(len(metrics_dict))]),
        ("BOX",         (0, 0), (-1, -1), 0.5, colors.HexColor("#E0E0E0")),
        ("INNERGRID",   (0, 0), (-1, -1), 0.3, colors.HexColor("#F0F0F0")),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("ALIGN",       (1, 0), (-1, -1), "CENTER"),
    ]))
    story.append(met_table)
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        f"★ Meilleur modèle : {best_model_name} "
        f"(AUC-ROC = {metrics_dict[best_model_name].get('AUC', 0):.4f})",
        ParagraphStyle("note", parent=S["small"],
                       textColor=RED, fontName="Helvetica-Bold")
    ))
    story.append(Spacer(1, 0.3 * cm))

    # Graphique AUC
    bar_img = make_bar_metrics(metrics_dict, best_model_name)
    story.append(Image(bar_img, width=W, height=4.5 * cm))
    story.append(Spacer(1, 0.4 * cm))

    # ── Historique des prédictions ───────────────────────
    if history:
        story.append(Paragraph(
            f"📋 Historique des prédictions ({len(history)} patient(s))",
            S["section"]
        ))
        story.append(HRFlowable(width=W, thickness=1.5, color=RED, spaceAfter=6))

        hist_header = [
            Paragraph("#", S["label"]),
            Paragraph("Âge", S["label"]),
            Paragraph("Sexe", S["label"]),
            Paragraph("Cholestérol", S["label"]),
            Paragraph("Modèle", S["label"]),
            Paragraph("Prédiction", S["label"]),
            Paragraph("Probabilité", S["label"]),
        ]
        hist_rows = [hist_header]
        for i, h in enumerate(history[-15:], 1):   # max 15 lignes
            pred_style = ParagraphStyle(
                "ph", parent=S["body"],
                textColor=RED if h.get("Prédiction") == "Malade" else GREEN,
                fontName="Helvetica-Bold"
            )
            hist_rows.append([
                Paragraph(str(i), S["body"]),
                Paragraph(str(h.get("Age", "—")), S["body"]),
                Paragraph(h.get("Sexe", "—"), S["body"]),
                Paragraph(str(h.get("Cholestérol", "—")), S["body"]),
                Paragraph(h.get("Modèle", "—"), S["small"]),
                Paragraph(h.get("Prédiction", "—"), pred_style),
                Paragraph(h.get("Probabilité", "—"), S["body"]),
            ])

        hist_table = Table(
            hist_rows,
            colWidths=[W * 0.05, W * 0.08, W * 0.08,
                       W * 0.12, W * 0.25, W * 0.22, W * 0.20]
        )
        hist_table.setStyle(TableStyle([
            ("BACKGROUND",     (0, 0), (-1, 0), LGREY),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [WHITE, colors.HexColor("#FAFAFA")] * 10),
            ("BOX",            (0, 0), (-1, -1), 0.5, colors.HexColor("#E0E0E0")),
            ("INNERGRID",      (0, 0), (-1, -1), 0.3, colors.HexColor("#F0F0F0")),
            ("TOPPADDING",     (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
            ("LEFTPADDING",    (0, 0), (-1, -1), 5),
            ("ALIGN",          (0, 0), (0, -1), "CENTER"),
        ]))
        story.append(hist_table)

    # ── Conclusion ───────────────────────────────────────
    story.append(Spacer(1, 0.5 * cm))
    concl_bg = colors.HexColor("#FFF5F5") if prediction == 1 else colors.HexColor("#F0FFF4")
    concl_border = RED if prediction == 1 else GREEN

    if prediction == 1:
        concl_txt = (
            "L'analyse par intelligence artificielle indique un <b>risque élevé</b> "
            f"de maladie cardiaque (probabilité : {probability:.1%}). "
            "Une consultation cardiologique urgente est fortement recommandée. "
            "Ce rapport ne remplace pas un avis médical professionnel."
        )
    else:
        concl_txt = (
            "L'analyse par intelligence artificielle indique une <b>faible probabilité</b> "
            f"de maladie cardiaque (probabilité : {probability:.1%}). "
            "Il est néanmoins conseillé de maintenir un suivi médical régulier "
            "et un mode de vie sain."
        )

    concl_data = [[Paragraph(f"💡 Conclusion — {concl_txt}", S["body"])]]
    concl_table = Table(concl_data, colWidths=[W])
    concl_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), concl_bg),
        ("BOX",           (0, 0), (-1, -1), 1.5, concl_border),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(concl_table)

    # ── Build ────────────────────────────────────────────
    doc.build(story, canvasmaker=NumberedCanvas)
    return buf.getvalue()
