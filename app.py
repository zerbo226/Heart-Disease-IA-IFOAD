# =========================================================
# HEART DISEASE AI - ULTRA UI VERSION
# Version adaptative - fonctionne avec n'importe quel dataset
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix
)

# Import du module PDF
try:
    from generate_report import generate_pdf_report
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Heart Disease AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>
.stApp { background: #f5f7fa; }
html, body, [class*="css"] { color: #1a1a2e; }
h1 { color: #1a1a2e !important; font-weight: 700; }
h2 { color: #1a1a2e !important; font-weight: 600; }
h3 { color: #1a1a2e !important; font-weight: 600; }
h4, h5, h6 { color: #2d3748 !important; }
p { color: #2d3748 !important; }

.glass {
    background: #ffffff;
    border-radius: 20px;
    padding: 30px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.live-indicator {
    display: inline-block;
    width: 10px; height: 10px;
    background: #00c853;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(0,200,83,0.5); }
    70%  { box-shadow: 0 0 0 8px rgba(0,200,83,0); }
    100% { box-shadow: 0 0 0 0 rgba(0,200,83,0); }
}

.pdf-box {
    background: linear-gradient(135deg, #1a1a2e, #2d3748);
    border-radius: 16px;
    padding: 20px 24px;
    margin-top: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
}

.stTabs [data-baseweb="tab-list"] { gap: 10px; }
.stTabs [data-baseweb="tab"] {
    background: #ffffff; border-radius: 12px;
    color: #2d3748 !important; padding: 10px 20px;
    border: 1px solid #e2e8f0; font-weight: 500;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.stTabs [aria-selected="true"] {
    background: #e63946 !important;
    color: #ffffff !important;
    border: 1px solid #e63946;
    font-weight: 600;
    box-shadow: 0 4px 15px rgba(230,57,70,0.3);
}

.stButton>button {
    background: #e63946; color: white !important;
    border: none; border-radius: 12px; height: 50px;
    font-size: 16px; font-weight: 600; transition: 0.3s;
    box-shadow: 0 4px 15px rgba(230,57,70,0.3);
}
.stButton>button:hover {
    background: #c1121f;
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(230,57,70,0.4);
}

[data-testid="metric-container"] {
    background: #ffffff; border-radius: 15px;
    padding: 20px; border: 1px solid #e2e8f0;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}
[data-testid="metric-container"] label { color: #718096 !important; font-size: 14px; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #1a1a2e !important; font-size: 28px; font-weight: 700;
}

.stSelectbox label, .stNumberInput label {
    color: #1a1a2e !important; font-weight: 500; font-size: 14px;
}
.stSelectbox [data-baseweb="select"] {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px;
}
.stSelectbox [data-baseweb="select"] div { color: #1a1a2e !important; }
.stNumberInput input {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 10px; color: #1a1a2e !important; padding: 10px;
}

section[data-testid="stSidebar"] {
    background: #1a1a2e;
    border-right: 1px solid #2d3748;
}
section[data-testid="stSidebar"] * { color: #ffffff !important; }
section[data-testid="stSidebar"] h1 { color: #e63946 !important; }

[data-testid="stDataFrame"] {
    border-radius: 15px; overflow: hidden;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}
[data-testid="stDataFrame"] th {
    background: #1a1a2e !important; color: #ffffff !important;
    font-weight: 600; padding: 12px;
}
[data-testid="stDataFrame"] td {
    background: #ffffff; color: #1a1a2e !important;
    padding: 10px; border-bottom: 1px solid #f0f0f0;
}

::-webkit-scrollbar { width: 10px; }
::-webkit-scrollbar-track { background: #f5f7fa; }
::-webkit-scrollbar-thumb { background: #cbd5e0; border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: #e63946; }
code { color: #e63946 !important; background: #f5f7fa; padding: 3px 8px; border-radius: 5px; }
hr { border-color: #e2e8f0; margin: 20px 0; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "new_patients" not in st.session_state:
    st.session_state.new_patients = None

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("# ❤️ HEART AI")
    st.markdown("### 🧠 Machine Learning")
    st.markdown("### 📊 Dashboard Medical")
    st.markdown("### 🩺 Analyse Cardiaque")
    st.markdown("### 🤖 Prediction IA")
    st.markdown("### 📄 Export PDF")
    st.markdown("---")

    n_new_sidebar = len(st.session_state.new_patients) if st.session_state.new_patients is not None else 0
    if n_new_sidebar > 0:
        st.markdown(f"""
        <div style="background:rgba(230,57,70,0.15);border:1px solid #e63946;
                    border-radius:12px;padding:12px;margin:8px 0;">
          <div class="live-indicator"></div>
          <span style="color:white;font-weight:600;font-size:0.85rem;">
            {n_new_sidebar} nouveau(x) patient(s)
          </span><br>
          <span style="color:rgba(255,255,255,0.6);font-size:0.72rem;">
            ajouté(s) au dashboard
          </span>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🗑️ Réinitialiser patients", use_container_width=True):
            st.session_state.new_patients = None
            st.session_state.history = []
            st.session_state.last_prediction = None
            st.rerun()

    st.markdown("---")
    if st.session_state.last_prediction:
        lp = st.session_state.last_prediction
        color = "#e63946" if lp["prediction"] == 1 else "#00c853"
        label = "Malade" if lp["prediction"] == 1 else "Sain"
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.05);border-radius:12px;padding:12px;margin:8px 0;">
          <div style="font-size:0.7rem;color:rgba(255,255,255,0.5);margin-bottom:4px;">
            Dernier patient analysé
          </div>
          <div style="color:{color};font-weight:700;font-size:0.9rem;">{label}</div>
          <div style="color:rgba(255,255,255,0.7);font-size:0.75rem;">
            {lp['patient_data']['age']} ans · {lp['probability']:.1%} risque
          </div>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="glass">
  <h1 style='text-align:center;font-size:50px;color:#1a1a2e;'>❤️ HEART DISEASE AI</h1>
  <p style='text-align:center;font-size:18px;color:#4a5568;'>
    Plateforme intelligente de prédiction des maladies cardiaques avec Intelligence Artificielle
  </p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# CHARGEMENT DES DONNÉES (VERSION ADAPTATIVE)
# =========================================================

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("heart_disease_data.csv")
        
        # Nettoyage des colonnes
        df = df.drop(columns=['id', 'dataset'], errors='ignore')
        
        # Gestion de la colonne target
        if 'num' in df.columns and 'target' not in df.columns:
            df['target'] = (df['num'] > 0).astype(int)
            df = df.drop(columns=['num'])
        
        # Renommage des colonnes si nécessaire
        rename_mapping = {
            'thalch': 'thalach',
            'trestbps': 'trestbps',
            'restecg': 'restecg',
            'thalach': 'thalach',
            'exang': 'exang',
            'oldpeak': 'oldpeak',
            'ca': 'ca',
            'thal': 'thal'
        }
        df = df.rename(columns=rename_mapping)
        
        # Conversion des types string vers numériques
        if 'sex' in df.columns and df['sex'].dtype == object:
            df['sex'] = df['sex'].map({'Male': 1, 'Female': 0, 'male': 1, 'female': 0})
        if 'cp' in df.columns and df['cp'].dtype == object:
            df['cp'] = df['cp'].str.lower().str.strip().map(
                {'typical angina': 0, 'atypical angina': 1, 'non-anginal': 2, 'asymptomatic': 3,
                 'typical': 0, 'atypical': 1})
        if 'fbs' in df.columns and df['fbs'].dtype == object:
            df['fbs'] = df['fbs'].map({True: 1, False: 0, 'True': 1, 'False': 0, 'TRUE': 1, 'FALSE': 0})
        if 'restecg' in df.columns and df['restecg'].dtype == object:
            df['restecg'] = df['restecg'].str.lower().str.strip().map(
                {'normal': 0, 'st-t wave abnormality': 1, 'lv hypertrophy': 2,
                 'left ventricular hypertrophy': 2})
        if 'exang' in df.columns and df['exang'].dtype == object:
            df['exang'] = df['exang'].map({True: 1, False: 0, 'True': 1, 'False': 0, 'Yes': 1, 'No': 0})
        if 'slope' in df.columns and df['slope'].dtype == object:
            df['slope'] = df['slope'].str.lower().str.strip().map(
                {'upsloping': 0, 'flat': 1, 'downsloping': 2})
        if 'thal' in df.columns and df['thal'].dtype == object:
            df['thal'] = df['thal'].str.lower().str.strip().map(
                {'normal': 3, 'fixed defect': 6, 'reversable defect': 7, 'reversible defect': 7})
        
        # Conversion en numérique
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
        
    except Exception as e:
        st.error(f"Erreur de chargement du fichier CSV: {e}")
        # Dataset de secours
        st.warning("Utilisation d'un dataset de démonstration")
        return pd.DataFrame({
            'age': [50, 60, 45, 55, 65],
            'sex': [1, 0, 1, 1, 0],
            'cp': [0, 1, 2, 0, 1],
            'trestbps': [120, 140, 130, 135, 125],
            'chol': [200, 250, 180, 220, 210],
            'fbs': [0, 1, 0, 0, 1],
            'restecg': [0, 1, 0, 1, 0],
            'thalach': [150, 140, 160, 145, 155],
            'exang': [0, 1, 0, 0, 1],
            'oldpeak': [1.0, 2.0, 0.5, 1.5, 2.5],
            'slope': [1, 2, 1, 2, 1],
            'ca': [0, 1, 0, 1, 0],
            'thal': [3, 6, 3, 7, 3],
            'target': [0, 1, 0, 1, 1]
        })

@st.cache_data
def train_models(df):
    # Liste des colonnes features attendues
    expected_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                         'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    # Garder uniquement les colonnes qui existent
    available_features = [col for col in expected_features if col in df.columns]
    
    # Vérification de la colonne target
    if 'target' not in df.columns:
        raise ValueError("La colonne 'target' est manquante dans le fichier CSV")
    
    # Afficher les colonnes utilisées
    st.info(f"📊 {len(available_features)} features utilisées pour l'entraînement : {available_features}")
    
    # Préparation des données
    X = df[available_features].copy()
    y = df['target'].copy()
    
    # Imputation des valeurs manquantes
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=available_features)
    
    # Division entraînement/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Normalisation
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    # Modèles
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(probability=True),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'AdaBoost': AdaBoostClassifier(),
    }
    
    results, trained = [], {}
    for name, model in models.items():
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        y_prob = model.predict_proba(X_test_sc)[:, 1]
        results.append({
            'Modele': name,
            'Accuracy': round(accuracy_score(y_test, y_pred), 4),
            'Precision': round(precision_score(y_test, y_pred), 4),
            'Recall': round(recall_score(y_test, y_pred), 4),
            'F1': round(f1_score(y_test, y_pred), 4),
            'AUC': round(roc_auc_score(y_test, y_prob), 4)
        })
        trained[name] = (model, y_prob, y_pred)
    
    return pd.DataFrame(results), trained, scaler, imputer, X_test_sc, y_test, available_features

# =========================================================
# CHARGEMENT PRINCIPAL
# =========================================================

df_original = load_data()

# Affichage des colonnes disponibles
with st.expander("📋 Informations sur le dataset", expanded=False):
    st.write("**Colonnes disponibles :**")
    st.write(df_original.columns.tolist())
    st.write(f"**Nombre total de patients :** {len(df_original)}")
    st.write(f"**Nombre de patients malades :** {df_original['target'].sum()}")

# Calcul des bases pour les métriques
AGE_BASE = df_original['age'].mean()
CHOL_BASE = df_original['chol'].mean()

# Entraînement des modèles
results_df, trained_models, scaler_train, imputer_train, X_test_sc, y_test, feature_names = train_models(df_original)

best_model = results_df.sort_values('AUC', ascending=False).iloc[0]['Modele']
hex_colors = ['#e63946', '#3498db', '#00c853', '#f39c12', '#9b59b6', '#1abc9c']

# =========================================================
# HELPERS
# =========================================================

def get_live_df():
    if st.session_state.new_patients is not None and len(st.session_state.new_patients) > 0:
        return pd.concat([df_original, st.session_state.new_patients], ignore_index=True)
    return df_original.copy()

def plotly_base(fig, h=400):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#1a1a2e'), height=h,
        xaxis=dict(gridcolor='rgba(0,0,0,0.08)'),
        yaxis=dict(gridcolor='rgba(0,0,0,0.08)'),
        legend=dict(font=dict(color='#1a1a2e'))
    )
    return fig

def build_metrics_dict():
    return {
        row['Modele']: {
            'Accuracy': row['Accuracy'],
            'Precision': row['Precision'],
            'Recall': row['Recall'],
            'F1': row['F1'],
            'AUC': row['AUC'],
        }
        for _, row in results_df.iterrows()
    }

COLORS = {0: '#00c853', 1: '#e63946'}

# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "❤️ Prédiction",
    "📊 Dashboard",
    "🤖 Comparaison IA",
    "🧠 Analyse IA",
    "📈 Courbes",
    "📁 Dataset"
])

# =========================================================
# TAB 1 — PRÉDICTION + PDF
# =========================================================

with tab1:
    st.header("🩺 Prédiction médicale")

    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Âge", 1, 120, 50)
        sex = st.selectbox("Sexe", ["Homme", "Femme"])
        cp = st.selectbox("Douleur thoracique",
                          ["Typique", "Atypique", "Non-anginale", "Asymptomatique"])
        trestbps = st.number_input("Pression artérielle", 50, 250, 120)
        chol = st.number_input("Cholestérol", 50, 700, 200)

    with c2:
        fbs = st.selectbox("Glycémie > 120", ["Non", "Oui"])
        restecg = st.selectbox("ECG", ["Normal", "ST-T", "Hypertrophie"])
        thalach = st.number_input("Fréquence max", 50, 250, 150)
        exang = st.selectbox("Angine exercice", ["Non", "Oui"])
        oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)

    with c3:
        slope = st.selectbox("Pente ST", ["Ascendante", "Plate", "Descendante"])
        ca = st.number_input("Vaisseaux", 0, 4, 1)
        thal = st.selectbox("Thal", ["Normal", "Fixe", "Réversible"])
        model_choice = st.selectbox("Modèle IA", list(trained_models.keys()),
                                    index=list(trained_models.keys()).index(best_model))
        st.info(f"⭐ Meilleur modèle : **{best_model}**")

    # Encodage
    sex_num = 1 if sex == "Homme" else 0
    cp_num = {"Typique": 0, "Atypique": 1, "Non-anginale": 2, "Asymptomatique": 3}[cp]
    fbs_num = 1 if fbs == "Oui" else 0
    restecg_num = {"Normal": 0, "ST-T": 1, "Hypertrophie": 2}[restecg]
    exang_num = 1 if exang == "Oui" else 0
    slope_num = {"Ascendante": 0, "Plate": 1, "Descendante": 2}[slope]
    thal_num = {"Normal": 3, "Fixe": 6, "Réversible": 7}[thal]

    # Création du dictionnaire des entrées
    input_dict = {
        'age': age, 'sex': sex_num, 'cp': cp_num, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs_num, 'restecg': restecg_num, 'thalach': thalach,
        'exang': exang_num, 'oldpeak': oldpeak, 'slope': slope_num, 'ca': ca, 'thal': thal_num
    }
    
    # Filtrer selon les features disponibles
    input_dict_filtered = {k: v for k, v in input_dict.items() if k in feature_names}
    input_df = pd.DataFrame([list(input_dict_filtered.values())], 
                            columns=list(input_dict_filtered.keys()))
    
    # Prédiction
    input_imputed = imputer_train.transform(input_df)
    input_scaled = scaler_train.transform(input_imputed)

    if st.button("❤️ Prédire maintenant", use_container_width=True):
        model, _, _ = trained_models[model_choice]
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        # Historique
        st.session_state.history.append({
            "Age": age,
            "Sexe": "Homme" if sex_num == 1 else "Femme",
            "Cholestérol": chol,
            "Modèle": model_choice,
            "Prédiction": "Malade" if prediction == 1 else "Sain",
            "Probabilité": f"{probability:.2%}"
        })

        # Nouveau patient
        new_row_data = [input_dict[f] for f in feature_names] + [int(prediction)]
        new_row = pd.DataFrame([new_row_data], columns=feature_names + ['target'])
        
        if st.session_state.new_patients is None:
            st.session_state.new_patients = new_row
        else:
            st.session_state.new_patients = pd.concat(
                [st.session_state.new_patients, new_row], ignore_index=True)

        # Stockage pour PDF
        st.session_state.last_prediction = {
            "patient_data": input_dict,
            "prediction": int(prediction),
            "probability": float(probability),
            "model_name": model_choice,
        }

        # Affichage des résultats
        r1, r2 = st.columns(2)
        with r1:
            if prediction == 0:
                st.success("✅ Aucune maladie détectée")
            else:
                st.error("⚠️ Maladie cardiaque détectée")
            st.metric("Probabilité de risque", f"{probability:.2%}")

        with r2:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={'text': "Risque cardiaque (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': '#e63946' if prediction == 1 else '#00c853'},
                    'steps': [{'range': [0, 33], 'color': '#f0fff4'},
                              {'range': [33, 66], 'color': '#fffff0'},
                              {'range': [66, 100], 'color': '#fff5f5'}],
                    'threshold': {'line': {'color': '#e63946', 'width': 3},
                                  'thickness': 0.8, 'value': probability * 100}
                }
            ))
            fig_g.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='#1a1a2e'), height=250)
            st.plotly_chart(fig_g, use_container_width=True)

    # Historique
    if st.session_state.history:
        st.markdown("---")
        st.subheader(f"📋 Historique des prédictions ({len(st.session_state.history)} patients)")
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, height=200)

    # Export PDF
    st.markdown("---")
    st.markdown("""
    <div class="pdf-box">
      <div style="color:white;font-size:1.1rem;font-weight:700;margin-bottom:6px;">
        📄 Rapport PDF médical professionnel
      </div>
      <div style="color:rgba(255,255,255,0.65);font-size:0.82rem;">
        Génère un rapport complet 2 pages : données patient · diagnostic IA ·
        interprétation médicale · analyse du dataset · comparaison des modèles · historique.
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not PDF_AVAILABLE:
        st.warning("⚠️ Module `generate_report.py` introuvable.")
    elif st.session_state.last_prediction is None:
        st.info("ℹ️ Effectuez d'abord une prédiction ci-dessus.")
    else:
        lp = st.session_state.last_prediction
        if st.button("📄 Générer le rapport PDF", use_container_width=True):
            with st.spinner("⏳ Génération..."):
                try:
                    pdf_bytes = generate_pdf_report(
                        patient_data=lp["patient_data"],
                        prediction=lp["prediction"],
                        probability=lp["probability"],
                        model_name=lp["model_name"],
                        metrics_dict=build_metrics_dict(),
                        best_model_name=best_model,
                        df_live=get_live_df(),
                        history=st.session_state.history
                    )
                    fname = f"rapport_cardioai_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    st.download_button("⬇️ Télécharger", pdf_bytes, fname, "application/pdf")
                except Exception as e:
                    st.error(f"Erreur: {e}")

# =========================================================
# TAB 2 à 6 (Dashboard, Comparaison, Analyse, Courbes, Dataset)
# =========================================================
# [Le reste du code pour les tabs 2-6 reste identique]
# Pour éviter la répétition, je garde la structure mais vous pouvez 
# copier-coller les tabs 2-6 du code précédent qui fonctionnent déjà
