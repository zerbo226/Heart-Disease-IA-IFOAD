# ❤️ CardioAI — Prédiction des Maladies Cardiaques

## 📋 Présentation

Projet réalisé dans le cadre du cours **Intelligence Artificielle**
de l'**IFOAD**, sous la direction du **Dr Arthur Sawadogo**.

Application web interactive de prédiction des maladies cardiaques basée sur
6 algorithmes de Machine Learning, développée avec **Streamlit**.

---

## 🎯 Objectif

Appliquer des techniques de Machine Learning pour prédire la présence de maladies
cardiaques à partir de données cliniques issues du dataset **Heart Disease UCI**.

---

## ✨ Fonctionnalités

- 🩺 **Prédiction en temps réel** — saisie des données patient et résultat instantané avec jauge de risque
- 📊 **Dashboard dynamique** — les nouveaux patients prédits s'ajoutent automatiquement aux statistiques
- 🤖 **Comparaison de 6 algorithmes** — tableau de métriques, radar, courbes ROC
- 🧠 **Analyse IA** — importance des features, matrice de confusion, corrélations
- 📈 **Courbes interactives** — nuage de points, violons, distributions
- 📁 **Dataset enrichi** — téléchargement du CSV incluant les nouvelles prédictions
- 📄 **Export PDF médical** — rapport professionnel 2 pages avec diagnostic, interprétation médicale et comparaison des 6 modèles

---

## 🤖 Algorithmes utilisés

| Algorithme | Description |
|---|---|
| Logistic Regression | Régression logistique |
| KNN | K-Nearest Neighbors |
| SVM | Support Vector Machine |
| Decision Tree | Arbre de décision |
| Random Forest | Forêt aléatoire |
| AdaBoost | Boosting adaptatif |

---

## 📊 Métriques d'évaluation

- Accuracy · Précision · Rappel · F1-Score · AUC-ROC

---

## 📁 Structure du projet

```
Heart-Disease-IA/
├── app.py                      # Application Streamlit principale
├── generate_report.py          # Module export PDF médical professionnel
├── heart_disease_data.csv      # Dataset Heart Disease UCI
├── IFOAD_Heart_Disease.ipynb   # Notebook Jupyter (EDA + modèles)
├── requirements.txt            # Dépendances Python
└── readme.md                   # Ce fichier
```

---

## ⚙️ Installation et lancement

### 1. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 2. Lancer l'application

```bash
streamlit run app.py
```

### 3. Ouvrir dans le navigateur

```
http://localhost:8501
```

---

## 📦 Dataset

**Heart Disease UCI**
- Source : [archive.ics.uci.edu](https://archive.ics.uci.edu/dataset/45/heart+disease)
- 300 patients · 13 variables · 2 classes (Sain / Malade)

| Variable | Description |
|---|---|
| age | Âge en années |
| sex | Sexe (1=Homme, 0=Femme) |
| cp | Type de douleur thoracique (0-3) |
| trestbps | Pression artérielle au repos (mm Hg) |
| chol | Cholestérol sérique (mg/dl) |
| fbs | Glycémie à jeun > 120 mg/dl |
| restecg | Résultats ECG au repos |
| thalach | Fréquence cardiaque maximale |
| exang | Angine induite par l'exercice |
| oldpeak | Dépression ST |
| slope | Pente du segment ST |
| ca | Nombre de vaisseaux majeurs (0-3) |
| thal | Thalassémie |
| target | Maladie cardiaque (1=Oui, 0=Non) |

---

## 🛠️ Technologies

- **Python 3.10**
- **Streamlit** — Interface web
- **Scikit-learn** — Algorithmes ML
- **Plotly** — Visualisations interactives
- **ReportLab + Matplotlib** — Export PDF médical
- **Pandas / NumPy** — Traitement des données

---

## 👨‍💻 Auteurs

Projet IFOAD — KY LAURENT && ILBOUDO VANESSA
