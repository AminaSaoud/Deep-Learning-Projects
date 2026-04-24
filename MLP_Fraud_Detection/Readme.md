# Détection de Fraude Bancaire avec MLP

> **Dataset** : `dataset_fraude.csv` — 5 228 transactions bancaires  
> **Framework** : TensorFlow / Keras | Scikit-learn


---

## Dataset

| Variable | Type | Description |
|----------|------|-------------|
| `amount` | numérique | Montant de la transaction |
| `hour` | numérique | Heure de la transaction |
| `merchant_category` | catégorielle | Catégorie du commerçant |
| `country` | catégorielle | Pays de la transaction |
| `country_risk` | numérique | Score de risque pays |
| `distance_km` | numérique | Distance depuis la transaction précédente |
| `tx_last_1h / 24h` | numérique | Nombre de transactions récentes |
| `device_change` | binaire | Changement d'appareil |
| `is_international` | binaire | Transaction internationale |
| `label_fraud` | **cible** | 0 = Normal / 1 = Fraude |

---

## Contenu du notebook

| Étape | Description |
|-------|-------------|
| **Q1** | Chargement du dataset — dimensions et aperçu |
| **Q2** | Analyse de la variable cible — distribution Fraude/Normal |
| **Q3** | Séparation features / target |
| **Q4** | One-Hot Encoding des variables catégorielles (`merchant_category`, `country`) |
| **Q5** | Split stratifié 70% train / 15% val / 15% test |
| **Q6** | Standardisation (`StandardScaler` fitté uniquement sur train) |
| **Q7** | Construction du modèle MLP (architecture ci-dessous) |
| **Q8** | Compilation — Adam, lr=1e-3, métriques accuracy + recall |
| **Q9** | Early Stopping (patience=5, restore_best_weights) |
| **Q10** | Class weight pour compenser le déséquilibre |
| **Q11** | Entraînement (200 epochs max, batch_size=64) |
| **Q12** | Courbes d'apprentissage (loss / accuracy / recall) |
| **Q13** | Évaluation finale + matrice de confusion |
| **Q14** | Analyse multi-seuils de décision |

---

## Architecture MLP

```
Input (n_features)
   ↓
Dense(64, relu)
   ↓
Dropout(0.30)
   ↓
Dense(32, relu)
   ↓
Dropout(0.20)
   ↓
Dense(1, sigmoid)   ← classification binaire
```

**Optimiseur** : Adam (lr=1e-3) | **Loss** : Binary Crossentropy


---

## Stack technique

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-API-red?logo=keras)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-f7931e?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-data-150458?logo=pandas)

---

## Lancer le notebook

```bash
# Cloner le repo
git clone https://github.com/AminaSaoud/Deep-Learning-Projects.git

cd deep-learning-projects/MLP_Fraud_Detection

# Installer les dépendances
pip install tensorflow scikit-learn pandas numpy matplotlib

# Placer le dataset dans le dossier
# dataset_fraude.csv 

# Lancer Jupyter
jupyter notebook MLP_Fraud_Detection.ipynb
```