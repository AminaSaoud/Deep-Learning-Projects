# Prétraitement d'Images & CNN sur MNIST

> **Dataset** : MNIST (chiffres manuscrits 0–9, images 28×28 niveaux de gris)  
> **Framework** : TensorFlow / Keras

---

## Contenu du notebook

| Étape | Description |
|-------|-------------|
| **Q1** | Chargement MNIST — split Train / Val / Test (50k / 10k / 10k) |
| **Q2** | Normalisation pixels → `float32 / 255` (valeurs dans [0, 1]) |
| **Q3** | Reshape `(N, 28, 28)` → `(N, 28, 28, 1)` pour entrée CNN 4D |
| **Q4** | `ImageDataGenerator` sans augmentation — visualisation grille 4×4 |
| **Q5** | `ImageDataGenerator` avec augmentation — comparaison avant/après |
| **Q6** | Entraînement CNN **(A) sans** et **(B) avec** augmentation |
| **Q7** | Analyse des courbes accuracy/val_accuracy et conclusions |

---

## Architecture CNN

```
Input (28, 28, 1)
   ↓
Conv2D(16, 3×3, relu) → MaxPooling2D(2×2)
   ↓
Conv2D(32, 3×3, relu) → MaxPooling2D(2×2)
   ↓
Flatten → Dense(64, relu) → Dense(10, softmax)
```

**Optimiseur** : Adam | **Loss** : SparseCategoricalCrossentropy | **Epochs** : 3

---

## Paramètres de Data Augmentation

```python
ImageDataGenerator(
    rotation_range=10,       # ±10°
    zoom_range=0.15,         # ±15%
    width_shift_range=0.10,  # décalage horizontal 10%
    height_shift_range=0.10  # décalage vertical 10%
)
```

---

## Résultats & Observations

- Le modèle **sans augmentation** monte en accuracy plus rapidement (données plus simples à mémoriser)
- Le modèle **avec augmentation** démarre plus bas mais généralise mieux sur la validation
- L'augmentation agit comme un **régulariseur implicite** en diversifiant artificiellement les exemples

---

## Stack technique

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-API-red?logo=keras)
![NumPy](https://img.shields.io/badge/NumPy-scientific-013243?logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-viz-11557c)

---


> Le dataset MNIST est téléchargé automatiquement via `tf.keras.datasets.mnist.load_data()`

## Lancer le notebook

```bash
# Cloner le repo
git clone https://github.com/AminaSaoud/Deep-Learning-Projects.git

cd deep-learning-projects/CNN_Image_Preprocessing

# Installer les dépendances
pip install tensorflow numpy matplotlib

# Lancer Jupyter
jupyter notebook CNN_Image_Preprocessing.ipynb
```

