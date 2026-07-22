# Prédiction du churn client — Banque

Prédire si un client va quitter la banque (churn) à partir de ses données démographiques et de son comportement bancaire, pour permettre d'anticiper les départs plutôt que de les constater.

## Contexte

Un client qui part coûte plus cher à remplacer qu'à retenir. L'objectif de ce projet est de construire un modèle capable d'identifier, en amont, les clients à risque de clôturer leur compte, à partir de données historiques.

## Données

- ~165 000 clients en entraînement, avec les variables : score de crédit, pays, genre, âge, ancienneté, solde, nombre de produits détenus, carte de crédit, statut d'activité, salaire estimé, et la cible `Exited` (0 = reste, 1 = part).
- Cible déséquilibrée : environ 21 % de churn contre 79 % de clients fidèles.

## Démarche

1. **Analyse exploratoire structurée** en cinq temps : univariée, bivariée discrète-discrète, discrète-continue, continue-continue, puis multivariée (matrice de corrélation), avec interprétation métier à chaque étape (ex. taux de churn nettement plus élevé chez les clientes en Allemagne, lien entre âge et probabilité de départ).
2. **Tests d'hypothèses formalisés** (H0, seuil α = 0.05) pour valider statistiquement les intuitions de l'EDA :
   - Test du Chi² : le pays influence significativement le churn (p < 0.001)
   - Test de Student : le solde moyen diffère significativement entre clients fidèles et partants (p < 0.001)
   - Test de Pearson : corrélation significative mais très faible entre âge et score de crédit (r ≈ -0.01)
3. **Préparation des données** : traitement des valeurs aberrantes par plafonnement IQR (âge, score de crédit, solde), encodage (Label Encoding pour le genre, One-Hot pour le pays), standardisation des variables numériques.
4. **Comparaison de modèles** avec validation croisée 5-fold (score F1, adapté à une cible déséquilibrée) :
   - Régression logistique (référence)
   - Random Forest avec recherche par grille (`GridSearchCV`)
   - XGBoost avec recherche aléatoire d'hyperparamètres (`RandomizedSearchCV`)
   - LightGBM avec recherche aléatoire d'hyperparamètres
5. **Évaluation finale** du meilleur modèle sur un jeu de test isolé (20 % des données, non vu à l'entraînement).

## Résultats

| Modèle | F1-score (validation croisée) |
|---|---|
| Régression logistique | 0.52 |
| XGBoost (meilleure configuration) | 0.63 |

Évaluation du modèle XGBoost retenu sur le jeu de test :

| | Précision | Rappel | F1-score |
|---|---|---|---|
| Client fidèle | 0.90 | 0.96 | 0.93 |
| Client en churn | 0.82 | 0.60 | 0.69 |

Accuracy globale : 0.89.

Le modèle identifie correctement 60 % des clients qui vont réellement partir, avec 82 % de précision sur ces prédictions : il est plus prudent que exhaustif, ce qui est cohérent avec un usage où une fausse alerte coûte une action de rétention inutile, alors qu'un départ manqué coûte le client.

## Stack technique

Python, pandas, scikit-learn, XGBoost, LightGBM, seaborn/matplotlib, SciPy (tests statistiques).

## Pistes d'amélioration

- Traiter le déséquilibre de classes en amont (SMOTE, pondération des classes) plutôt que de compter uniquement sur le F1-score.
- Pousser la recherche d'hyperparamètres LightGBM jusqu'au bout pour la comparer équitablement aux autres modèles.
- Étudier l'importance des variables (SHAP) pour rendre les prédictions du modèle final interprétables auprès d'un métier.
