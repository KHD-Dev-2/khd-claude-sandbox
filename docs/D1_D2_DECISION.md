# KHD — diagnostics D1/D2 et décision finale (03/11/2026)

## D1 — concentration
Mesure proxy sur les **24 001 paires Gen7** :
- **3 526** signatures de surface.
- **N_eff ≈ 94** (entropie de la distribution).
- **Part maximale : 9,75 %**.
- Top : « calcule # # avec étapes » = 2 340 paires.

Interprétation : la couverture est concentrée. Mais D1 seul ne décide pas.

## D2 — exécution dans la distribution
200 nouveaux cas générés depuis les **familles d’entraînement Gen7**, avec de nouveaux nombres :
- Gen6 : **0/200**
- Gen7 : **0/200**

Le modèle ne réussit pas les procédures déjà vues. Ce n’est donc pas un simple manque de nouvelles familles.

## Décision selon l’arbre de Claude
- D2 < 50 % → la diversité n’est pas la priorité.
- Variante B (traces) : **déjà exécutée avec Gen7**, score 7/199, non significatif.
- Aucune expérience supplémentaire justifiée.

## Statistiques corrigées
Test exact de McNemar :
| Comparaison | Discordants | p exact |
|---|---|---:|
| Gen6 vs Gen7 | 4 / 5 | **1,000** |
| Gen6 vs Gen8 | 0 / 3 | **0,250** |

Les anciens p = 0,739 et 0,083 étaient des approximations khi-deux, invalides avec si peu de paires.

## Verdict
- **Arrêt définitif** de la voie « ≤30 M, from scratch, réponse seule, 0 € » pour le raisonnement scientifique général.
- Aucun modèle ne dépasse **8/199** de façon significative.
- Gen6 reste le meilleur candidat observé, sans preuve statistique.
- Le serveur Render reste sur **Gen3d**, qui n’est pas un modèle de raisonnement scientifique général.

## Ce qui serait nécessaire
- Un pré-entraînement maison massif, ou
- un périmètre étroit avec une procédure à 1–2 étapes et D2 ≥ 90 %, ou
- davantage de moyens de calcul.
Aucun de ces trois n’est disponible à **0 €** aujourd’hui.

## Coût
**0 €.**

## Git
- Branche : `deep/ai-finalization-20261101`
- Commit à suivre.
