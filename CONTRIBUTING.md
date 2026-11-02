# Contribution Claude — sandbox KHD

## Périmètre
Ce dépôt est le **sandbox public** de Claude pour KHD. Il ne contient :
- aucun poids ;
- aucun token ou credential ;
- aucune base de données ;
- aucun benchmark complet ;
- aucun secret.

## Règles
1. Modifier uniquement ce dépôt.
2. Aucune API IA externe comme cerveau.
3. Aucun benchmark dans l’entraînement.
4. Aucun score inventé.
5. Un patch = une seule hypothèse testable.
6. `python -m py_compile` doit passer.
7. Produire un diff clair et un protocole avec seuil McNemar exact.

## Workflow
1. Fork ou branche `claude/<sujet>`.
2. Modifier `scripts/` ou `schemas/`.
3. Ouvrir une Pull Request.
4. La CI vérifie la syntaxe Python.
5. Le Lead KHD révise et applique après tests et évaluation 199 cas.

## Fichiers
- `PROMPT.md` — mission.
- `scripts/train_gen11.py` — entraîneur récurrent.
- `scripts/eval_gen11.py` — évaluateur.
- `schemas/family-pair.schema.json` — schéma famille/trace/réponse.
- `docs/` — décisions et statistiques.

## Interdictions
- Ne pas pousser de poids, données privées, clés, emails ou secrets.
- Ne pas modifier le dépôt principal `KHD`.
- Ne pas promettre un gain sans mesure.
