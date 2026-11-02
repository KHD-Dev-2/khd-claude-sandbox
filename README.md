# Sandbox Claude — KHD

## Accès
Ce dossier est le **seul périmètre modifiable** par Claude :
`C:\KHD\claude-sandbox`

Il ne contient :
- aucun poids ;
- aucun token/API key ;
- aucune base de données ;
- aucun secret ;
- aucun benchmark complet ;
- aucun accès Git.

## Interdictions
- Ne pas sortir de `C:\KHD\claude-sandbox`.
- Ne pas appeler une API IA externe.
- Ne pas installer de dépendance payante.
- Ne pas supprimer de fichiers hors sandbox.
- Ne pas modifier le dépôt Git réel.
- Ne pas inventer un score ; les tests doivent être locaux ou Kaggle.

## Fichiers fournis
- `scripts/train_gen11.py` — entraîneur récurrent Gen11.
- `scripts/eval_gen11.py` — évaluateur 200 cas.
- `schemas/family-pair.schema.json` — schéma famille/trace/réponse.
- `docs/D1_D2_DECISION.md` — décision : D2 = 0/200, arrêt de la voie ≤30 M.
- `docs/FAMILY_SCHEMA.md` — schéma validé.
- `docs/CLAUDE_REVIEW_ACTION.md` — statistiques et fuite benchmark.

## Mission de Claude
Modifier directement les scripts du sandbox pour proposer **une seule** amélioration testable :
- architecture de raisonnement explicitement entraînée ;
- ou objectif avec traces vérifiables ;
- ou correction de l’évaluateur, sans changer le benchmark après avoir vu les scores.

Claude doit :
1. modifier les fichiers dans le sandbox ;
2. exécuter au minimum `python -m py_compile` ;
3. produire un **diff unifié** ;
4. expliquer le protocole et le seuil d’acceptation ;
5. ne rien promettre sans mesure.

## Application en production
Le Lead applique le diff après :
- revue ;
- tests ;
- vérification anti-fuite ;
- évaluation 200 cas ;
- commit séparé.

## Commande indicative
```powershell
cd C:\KHD\claude-sandbox
# Claude Code ou autre agent avec accès limité à ce dossier
```

Aucun identifiant n’est fourni. Aucun accès distant direct au PC n’est ouvert.
