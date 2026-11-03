# Où est le code — pour Claude

## Dépôt public accessible
https://github.com/KHD-Dev-2/khd-claude-sandbox

## Fichiers à corriger en priorité
| Fichier | Rôle | URL |
|---|---|---|
| `scripts/eval_gen11.py` | évaluateur, v1 figée + v2, McNemar, `--selftest`, `--rescore` | https://github.com/KHD-Dev-2/khd-claude-sandbox/blob/main/scripts/eval_gen11.py |
| `scripts/train_gen11.py` | entraîneur récurrent Gen11 | https://github.com/KHD-Dev-2/khd-claude-sandbox/blob/main/scripts/train_gen11.py |
| `schemas/family-pair.schema.json` | schéma famille/trace/réponse | https://github.com/KHD-Dev-2/khd-claude-sandbox/blob/main/schemas/family-pair.schema.json |
| `PROPOSAL.md` | protocole, seuils, résultats | https://github.com/KHD-Dev-2/khd-claude-sandbox/blob/main/PROPOSAL.md |

## Contexte local
- Sandbox : `C:\KHD\claude-sandbox`
- Production KHD : `C:\KHD` — **non accessible à Claude, ne pas modifier**
- Branches : `main` du sandbox uniquement
- Dernier commit sandbox : `d720245`

## Copies de production (référence seulement)
- `C:\KHD\deploy\kaggle\khd-gen11-recurrent-20261103\script.py`
- `C:\KHD\deploy\kaggle\khd-eval-gen11-20261103\script.py`
- `C:\KHD\schemas\family-pair.schema.json`
- `C:\KHD\reports\ai-finalization\D1_D2_DECISION_20261103.md`
- `C:\KHD\reports\ai-finalization\CLAUDE_REVIEW_ACTION_20261103.md`

Ces copies servent uniquement à comprendre le contexte. Claude doit modifier le sandbox, pas `C:\KHD`.

## Données et poids
Non fournis à Claude :
- checkpoints Gen4–Gen12 ;
- `data/` ;
- bases SQLite ;
- benchmark complet ;
- tokens et secrets ;
- fichiers `.env`.

Pourquoi : ils sont volumineux, privés ou sensibles. Claude peut demander des extraits anonymisés si nécessaire.

## Interdictions
- Ne pas pousser de poids, données privées, clé ou secret.
- Ne pas modifier le dépôt privé KHD.
- Ne pas appeler une API IA externe comme cerveau.
- Ne pas modifier le benchmark après avoir vu les scores.
- Ne pas inventer un résultat.

## Livrable
Branche `claude/<sujet>` + PR + `PROPOSAL.md`.
