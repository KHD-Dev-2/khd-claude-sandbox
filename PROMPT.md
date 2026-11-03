# Prompt à Claude — modification directe du sandbox KHD

Copie tout ce bloc dans Claude. Le dépôt est public ; pour pousser une branche, tu dois avoir un accès collaborateur ou travailler depuis un fork.

---

Tu es ingénieur-chercheur en modèles de langage. Tu vas modifier directement le sandbox KHD sur GitHub.

## Dépôt
https://github.com/KHD-Dev-2/khd-claude-sandbox

```bash
git clone https://github.com/KHD-Dev-2/khd-claude-sandbox.git
cd khd-claude-sandbox
git checkout -b claude/reasoning-v1
```

Si tu n’as pas le droit d’écriture :
- fork le dépôt ;
- pousse `claude/reasoning-v1` sur ton fork ;
- ouvre une Pull Request vers `KHD-Dev-2/khd-claude-sandbox:main`.

Ne modifie jamais `main` directement.

## Faits vérifiés à respecter
- KHD est un petit modèle neuronal from scratch.
- Benchmark officiel : **199 cas**, après exclusion de `bench-155` qui fuitait.
- Gen6 : **8/199**, IC Wilson 2,1–7,7 %.
- Gen7 : 7/199, McNemar exact **p = 1,000**.
- Gen8 : 5/199, p exact **0,250**.
- D1 proxy : N_eff = 94, part maximale 9,75 %.
- D2 : **0/200** pour Gen6 et Gen7 sur des cas issus de leurs propres familles avec de nouveaux nombres.
- Conclusion : ajouter des familles ou des steps ne suffit pas ; le modèle n’exécute pas les procédures vues.

## Mission
Choisis **une seule** amélioration testable et implémente-la dans le sandbox :

### Option A — objectif avec traces vérifiables
- Cible : trace d’opérations atomiques puis réponse finale.
- Chaque étape doit être recalculable.
- Le modèle doit produire la réponse finale lui-même.
- Aucun solveur ne doit produire la réponse au moment de l’inférence.

### Option B — architecture de raisonnement explicite
- Calcul multi-étapes, récurrence, pause tokens ou équivalent.
- ≤30 M paramètres.
- Entraînable sur T4 gratuit.
- Aucun appel API externe.

### Option C — correction de l’évaluateur uniquement
- Uniquement si tu prouves un bug reproductible.
- Ne modifie pas le benchmark après avoir vu les scores.
- Le benchmark reste figé par empreinte.

## Fichiers modifiables
- `scripts/train_gen11.py`
- `scripts/eval_gen11.py`
- `schemas/family-pair.schema.json`
- nouveau fichier `PROPOSAL.md`
- nouveau fichier `diff.patch` si utile

Ne touche pas :
- aux poids ;
- aux bases ;
- aux secrets ;
- au dépôt principal `KHD` ;
- aux 199 cas du benchmark.

## Contraintes absolues
- 0 €, aucun GPU payant, aucun compte payant.
- Aucune API Mistral/Qwen/OpenAI/Claude comme cerveau.
- Aucun benchmark dans l’entraînement.
- Aucun score inventé.
- Aucune promesse sans mesure.
- Aucune suppression de checkpoint.
- `python -m py_compile` doit passer.
- Ne pousse ni poids, ni données privées, ni token.

## Livrable obligatoire
1. Branche `claude/reasoning-v1`.
2. Code modifié.
3. `PROPOSAL.md` contenant :
   - hypothèse ;
   - changement exact ;
   - données ;
   - architecture ;
   - hyperparamètres ;
   - protocole T4 ;
   - seuil d’acceptation ;
   - seuil d’arrêt.
4. `git diff` ou PR.
5. Aucune affirmation de gain avant évaluation.

## Seuils imposés
- Évaluation sur les **199 cas**, extracteur figé.
- Test **McNemar exact bilatéral** contre Gen6.
- Acceptation : **p < 0,05** et score **≥16/199**.
- Non concluant : 11–15/199 ou p ≥ 0,05.
- Arrêt : **≤10/199**.
- Si tu proposes une architecture, tu dois expliquer pourquoi elle corrige D2 et pas seulement la loss.

## Réponse attendue
- Lien de la branche ou de la PR.
- Résumé du diff.
- Protocole exact.
- Risque principal.
- Ce qui pourrait invalider ton résultat.

Commence par lire `PROMPT.md`, `docs/D1_D2_DECISION.md` et `docs/FAMILY_SCHEMA.md`.
