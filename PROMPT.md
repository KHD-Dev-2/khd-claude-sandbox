# Prompt pour Claude dans le sandbox

Tu travailles uniquement dans `C:\KHD\claude-sandbox`.

## Contexte
KHD est un petit modèle neuronal entraîné from scratch. Derniers résultats vérifiés :
- Gen6 : 8/199 au benchmark, IC Wilson 2,1–7,7 %.
- Gen7 : 7/199, McNemar exact p = 1,000.
- Gen8 : 5/199, p exact = 0,250.
- D1 proxy : N_eff = 94, part maximale 9,75 %.
- D2 : 0/200 pour Gen6 et Gen7 sur des cas issus de leurs propres familles avec de nouveaux nombres.
- Décision : arrêt de la voie ≤30 M, from scratch, réponse seule, 0 €.

## Ta mission
Proposer **une seule** modification testable dans les scripts du sandbox :
- architecture de raisonnement explicitement entraînée ;
- objectif avec traces vérifiables ;
- ou correction de l’évaluateur, sans modifier le benchmark après avoir vu les scores.

## Contraintes
- Aucun API externe comme cerveau.
- Aucun GPU payant, aucun compte, aucun abonnement.
- Aucun benchmark dans l’entraînement.
- Aucun fichier hors du sandbox.
- Aucune suppression de checkpoint.
- Aucun score inventé.
- Modifier directement les fichiers du sandbox.
- Exécuter `python -m py_compile` sur les scripts modifiés.
- Produire `git diff` depuis le commit baseline.
- Donner : hypothèse, changement exact, protocole T4, seuil d’acceptation (McNemar exact), seuil d’arrêt.

## Livrable
1. Fichiers modifiés dans le sandbox.
2. `diff.patch` généré par `git diff`.
3. `PROPOSAL.md` avec protocole et seuils.
4. Aucune promesse de gain sans mesure.
