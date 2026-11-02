# KHD — schéma canonique de famille (03/11/2026)

## Schéma intégré
- `schemas/family-pair.schema.json` : JSON Schema du couple famille/trace/réponse.
- `data/ai-finalization/family-schema-20261103/F087-000042.json` : exemple validé.
- Champs : `family_id`, `signature`, `depth`, `trace`, `answer`, `generator`.

## Validation
- 2 étapes de trace.
- `div(300,12000)=0,025` puis `mul(0,025,100)=2,5`.
- Réponse : **2,5 %**.
- `SCHEMA_OK F087-000042`.

## Décision de ne pas générer 20 000 paires
- D2 = **0/200** pour Gen6 et Gen7 sur des cas issus de leurs propres familles avec de nouveaux nombres.
- Le modèle n’exécute pas les procédures déjà vues.
- Générer 200 familles de plus ne débloque donc pas le raisonnement ; cela ajouterait du volume sans corriger D2.
- Aucune expérience T4 lancée.
- Aucun corpus entraîné.

## Condition pour générer
Le schéma est prêt. La génération complète ne sera justifiée que si :
1. un audit du benchmark est fait ;
2. une variante de raisonnement montre D2 > 50 % sur les familles vues ;
3. un protocole pré-enregistré fixe les seuils McNemar exacts.

## Coût
**0 €.**

## Git
- Branche : `deep/ai-finalization-20261101`
- Commit à suivre.
