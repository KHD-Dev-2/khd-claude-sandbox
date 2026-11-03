# PROPOSAL — option C : correction de l'évaluateur (`claude/reasoning-v1`)

Option choisie : **C**. Un bug reproductible de l'extracteur fausse les scores des modèles à tokenisation chiffre par chiffre (Gen6, Gen7, Gen11). Il fausse aussi D2, qui fonde la décision d'arrêt.
Le benchmark n'est pas modifié. L'extracteur v1 reste figé et reste le score principal, jusqu'à ce que le Lead adopte v2.
Aucun entraînement, aucun GPU, 0 €.

## 1. Hypothèse

Les scores 8/199 (Gen6), 7/199 (Gen7) et D2 = 0/200 mesurent en partie le **format de sortie**, pas le raisonnement. Trois défauts de `score()` v1, tous reproduits par `--selftest` :

1. **Nombres découpés.** Un modèle chiffre par chiffre décode 12,5 en `1 2 . 5` (décodeur WordLevel, tokens séparés par des espaces, vérifié avec `tokenizers`). `numbers()` y lit 1, 2 et 5, jamais 12,5. Une réponse décimale ou à plusieurs chiffres ne peut donc presque jamais être comptée juste.
2. **Réponses D2 en texte exact.** D2 note `kind: "text"` avec des attendus comme `'0,3723.'` comparés par sous-chaîne. Une sortie `0 , 3 7 2 3 .`, même parfaitement juste, ne contient jamais cette chaîne. **D2 = 0/200 était structurellement impossible à dépasser** pour Gen6 et Gen7.
3. **Faux positifs.** Un attendu texte `A` est « trouvé » dans `Médiane` (sous-chaîne). Un attendu numérique 2 est trouvé dans `V2 = 2,5 L` ou dans un opérande d'une formule, parce que les 4 derniers nombres sont tous acceptés.

## 2. Changement exact (`scripts/eval_gen11.py` uniquement)

- `score()` v1 est **inchangé**. Le champ `ok` et `correct` restent v1.
- Ajout de **v2**, qui n'est jamais utilisé seul :
  - `undigitize()` : inverse de `digitize()` sur le texte décodé (`1 2 . 5` → `12.5`, `0 , 0 2 5` → `0,025`). Il n'est appliqué qu'aux modèles chiffre par chiffre.
  - texte : correspondance par **tokens entiers**, plus par sous-chaîne ;
  - texte purement numérique (cas D2) : comparaison numérique **à la précision écrite** (½ unité du dernier chiffre), donc toujours exacte. La tolérance de 2 % n'y est pas appliquée : 133 ne valide pas 131 ;
  - `numbers_v2()` : un chiffre collé à une lettre (`V2`, `E_c`) n'est pas un nombre.
- **v2_strict** (diagnostic) : seulement le **dernier** nombre compte.
- Statistiques : `wilson()`, `mcnemar_exact()` (binomial bilatéral, pure Python).
- Deux modes CPU, sans torch : `--selftest` et `--rescore <result.json> <bench.jsonl> [<baseline.json>]`. Le rescoring relit les prédictions déjà sauvegardées, sans modèle ni GPU.
- Dans le run Kaggle : enregistre `ok_v2`, `ok_v2_strict`, `bench_sha256`, `extractor`. Signale `WARNING_DIGIT_MISMATCH` si la règle de prompt (`gen6/gen7/gen11` → `digitize`) contredit le vocabulaire du checkpoint. La règle de prompt elle-même n'est pas changée.

`train_gen11.py` et le schéma ne sont pas modifiés.

## 3. Mesure déjà faite (CPU, prédictions existantes, 0 €)

Rescoring local des fichiers de résultats existants, avec le benchmark 199 cas (200 moins `bench-155`, empreinte du fichier filtré `5138f781…`) et `d2.jsonl` (`063a409b…`).
**Contrôle** : v1 recalculé redonne exactement les `ok` enregistrés (0 écart sur tous les modèles).

| Modèle | v1 (figé) | v2 | v2_strict | IC Wilson 95 % v2 |
|---|---:|---:|---:|---|
| Gen6 | 8/199 | 7/199 | 3/199 | 1,7–7,1 % |
| Gen7 | 7/199 | **15/199** | **12/199** | 4,6–12,1 % |
| Gen8 | 5/199 | 4/199 | 1/199 | 0,8–5,1 % |
| Gen11 | 6/199 | 5/199 | 3/199 | 1,1–5,8 % |

McNemar exact contre Gen6 :

| Comparaison | v1 (b/c, p) | v2 (b/c, p) | v2_strict (b/c, p) |
|---|---|---|---|
| Gen7 vs Gen6 | 4/5, 1,000 | 12/4, **0,077** | 10/1, **0,012** |
| Gen8 vs Gen6 | 0/3, 0,250 | 3/6, 0,508 | 1/3, 0,625 |
| Gen11 vs Gen6 | 2/4, 0,688 | 4/6, 0,754 | 3/3, 1,000 |

D2 (200 cas, familles vues, nombres neufs) :

| Modèle | v1 (figé) | v2 | IC Wilson 95 % v2 |
|---|---:|---:|---|
| Gen6 | 0/200 | 5/200 | 1,1–5,7 % |
| Gen7 | 0/200 | **35/200** | 12,9–23,4 % |

Les 35 réussites D2 de Gen7 ont été relues une par une : la ligne `Réponse :` du modèle porte la valeur attendue.

**Lecture, sans revendiquer de gain :**
- « Gen6 meilleur candidat » n'est pas établi. 7 de ses 8 succès v1 disparaissent en v2 : ce sont des correspondances fortuites.
- « D2 = 0/200, le modèle n'exécute pas les procédures vues » est **faux tel qu'énoncé**. Gen7 en exécute environ 17,5 %. C'est toujours sous le seuil de 50 % : la conclusion « la diversité seule n'est pas la priorité » tient, mais la décision « arrêt définitif » reposait sur une mesure erronée.
- Selon les seuils imposés, Gen7 est **non concluant** dans les deux variantes : 15/199 avec p = 0,077 (v2), ou 12/199 avec p = 0,012 mais moins de 16 (v2_strict). Aucune promotion.

## 4. Données, architecture, hyperparamètres

Aucun changement : ni données d'entraînement, ni architecture, ni hyperparamètres. L'option C ne touche que la mesure.

## 5. Protocole (pré-enregistré)

1. Le Lead relit ce diff et lance `python scripts/eval_gen11.py --selftest` (CPU).
2. Il **fige v2 par empreinte** (sha256 de `eval_gen11.py`) **avant** tout nouveau run. Les règles ne changent plus ensuite.
3. Il relance `--rescore` sur ses propres fichiers (Gen6, Gen7, Gen8, Gen11, D2) pour confirmer les chiffres du §3 sur sa machine. Coût nul.
4. Audit humain des verdicts qui changent entre v1 et v2 : 18 cas pour Gen7, 13 pour Gen6 sur le benchmark. Un vrai positif doit avoir la valeur attendue dans la réponse finale du modèle.
5. Facultatif, CPU Kaggle suffisant : réévaluer Gen6 et Gen7 avec ce script (prédictions non tronquées, champ `bench_sha256`).
6. Tout entraînement futur sera jugé sur v2, avec v1 et v2_strict publiés à côté.

**Protocole T4** : aucun pour cette option. Un run T4 n'est justifié qu'après les étapes 1 à 4.

## 6. Seuils

Seuils appliqués à v2, contre Gen6 v2, McNemar exact bilatéral, sur les 199 cas :
- **Acceptation** : p < 0,05 et au moins 16/199 ;
- **Non concluant** : 11 à 15/199, ou p ≥ 0,05 ;
- **Arrêt** : au plus 10/199.

Pour v2 lui-même :
- **Adopté** si l'audit humain (étape 4) confirme au moins 90 % des verdicts modifiés ;
- **Rejeté** si plus de 10 % des nouveaux « justes » sont faux à la relecture. On revient alors à v1 et on corrige la règle fautive en v3, sans voir de nouveaux scores entre-temps.

## 7. Risque principal

**Degrés de liberté du chercheur.** Les règles v2 ont été écrites après avoir vu des exemples de prédictions, et ajustées une fois : la tolérance de 2 % appliquée aux réponses D2 acceptait 133 pour 131, ce que j'ai refusé. Chaque règle est générique et couverte par `--selftest`, mais seul le gel par empreinte (étape 2) et l'audit humain (étape 4) rendent la mesure indépendante.

## 8. Ce qui pourrait invalider ce résultat

- L'audit humain trouve des faux positifs v2 (par exemple un opérande égal à la réponse parmi les 4 derniers nombres). `v2_strict` borne ce cas par le bas.
- Les prédictions enregistrées sont tronquées à 300 caractères. Une réponse finale coupée échappe à v1 comme à v2, ce qui sous-estime les deux.
- Gen11 : si son corpus n'était pas chiffre par chiffre, le prompt `digitize` de l'évaluation est un décalage de distribution. Le nouveau `WARNING_DIGIT_MISMATCH` le signalera au prochain run, sans le corriger en silence.
- Le fichier 199 cas a été obtenu en retirant `bench-155` du fichier 200 cas. Si le benchmark officiel diffère, les chiffres du §3 changent. Comparer `bench_sha256`.
