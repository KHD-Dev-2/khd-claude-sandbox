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

**Correction (contamination de D2).** 44 des 200 questions D2 figurent mot pour mot dans le corpus d'entraînement de Gen7. 19 des 35 réussites de Gen7 sont parmi elles. Sur les **156 cas D2 propres**, Gen7 obtient **16/156 (10,3 %)**. C'est ce chiffre qu'il faut retenir pour « exécution des procédures vues avec de nouveaux nombres ». Ces mêmes 44 questions sont aussi dans le corpus Gen12, alors que son manifeste indique `excluded_d2: 200`.

**Lecture, sans revendiquer de gain :**
- « Gen6 meilleur candidat » n'est pas établi. 7 de ses 8 succès v1 disparaissent en v2 : ce sont des correspondances fortuites.
- « D2 = 0/200, le modèle n'exécute pas les procédures vues » est **faux tel qu'énoncé**. Gen7 en exécute environ 10 % sur les cas D2 propres (16/156). C'est toujours sous le seuil de 50 % : la conclusion « la diversité seule n'est pas la priorité » tient, mais la décision « arrêt définitif » reposait sur une mesure erronée.
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
- Gen11 : confirmé au §9. Son vocabulaire contient 965 nombres à plusieurs chiffres, mais l'évaluation découpait ses prompts chiffre par chiffre.
- Le fichier 199 cas a été obtenu en retirant `bench-155` du fichier 200 cas. Si le benchmark officiel diffère, les chiffres du §3 changent. Comparer `bench_sha256`.

---

# Mise à jour du 03/11/2026 — contrat de prétraitement (branche `claude/preprocess-contract`)

## 9. Bug corrigé : le prétraitement des prompts était deviné d'après le nom du modèle

- L'évaluateur découpait les nombres du prompt chiffre par chiffre pour les modèles listés dans `DIGIT_MODELS`, sans regarder comment le modèle avait été entraîné.
- **Gen11 n'a pas été entraîné ainsi.** `train_gen11.py` n'appelle pas `digitize`, et le vocabulaire Gen11 contient **965 tokens numériques à plusieurs chiffres**. Son score de 6/199 a donc été mesuré sur des prompts écrits dans une forme jamais vue à l'entraînement. Ce score ne mesure pas Gen11.
- **Correctif** :
  - `train_gen11.py` applique un prétraitement déclaré, `PREPROCESS = {"digits": True, "sequences": True}`, au corpus d'entraînement, de validation et de holdout, et l'enregistre dans `config["preprocess"]` ;
  - `eval_gen11.py` lit ce contrat (`preprocess_for`). En l'absence de contrat, c'est le vocabulaire qui décide, jamais le nom du modèle. Un écart avec l'ancienne règle par nom est seulement signalé (`NOTE_NAME_RULE_DIFFERS`) ;
  - les séquences ADN/ARN sont découpées lettre par lettre (`seqsplit`), puis recollées avant la notation v2 (`unseq`). Sans cela, `CGGTCAAA` est un seul mot du vocabulaire, impossible à généraliser ;
  - `--rescore` détecte aussi les sorties ADN découpées (`predictions_seq_split`) ;
  - `train_gen11.py` tronquait à `ctx + 1` tokens alors que la table des positions en a `ctx` : c'est corrigé en `ctx`.
- **v1 est inchangé.** Avec ce code, les scores v2 de Gen6, Gen7 et Gen11 sont identiques au §3.
- **Correction du §3 pour Gen8** : ses sorties sont aussi découpées chiffre par chiffre (218 occurrences, aucun nombre collé). Son v2 réel est **2/199** (v2_strict 0/199), pas 4/199. L'ancien chiffre venait de la règle par nom, d'avant la détection automatique.

## 10. Résultat mesuré : Gen12b (fine-tune de Gen7, Kaggle T4, 0 €)

Gen12b est le run Gen12 de Deep avec ses défauts corrigés. Même départ (Gen7, 27,7 M paramètres) et même corpus de 47 809 traces, plus :
- **vocabulaire étendu** (2 310 → 2 368 tokens). Dans Gen12, 59 % des exemples contenaient `<unk>`, y compris dans les réponses ; dans Gen12b, il n'en reste aucun ;
- **ADN découpé** lettre par lettre ;
- **vraies étapes de calcul** : retenues de l'addition colonne par colonne, multiplication décomposée par position. Les erreurs de Gen7 sur D2 étaient surtout des erreurs de calcul faites avec la bonne procédure ;
- **rejeu du corpus Gen7** (24 001 traces) ;
- **anti-fuite** : 0 recouvrement avec les 199 cas (signature et 13-grammes), et les 44 questions D2 présentes dans les corpus Gen7/Gen12 retirées ;
- 8 000 étapes, batch 32, lr 2e-4 cosinus. Validation 3,70 → 0,095. 17 minutes de T4.

Les prédictions sont générées en greedy, sans outil, avec 96 tokens maximum. Gen7 a été réévalué dans le même kernel avec les mêmes réglages. Gen6 reprend ses prédictions enregistrées.

| Modèle | bench199 v1 | **bench199 v2** | bench199 v2_strict | D2 propre v2 (156 cas) |
|---|---:|---:|---:|---:|
| Gen6 | 8 | 7 | 3 | 3 |
| Gen7 | 7 | 15 | 12 | 16 |
| **Gen12b** | 11 | **24** | **19** | **78** |

IC de Wilson à 95 % : Gen12b bench199 v2 8,2–17,3 % ; D2 propre 42,3–57,8 %.

McNemar exact bilatéral :

| Comparaison | v2 (b/c, p) | v2_strict (b/c, p) | v1 figé (b/c, p) |
|---|---|---|---|
| Gen12b vs Gen6, bench199 | 18/1, **p = 0,0001** | 16/0, **p < 0,0001** | 6/3, p = 0,51 |
| Gen12b vs Gen7, bench199 | 12/3, p = 0,035 | 11/4, p = 0,12 | — |
| Gen12b vs Gen7, D2 propre | 64/2, p < 0,0001 | 64/2, p < 0,0001 | — |

**Application des seuils imposés** (v2, contre Gen6, McNemar exact) : **24/199 avec p = 0,0001 : critère d'acceptation atteint.** Avec v2_strict aussi : 19/199, p < 0,0001.

**Réserves, à lever avant toute promotion :**
1. **Audit à la main des 24 réussites** :
   - 1 faux positif certain : `bench-025`, attendu 34,5. La sortie contient l'opérande 35, accepté par la règle « 4 derniers nombres ». `v2_strict` l'exclut ;
   - 3 réussites sont acceptées par la tolérance officielle de 2 % mais pas exactes (écart de 1,4 à 1,9 %) ;
   - les autres sont exactes.
2. **Gain concentré** : 14 des 24 réussites viennent de deux familles d'épidémiologie (prévalence en %, incidence pour 1000), et 4 de la quantité de matière de l'eau. Ce n'est pas un gain général sur les 6 domaines.
3. **Avec v1 figé seul** : 11/199, p = 0,51 contre Gen6. L'acceptation dépend donc de l'adoption de v2. Les PR #1 et #3 sont fusionnées, mais l'audit humain des verdicts modifiés (§5, étape 4) n'est pas consigné.
4. Gen12b a eu 96 tokens de génération, Gen6 48 dans ses prédictions enregistrées. Gen6 s'arrête bien avant 48 tokens, mais cet écart doit être noté.

**Ce qui est démontré :** sur les familles déjà vues avec de nouveaux nombres (D2 propre), l'exécution des procédures passe de **10 % (Gen7) à 50 % (Gen12b)**. L'affirmation « un modèle de 27 M from scratch n'exécute pas les procédures vues » est réfutée.

## 11. Protocole suivant (pré-enregistré avant Gen13)

1. Le Lead audite les 24 réussites de Gen12b (le tableau par cas est dans le résultat Kaggle, non publié ici).
2. Il décide de promouvoir ou non Gen12b sur la base de **v2_strict** (19/199), qui écarte les correspondances fortuites.
3. Réévaluer Gen11 avec le contrat (prompts non découpés, comme à l'entraînement) avant de conclure sur l'architecture récurrente.
4. Seuils inchangés pour tout run suivant : acceptation à au moins 16/199 avec p < 0,05 contre le meilleur modèle accepté ; non concluant de 11 à 15 ; arrêt à 10 ou moins.
5. Aucune donnée, aucun poids et aucune prédiction par cas ne sont publiés dans ce dépôt.

## 12. Résultat mesuré : Gen13 (traces atomiques) et Gen11 sous contrat

**Gen13** reprend Gen12b (même architecture, 27,8 M) et le corpus Gen12b. Les familles qui échouaient sur D2 y sont réécrites en étapes atomiques, chacune vérifiable :
- moyennes : addition en colonnes avec retenues, puis division courte chiffre par chiffre ;
- travail W = F×d : produits chiffre par chiffre, puis somme des produits partiels ;
- énergie cinétique : v², puis m×v², puis ÷2.

Le générateur recalcule 129 243 étapes avant l'écriture. Les rapports de risque ne sont pas réécrits : leur division longue dépasse 256 tokens.

| Paramètre d'entraînement | Valeur |
|---|---|
| Steps | 8 000 |
| Batch | 32 |
| LR | 1,5e-4, cosinus |
| Durée | 3 558 s sur T4 |
| val loss | 0,036 (Gen12b : 0,095) |
| `<unk>` et lignes tronquées | 0 |

Évaluateur et cas inchangés (bench199, D2 propre = 156 cas). McNemar exact bilatéral.

| Ensemble | Gen6 | Gen7 | Gen12b | **Gen13** |
|---|---|---|---|---|
| bench199 v2 | 7 | 15 | 24 | **21** |
| bench199 v2_strict | 3 | 12 | 19 | **14** |
| D2 propre v2 | – | 16/156 | 78/156 | **103/156** |

Comparaisons appariées :

| Comparaison | Discordances (b/c) | p exact |
|---|---|---|
| Gen13 vs Gen6, bench, v2 | 17/3 | 0,0026 |
| Gen13 vs Gen6, bench, v2_strict | 13/2 | 0,0074 |
| Gen13 vs Gen12b, bench, v2 | 4/7 | 0,55 |
| Gen13 vs Gen12b, bench, v2_strict | 2/7 | 0,18 |
| Gen13 vs Gen12b, D2 propre | 27/2 | 1,6e-6 |

Détail D2 propre (Gen12b → Gen13) :
- moyennes : 3 → 15/29 ;
- travail : 5 → 17/19 ;
- énergie cinétique : 0 → 1/8 ;
- additions, ARN et prévalence : inchangées, à 100 % ;
- rapports de risque : toujours 0.

Lecture :
1. Les traces atomiques font ce qu'elles visent. Sur les procédures déjà vues avec de nouveaux nombres, le taux passe de 50 % à 66 % (p = 1,6e-6).
2. Elles ne transfèrent pas au benchmark. Gen13 passe le seuil contre Gen6 (21 ≥ 16, p < 0,05), mais pas contre Gen12b : 3 réussites de moins en v2, 5 en strict, différence non significative. Les cas perdus sont surtout en épidémiologie (14 → 12). Gen12b reste le meilleur candidat sur le benchmark.
3. L'énergie cinétique reste à 1/8 : la chaîne v² → m×v² → ÷2 est trop longue pour être tenue sans erreur.

**Gen11 réévalué sous contrat.** Le vocabulaire de Gen11 n'est pas découpé en chiffres, donc ses prompts ne sont plus digitalisés. Résultats :
- 5/199 en v2 et 3/199 en strict, identiques à l'ancienne évaluation ;
- p = 1,0 contre l'ancienne évaluation, p = 0,75 contre Gen6.

Le bug de prétraitement (§9) n'expliquait donc pas l'échec de Gen11 : l'architecture récurrente reste rejetée sur ces données.

## 13. Protocole suivant (pré-enregistré avant Gen14)

1. Aucune promotion automatique. Gen12b reste le candidat benchmark et attend l'audit du Lead (§11). Gen13 est un candidat D2 seulement.
2. Pour tout run suivant, la référence est Gen12b. Seuils inchangés : au moins 16/199 et p < 0,05 contre le meilleur modèle accepté, sur le même bench199 et le même évaluateur.
3. Avant tout nouveau run, examiner pourquoi l'épidémiologie régresse : un format de trace plus long peut déplacer la sortie des réponses courtes.
4. Aucune donnée, aucun poids et aucune prédiction par cas ne sont publiés dans ce dépôt.

## 14. Résultat mesuré : Gen14 (prévalence et incidence en étapes)

**Diagnostic avant le run.** En épidémiologie, Gen12b et Gen13 répondaient d'un seul coup (« Incidence = 23 pour 1000 »), sans étapes. Leurs réussites et leurs échecs s'inversaient donc d'un run à l'autre.

**Corpus.** Le corpus Gen13 est repris tel quel. Seules les réponses de prévalence et d'incidence (8 434) sont réécrites, en trois étapes :
1. cas × k ;
2. division courte par le premier chiffre de la population ;
3. décalage de la virgule.

Aucune question n'est modifiée. Six réponses du corpus d'origine étaient fausses (145/500 noté 28 % au lieu de 29 %) ; elles sont corrigées.

**Entraînement.** Reprise depuis Gen13 : 6 000 steps, LR 1e-4, 45 min sur T4. Val loss 0,033.

| Ensemble | Gen6 | Gen12b | Gen13 | **Gen14** |
|---|---|---|---|---|
| bench199 v2 | 7 | 24 | 21 | **17** |
| bench199 v2_strict | 3 | 19 | 14 | **15** |
| D2 propre (156) | – | 78 | 103 | **108** |

Comparaisons appariées (McNemar exact, discordances b/c) :

| Comparaison | v2 | v2_strict |
|---|---|---|
| Gen14 vs Gen6 | 12/2, p = 0,013 | 13/1, p = 0,002 |
| Gen14 vs Gen12b | 3/10, p = 0,09 | 2/6, p = 0,29 |
| Gen14 vs Gen12b, D2 | 32/2, p < 1e-6 | |

**Lecture.**
- Les traces de prévalence sont justes sur 7 des 8 cas de bench199.
- Les 7 cas d'incidence échouent tous. La cause est une erreur de copie : le modèle recopie « 1000 » en « 10000 », puis exécute correctement le reste de la trace. Dans le corpus, les populations d'incidence se limitaient à 500, 1000, 5000, 10000 et 20000, et un modèle de 27 M compte mal les suites de zéros.
- En épidémiologie, Gen14 passe à 9/30, contre 14 pour Gen12b.

**Réplication de Gen12b par Deep (Gen12c).** Deep a relancé Gen12b avec une autre graine : 21/200 en v2, 16 en strict, contre 24 et 19 pour Gen12b.
- Un écart de ±3 sur bench199 est donc du même ordre que la variance due à la graine.
- Les baisses de Gen13 et Gen14 face à Gen12b ne sont pas significatives, et le gain de Gen12b n'est pas non plus robuste.

## 15. Runs en cours (résultats à venir, mêmes seuils)

1. **Gen15** : reprise depuis Gen14 sur un corpus enrichi.
   - Familles « prépa » générées, chacune avec un exemple de trace : PGCD d'Euclide, puissances modulo, suites récurrentes, coefficients binomiaux, Horner, déterminants 3×3, Cramer, sommes de carrés, facteurs premiers.
   - Prévalence et incidence avec des populations variées.
   - 80 372 étapes recalculées par le générateur.
   - Le filtre anti-fuite a retiré 1 879 questions qui reprenaient un gabarit du benchmark.
   - Nouveau jeu tenu à part, **D3** (220 questions, autre graine, notées sur la réponse finale exacte). D3 mesure l'exécution des procédures sur de nouveaux nombres. Il ne dit rien sur des sujets de concours réels : aucun texte de sujet n'est utilisé, leurs droits n'étant pas établis.
   - Réserve : le correctif d'incidence vient de l'analyse des échecs sur bench199. Un gain en épidémiologie sur bench199 sera donc signalé comme tel.
2. **Évaluation de l'adaptateur Mistral-7B de Deep** (`khd-mistral-qlora-20261103`, base Mistral-7B-v0.1, Apache 2.0). Mesure sur bench199 et D2 propre en décodage glouton. Le témoin est la même base sans adaptateur, pour séparer ce qui vient du modèle de base de ce qui vient de l'entraînement KHD.
3. **Mistral-étapes** : le pipeline QLoRA de Deep entraîné sur le corpus Gen15, avec trois corrections :
   - la perte ne porte que sur la réponse ;
   - un token de fin est appris (le collateur d'origine le masquait, car pad = eos) ;
   - les prédictions bench199, D2 et D3 sont produites dans le même kernel.

Pour chaque run, la décision suit les mêmes seuils :
- **Référence** : Gen12b, à comparer avec la variance de graine mesurée par Deep.
- **Acceptation** : au moins 16/199 et p < 0,05 contre le meilleur modèle accepté.
- **Promotion** : aucune avant l'audit du Lead.

Inférence locale (poids ouverts), sans API externe, 0 €.

## 16. Résultat mesuré : Gen15 (familles prépa + populations variées)

Reprise depuis Gen14 : 6 000 steps, 44 min sur T4. Val loss 0,059, plus haute qu'avant car le corpus contient de nouvelles familles.

| Ensemble | Gen6 | Gen12b | Gen14 | **Gen15** |
|---|---|---|---|---|
| bench199 v2 | 7 | 24 | 17 | **27** |
| bench199 v2_strict | 3 | 19 | 15 | **18** |
| D2 propre (156) | – | 78 | 108 | **107** |
| D3 prépa (220, réponse exacte) | – | – | – | **41** |

Comparaisons appariées (McNemar exact, discordances b/c) :

| Comparaison | v2 | v2_strict |
|---|---|---|
| Gen15 vs Gen6 | 21/1, p < 1e-4 | 15/0, p = 1e-4 |
| Gen15 vs Gen12b | 10/7, p = 0,63 | 6/7, p = 1,0 |
| Gen15 vs Gen14 | 15/5, p = 0,041 | 8/5, p = 0,58 |
| Gen15 vs Gen12b, D2 propre | 31/2, p < 1e-6 | |

Par domaine, en v2 :
- épidémiologie 13/30 (Gen14 : 9) ;
- chimie 7/25 ;
- physique 5/34 ;
- biologie 1/41 ;
- mathématiques 1/35 ;
- informatique 0/34.

D3 par famille :

| Famille | Score |
|---|---|
| Prévalence | 12/20 |
| Incidence | 10/20 |
| Horner | 10/20 |
| Euclide | 3/20 |
| Facteurs premiers | 2/20 |
| Puissances modulo | 2/20 |
| Déterminants 3×3 | 1/20 |
| Suites récurrentes | 1/20 |
| Cramer | 0/20 |
| Binomiaux | 0/20 |
| Sommes de carrés | 0/20 |

Lecture :
1. Gen15 est le meilleur modèle maison en v2 sur bench199 (27/199). Il passe le seuil contre Gen6. Il ne bat pas Gen12b de façon significative (p = 0,63), et en strict il est au même niveau (18 contre 19). Avec la variance de graine mesurée par Deep (±3), il n'y a pas de gain démontré contre Gen12b.
2. Le correctif des populations remonte l'épidémiologie de 9 à 13. Ce correctif vient de l'analyse des échecs sur bench199 (§15) : ce gain ne compte pas comme une preuve indépendante.
3. Sur D3, le modèle apprend les traces courtes (Horner, ratios), pas encore les longues (Cramer, Euclide, déterminants). Deux familles ont trop peu d'exemples distincts : 97 pour les binomiaux, 92 pour les sommes de carrés. 6 000 steps ne suffisent pas pour 9 familles nouvelles.
4. Aucune promotion. Pour qu'un candidat maison soit défendable, il faut d'abord une réplication à une autre graine de Gen12b ou de Gen15.

## 17. Résultat mesuré : adaptateur Mistral-7B de Deep (`khd-mistral-qlora-20261103`)

**Modèle.** Mistral-7B-v0.1 (Apache 2.0), QLoRA rang 16 entraîné par Deep : 600 steps sur son corpus de raisonnement Gen11 (43 518 exemples).

**Mesure.** Inférence locale 4 bits sur T4, sans API, en décodage glouton, avec le format de prompt de Deep. Même évaluateur v2, mêmes 199 cas, même D2 propre.

L'adaptateur ne sait pas s'arrêter et répète ses blocs « ### Réponse ». Sa prédiction est donc coupée au premier bloc. Le témoin (la même base sans adaptateur) est mesuré de deux façons, parce que ses réponses commencent par un saut de ligne :
- coupé au premier paragraphe, comme l'adaptateur ;
- coupé au premier « ### », ce qui lui est plus favorable.

| Modèle | bench199 v2 | bench199 v2_strict | D2 propre |
|---|---|---|---|
| Gen6 | 7 | 3 | – |
| Gen12b (meilleur maison accepté) | 24 | 19 | 78 |
| Gen15 | 27 | 18 | 107 |
| Mistral-7B sans adaptateur (coupe « ### ») | 72 | 66 | 18 |
| **Mistral-7B + LoRA KHD (Deep)** | **113** | **103** | **94** |

Comparaisons appariées de l'adaptateur (McNemar exact, discordances b/c) :

| Contre | v2 | v2_strict |
|---|---|---|
| Gen6 | 106/0, p < 1e-6 | 100/0, p < 1e-6 |
| Gen12b | 89/0, p < 1e-6 | 87/3, p < 1e-6 |
| Base Mistral, bench199 | 52/11, p < 1e-6 | 51/14, p = 4e-6 |
| Base Mistral, D2 propre | 76/0, p < 1e-6 | |
| Gen12b, D2 propre | 30/14, p = 0,023 | |

**Recherche de fuite.**
- Aucun gabarit de question de bench199 (nombres masqués) n'apparaît dans le corpus de Deep.
- 56 questions de bench199 partagent leur suite de nombres avec une question d'entraînement, mais ce sont surtout des coïncidences entre familles différentes : « moyenne de 39 et 7 » face à « 39 kg à 7 m/s ».
- Les corpus de Gen12b et Gen15 en contiennent autant (58 et 60), sans produire un tel score.
- Le corpus contient les 44 cas contaminés de D2, déjà exclus de D2 propre.

Par domaine (v2) :
- physique 28/34 ;
- chimie 20/25 ;
- épidémiologie 22/30 ;
- mathématiques 23/35 ;
- biologie 15/41 ;
- informatique 5/34.

**Lecture.**
1. Le seuil d'acceptation est franchi de très loin : 113/199 avec p < 1e-6 contre Gen6 et contre Gen12b. Mistral-7B seul en donne déjà une bonne part ; l'entraînement KHD ajoute environ 40 cas (p < 1e-5).
2. C'est la voie la plus rapide vers une IA KHD : poids ouverts, adaptateur à nous, inférence locale.
3. Avant toute promotion, il reste :
   - l'audit du Lead ;
   - une réplication avec une autre graine ;
   - corriger l'arrêt de génération (token de fin, objet de Mistral-étapes, §15) ;
   - régler la question de l'hébergement : un 7B demande un GPU ou une quantification CPU. L'adaptateur LoRA Cloudflare existant repose sur Mistral-7B-Instruct-v0.2, qui n'est pas la même base.

## 18. Résultat mesuré : Mistral-étapes (`khd-mistral-steps-20261103`)

**Modèle.** Même base et même LoRA que Deep (rang 16, q/k/v/o). 1 000 steps sur le corpus Gen15 (83 177 exemples, environ 0,1 époque), graine 1503, 1 h 58 sur T4. Perte finale 0,03.

Corrections par rapport au script de Deep :
- perte sur la réponse seule ;
- token de fin appris ;
- prédictions faites dans le même kernel.

**Arrêt.** Le modèle s'arrête seul : 112 caractères en moyenne. Les prédictions sont notées **sans aucune coupe**.

| Modèle | bench199 v2 | v2_strict | D2 propre | D3 prépa |
|---|---|---|---|---|
| Gen12b | 24 | 19 | 78 | – |
| Gen15 | 27 | 18 | 107 | 41/220 |
| Mistral + LoRA Deep (coupé) | 113 | 103 | 94 | – |
| **Mistral-étapes** | **111** | **105** | **105** | **162/220** |

Comparaisons appariées (McNemar exact, b/c) :

| Comparaison | v2 | v2_strict |
|---|---|---|
| Mistral-étapes vs Gen6 | 104/0, p < 1e-6 | 102/0, p < 1e-6 |
| Mistral-étapes vs Gen12b | 92/5, p < 1e-6 | 90/4, p < 1e-6 |
| Mistral-étapes vs LoRA Deep, bench199 | 17/19, p = 0,87 | 17/15, p = 0,86 |
| Mistral-étapes vs LoRA Deep, D2 propre | 15/4, p = 0,019 | |
| Mistral-étapes vs Gen12b, D2 propre | 36/9, p = 1e-4 | |

Par domaine (v2), avec entre parenthèses le score de l'adaptateur de Deep :
- physique 28/34 (28) ;
- chimie 22/25 (20) ;
- biologie 23/41 (15) ;
- mathématiques 19/35 (23) ;
- épidémiologie 18/30 (22) ;
- informatique 1/34 (5).

D3 par famille :

| Famille | Score |
|---|---|
| Horner | 20/20 |
| Prévalence | 20/20 |
| Incidence | 20/20 |
| Suites récurrentes | 20/20 |
| Binomiaux | 18/20 |
| Cramer | 17/20 |
| Puissances modulo | 14/20 |
| Facteurs premiers | 12/20 |
| Déterminants 3×3 | 10/20 |
| Euclide | 9/20 |
| Sommes de carrés | 2/20 |

**Lecture.**
1. Sur bench199, Mistral-étapes est au niveau de l'adaptateur de Deep (p = 0,87), mais il répond proprement, sans troncature.
2. Il est meilleur sur D2 propre (p = 0,019) et sur D3 : 162/220, contre 41 pour Gen15 qui a été entraîné sur le même corpus. Les traces vérifiées s'apprennent bien mieux sur une base 7B.
3. Le corpus est celui de Gen15, déjà filtré : aucun gabarit de bench199. D3 a été généré avec une autre graine.
4. Point faible commun aux deux adaptateurs : l'informatique (1 et 5 sur 34). Il n'existe pas encore de famille de traces pour ce domaine.
5. Aucune promotion. Il reste :
   - l'audit du Lead ;
   - une seconde graine ;
   - l'hébergement.

## 19. Protocole suivant (pré-enregistré avant les résultats) : Mistral-étapes graine 2 + informatique

**Run.** `khd-mistral-steps2-20261103`, même pipeline que §18, graine 1607, 1 000 steps, sur le corpus **Gen16**. Gen16 reprend Gen15 et ajoute 11 familles d'informatique générées :
- conversions binaire, décimal et hexadécimal ;
- opérations bit à bit ;
- complément à deux ;
- bits nécessaires ;
- recherche dichotomique ;
- boucles imbriquées ;
- taille mémoire ;
- notation polonaise inverse ;
- mise à l'échelle d'une complexité.

13 834 nouveaux exemples, 51 324 étapes recalculées. Chaque réponse est aussi vérifiée contre une fonction Python de référence. Le filtre anti-fuite n'a retiré aucune question.

**Réserve.** Le domaine a été choisi parce que les deux adaptateurs y font 1 à 5 sur 34 dans bench199. Ce choix est fait au niveau du domaine, sans regarder les questions une à une, mais un gain en informatique sur bench199 ne compte pas comme une preuve indépendante. La preuve indépendante est **D3cs** : 220 questions tenues à part, générées avec une autre graine et exclues de l'entraînement.

**Confusion assumée.** La graine et le corpus changent en même temps. La réplication de §18 se lit donc sur ce qui n'a pas changé : bench199 hors informatique, D2 propre et D3.

**Seuils.**
- **Réplication** : bench199 hors informatique à ±10 cas de Mistral-étapes (graine 1503), et aucune différence appariée significative sur D2 propre ni sur D3.
- **Informatique** : D3cs ≥ 110/220. Le gain en informatique sur bench199 est rapporté à titre indicatif.
- **Promotion** : aucune avant l'audit du Lead.

## 20. Protocole suivant (pré-enregistré avant les résultats) : Mistral-étapes 3, prépa v2

**Diagnostic sur D3** (Mistral-étapes, graine 1503). La procédure est juste, mais un calcul intermédiaire est faux :
- un reste de division (`350=4×70+100`) ;
- une somme de trois termes (`100+42+150=392`) ;
- un produit de deux nombres à plusieurs chiffres (`702×53`) ;
- un exposant mal compté (`2^4` au lieu de `2^5`).

**Corpus Gen17** (`khd-gen17-prepa-v2-20261103`) :
- 24 000 lignes de Gen14 ;
- les familles de Gen15 et celles d'informatique de Gen16 ;
- **prépa v2** : Euclide, puissances modulo, déterminants, sommes de carrés et facteurs premiers, avec des étapes atomiques. Chaque division est vérifiée (`q×b=P ; a-P=r ; r<b`), les termes s'additionnent deux à deux, les produits sont découpés chiffre par chiffre et les facteurs comptés explicitement ;
- **7 nouvelles familles** : Bézout (Euclide étendu), inverse modulaire, suites d'ordre 2, sommes géométriques, produits de matrices 2×2, restes chinois, sommes arithmétiques.

33 799 nouveaux exemples, 279 151 étapes recalculées, 1 780 questions retirées par le filtre anti-fuite. Aucun texte de sujet de concours n'est utilisé.

**Run.** `khd-mistral-steps3-20261103` : même pipeline, graine 1709, 1 500 steps, jusqu'à 400 tokens générés.

**Réserve.** Les corrections viennent de l'analyse des échecs sur D3 : D3 n'est donc plus indépendant pour ces familles. La mesure indépendante est **D4** : 240 questions sur les 12 familles v2 et nouvelles, autre graine, tenues à part. Les questions de D3, D3cs et D4 sont exclues de l'entraînement.

**Seuils.**
- **D4** : au moins 168/240 (70 %).
- **Non-régression** : bench199 et D2 propre sans perte significative (p < 0,05) contre Mistral-étapes (§18).
- **D3** : rapporté à titre indicatif.
- **Promotion** : aucune avant l'audit du Lead.

## 21. Résultat mesuré : Mistral-étapes graine 2 + informatique (§19)

Run `khd-mistral-steps2-20261103` : 1 000 steps, 1 h 53 sur T4, perte finale 0,029. Prédictions notées sans coupe.

| Ensemble | Graine 1 (Gen15) | **Graine 2 (Gen16)** | Comparaison appariée b/c |
|---|---|---|---|
| bench199 v2 | 111 | **126** | 23/8, p = 0,011 |
| bench199 v2_strict | 105 | **115** | 19/9, p = 0,087 |
| bench199 hors informatique v2 | 110/165 | **123/165** | 20/7, p = 0,019 |
| bench199 informatique v2 | 1/34 | 3/34 | 3/1, p = 0,63 |
| D2 propre | 105 | 102 | 4/7, p = 0,55 |
| D3 prépa | 162 | 165 | 16/13, p = 0,71 |
| **D3cs informatique** | – | **144/220** | |

Par domaine (v2, graine 2) :
- physique 29/34 ;
- chimie 20/25 ;
- biologie 28/41 ;
- mathématiques 24/35 ;
- épidémiologie 22/30 ;
- informatique 3/34.

D3cs par famille :

| Famille | Score |
|---|---|
| Opérations bit à bit | 20/20 |
| Recherche dichotomique | 20/20 |
| Notation polonaise inverse | 20/20 |
| Bits nécessaires | 18/20 |
| Hexadécimal | 16/20 |
| Binaire → décimal | 15/20 |
| Complexité | 15/20 |
| Boucles imbriquées | 7/20 |
| Taille mémoire | 7/20 |
| Décimal → binaire | 4/20 |
| Complément à deux | 2/20 |

**Décision selon les seuils pré-enregistrés.**
1. **Réplication : non, au sens strict.** Hors informatique, l'écart sur bench199 est de +13 cas, au-delà du ±10 prévu, et il est significatif (p = 0,019), mais en faveur de la graine 2. D2 propre et D3 ne bougent pas de façon significative. La graine, le corpus ou les deux déplacent donc bench199 d'environ 13 cas. Le score de Mistral-étapes est à lire comme 111 à 126, pas comme un point fixe.
2. **Informatique : seuil atteint sur D3cs** (144/220, contre 110 requis). Aucun transfert significatif vers l'informatique de bench199 (3/34) : les questions de bench199 ne recoupent apparemment pas ces familles. Je ne les ai pas regardées, pour ne pas ajuster sur le test.
3. **Échecs D3cs**, même schéma que D3 : la méthode est juste, le calcul atomique est faux.
   - Restes lus à l'envers : 236 → `11101010`.
   - Produits à plusieurs chiffres : `141×32=4496`.
   - Formule des paires i<j remplacée par n×(n-1).
   - Décomposition en puissances de 2 fausse dans le complément à deux.

   Le corpus Gen17 (§20) était déjà figé et pré-enregistré : il n'est pas modifié. Ces corrections iront dans une version suivante.
4. Aucune promotion.

## 22. Protocole suivant (pré-enregistré avant les résultats) : Mistral-étapes 4, informatique v2

Écrit pendant que Mistral-étapes 3 (§20) tourne, **avant d'en connaître les scores**.

**Diagnostic sur D3cs** (graine 2, §21) : la méthode est juste, le calcul atomique est faux. Restes lus à l'envers, produits à plusieurs chiffres, formule des paires, décomposition en puissances de 2.

**Corpus Gen18** (`khd-gen18-informatique-v2-20261103`) : la recette de Gen17, avec quatre familles d'informatique réécrites (v2) :
- **décimal → binaire** : chaque division par 2 est vérifiée (`37÷2=18 r1 (18×2+1=37)`), puis les restes sont listés dans l'ordre obtenu et retournés explicitement (« lus à l'envers, dernier reste en premier ») ;
- **complément à deux** : décomposition par soustractions successives (`97-64=33 ; 33-32=1 ; 1-1=0`), puis les 8 bits écrits position par position ;
- **boucles imbriquées** : produits découpés chiffre par chiffre ; pour les paires i<j, la formule n×(n-1)÷2 est rappelée avec « chaque paire comptée une seule fois » et la division par 2 posée ;
- **taille mémoire** : produit découpé, puis division par 8 posée.

33 779 nouveaux exemples, 326 674 étapes recalculées, 1 780 questions retirées par le filtre anti-fuite. Aucun texte de sujet de concours.

**Run.** `khd-mistral-steps4-20261103` : même pipeline, graine 1811, 1 500 steps. Lancé seulement après le verdict de §20, pour ne pas dépasser deux sessions GPU Kaggle.

**Réserve.** Les corrections viennent des échecs sur D3cs : D3cs n'est plus indépendant. La mesure indépendante est **D5cs** : 220 questions sur les 11 familles d'informatique, nouvelle graine, tenues à part. Les questions de D3, D3cs, D4 et D5cs sont exclues de l'entraînement.

**Seuils.**
- **D5cs** : au moins 176/220 (80 %), et au moins 14/20 sur chacune des quatre familles réécrites.
- **Non-régression** : bench199 et D2 propre sans perte significative (p < 0,05) contre Mistral-étapes (§18) ; D4 sans perte significative contre Mistral-étapes 3 (§20).
- **Promotion** : aucune avant l'audit du Lead.

## 23. Résultat mesuré : Mistral-étapes 3, prépa v2 (§20)

Run `khd-mistral-steps3-20261103` : graine 1709, 1 500 steps sur Gen17 (57 799 exemples), 3 h 35 sur T4, loss finale 0,013.

| Modèle | bench199 v2 | bench199 strict | D2 propre | D3 | D3cs | D4 |
|---|---|---|---|---|---|---|
| Mistral-étapes graine 1 (§18) | 111 | 105 | 105 | 162 | – | – |
| Mistral-étapes graine 2 (§21) | 126 | 115 | 102 | 165 | 144 | – |
| **Mistral-étapes 3** | **105** | **97** | **100** | **182** | **165** | **191/240** |

Tests appariés (McNemar exact), Mistral-étapes 3 contre :
- graine 1 : bench199 b/c 20/26, p = 0,46 ; D2 4/9, p = 0,27 ; D3 32/12, p = 0,004 ;
- graine 2 : bench199 6/27, **p = 0,0003** ; D2 5/7, p = 0,77 ; D3 30/13, p = 0,014 ; D3cs 28/7, p = 0,0005.

D4 par famille : restes chinois 20/20, produits de matrices 20/20, suites d'ordre 2 20/20, inverse modulaire 19/20, sommes géométriques 18/20, déterminants 17/20, facteurs premiers 16/20, puissances modulo 16/20, Euclide 15/20, sommes arithmétiques 14/20, Bézout 11/20, sommes de carrés 5/20.

**Décision selon les seuils pré-enregistrés.**
1. **D4 : seuil atteint** (191/240 = 80 %, contre 168 requis).
2. **Non-régression contre §18 : tenue** (bench199 p = 0,46, D2 p = 0,27).
3. **Réserve importante, hors seuils** : contre la graine 2, bench199 baisse nettement (126 → 105, p = 0,0003). Les trois runs donnent 105, 111 et 126 : bench199 varie de ±10 cas selon la graine et le corpus, et la prépa ne transfère pas vers bench199. Le gain est réel sur les exercices générés (D3 +20, D3cs +21), pas sur le benchmark métier.
4. Points faibles restants : sommes de carrés (5/20, produit à trois facteurs), Bézout (11/20), et en informatique décimal → binaire 5/20, complément à deux 7/20, taille mémoire 8/20. Les trois derniers sont visés par Gen18 (§22), pré-enregistré avant ces résultats.
5. Aucune promotion. Mistral-étapes 4 (§22) est lancé.

## 24. Protocole suivant (pré-enregistré avant les résultats) : Mistral-étapes 5, sciences

Demande du propriétaire : entraîner l'IA sur les maths, la physique, la chimie et la biologie. Écrit avant tout résultat de Mistral-étapes 4 et 5.

**Corpus Gen19** (`khd-gen19-sciences-20261103`) : 30 000 lignes de Gen18 et **55 familles d'exercices générés**, du lycée à la prépa.
- **Maths (18)** : équations du premier degré, racines du second degré, dérivées, intégrales de polynômes, probabilités (dés, urnes), espérance, variance, Pythagore, volumes, complexes (produit, module), produit scalaire, équations exponentielles, pourcentages successifs, arrangements, composition de fonctions, moyennes pondérées.
- **Physique (14)** : mouvement uniformément accéléré, chute libre, conservation de l'énergie, deuxième loi de Newton avec frottements, résistances en série et en parallèle, énergie électrique et coût, ondes, pression hydrostatique, chaleur massique, lentilles minces, chocs mous, décroissance radioactive, conversions de vitesse.
- **Chimie (9)** : masses molaires, quantités de matière, stœchiométrie, réactif limitant, volume molaire, concentrations, pH, pourcentage massique, dosage.
- **Biologie (12)** : croisements à un gène, croisements à deux gènes, Hardy-Weinberg, traduction avec le code génétique réel, complémentarité et transcription, taux de GC, croissance bactérienne, mitose et méiose, rendement écologique, débit cardiaque, dénombrement de colonies, IMC.
- **Épidémiologie (2)** : sensibilité, spécificité et VPP, odds ratio.

25 476 nouveaux exemples, 103 410 égalités recalculées (exactes en fractions, ou à l'arrondi près pour « ≈ »), aucune question retirée par le filtre anti-fuite (bench199 et D2). Aucun texte de manuel ni de sujet d'examen. Certaines familles sont petites parce que leur espace de questions est limité (pH 22, dés 14, chute libre 8).

**Run.** `khd-mistral-steps5-20261103` : même pipeline, graine 1913, **2 000 steps**.

**Mesure indépendante.** **D6** : 440 questions, 8 par famille, nouvelle graine, tenues à part (avec D3, D3cs, D4 et D5cs).

**Seuils.**
- **D6** : au moins 308/440 (70 %), et au moins 60 % dans chacun des cinq domaines.
- **Non-régression** : bench199 et D2 propre sans perte significative (p < 0,05) contre Mistral-étapes (§18) ; D4 sans perte significative contre Mistral-étapes 3 (§20).
- **Indicatif** : bench199 par domaine (biologie, chimie, physique, maths). Le choix des domaines vient d'une demande du propriétaire, pas des questions de bench199, que je n'ai pas regardées.
- **Promotion** : aucune avant l'audit du Lead.

## 25. Protocole suivant (pré-enregistré avant les résultats) : Mistral-étapes 6, niveau avancé

Demande du propriétaire : les maths et les sciences les plus difficiles. Écrit avant tout résultat de Mistral-étapes 4, 5 et 6.

**Corpus Gen20** (`khd-gen20-avance-20261103`) : 15 000 lignes de Gen18, les 25 476 lignes de sciences de Gen19, et **43 familles de niveau prépa / licence** :
- **Maths (17)** : valeurs propres 2×2, inverse et puissances de matrices, équations différentielles linéaires d'ordre 1 et 2 avec conditions initiales, intégration par parties, séries géométriques et Σ n x^n, limites par équivalents, racines entières d'un polynôme de degré 3 (Horner), indicatrice d'Euler, petit théorème de Fermat, loi binomiale, formule de Bayes, dérangements, nombres de Catalan, puissances de complexes, somme des cubes.
- **Physique (10)** : relativité restreinte (dilatation du temps, contraction des longueurs), modèle de Bohr, énergie des photons, oscillateur élastique, circuit RC, rendement de Carnot, gaz parfait, tir horizontal, troisième loi de Kepler, loi de Coulomb.
- **Chimie (7)** : équilibre H2 + I2 ⇌ 2 HI, cinétique d'ordre 1, loi de Hess (ΔfH° de référence), solutions tampons, acides faibles, piles et ΔrG°, produit de solubilité.
- **Biologie (7)** : liaison génétique (distance en cM), test du χ², Michaelis-Menten, PCR, croissance logistique, hérédité liée à l'X, potentiel d'équilibre membranaire.
- **Épidémiologie (2)** : seuil d'immunité collective, nombre de sujets à traiter.

20 275 nouveaux exemples, 94 272 égalités recalculées, aucune fuite détectée. Certaines familles sont très petites parce que leur espace de questions est limité (Hess 2, Bohr 4, immunité collective 4) : elles sont surtout testées sur la méthode.

**Run.** `khd-mistral-steps6-20261103` : même pipeline, graine 2017, **2 500 steps**. Il passe après Mistral-étapes 5, dans la limite de deux sessions GPU Kaggle.

**Mesure indépendante.** **D7** : 344 questions, 8 par famille, nouvelle graine, tenues à part (avec D3 à D6).

**Seuils.**
- **D7** : au moins 206/344 (60 %). Le niveau est plus dur, d'où un seuil plus bas que pour D6.
- **Non-régression** : D6 sans perte significative contre Mistral-étapes 5 (§24) ; bench199 et D2 propre sans perte significative contre Mistral-étapes (§18).
- **Indicatif** : D7 par domaine et bench199 par domaine.
- **Promotion** : aucune avant l'audit du Lead.
