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
