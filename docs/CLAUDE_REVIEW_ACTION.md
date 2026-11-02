# KHD — application de la revue Claude (03/11/2026)

## Verdict repris
Claude : **à reformuler**, 20/200 non faisable avec ≤30 M, from scratch, 20–40 k paires, budget 0 €.

## Mesure corrigée
### Benchmark
- **bench-155** était présent dans les données Gen8/Gen11 : fuite structurelle détectée par 13-grammes et signature de template.
- Benchmark officiel corrigé : **199 cas**.
- Échantillon de 30 cas extrait pour audit humain :
  `reports/ai-finalization/BENCH200_AUDIT_SAMPLE_20261103.jsonl`.

### Scores sur 199 cas
| Modèle | Correct | IC Wilson 95 % |
|---|---:|---:|
| Gen6 | 8/199 | **2,1–7,7 %** |
| Gen7 | 7/199 | 1,7–7,1 % |
| Gen8 | 5/199 | 1,1–5,7 % |

### Tests appariés
| Comparaison | Discordants | p McNemar |
|---|---|---:|
| Gen6 vs Gen7 | 4 / 5 | **0,739** |
| Gen6 vs Gen8 | 0 / 3 | **0,083** |

→ Aucune supériorité significative. « Gen6 meilleur » n’est pas prouvé.

## Diversité déjà testée
- Gen7 : 3 531 templates, 24 k paires.
- Gen8 : 3 177 templates d’entraînement, 354 templates de validation tenus à l’écart.
- Aucun gain significatif sur 199 cas.
- L’expérience « diversité seule » a donc été approchée, mais **pas avec le plafond de 1 % par famille** exigé par Claude.
- Un nouveau run identique ne serait pas justifié.

## Décision
- **Aucun nouveau run T4.**
- Aucune promotion.
- Le meilleur candidat reste Gen6, **8/199**, non significatif.
- Le serveur Render reste sur **Gen3d**.
- La limite « ≤30 M, from scratch, données actuelles, 0 € » est **reconnue**.
- Prochaine étape utile : audit humain du benchmark, puis benchmark plus grand ; pas un douzième entraînement aveugle.

## Coût
**0 €.**

## Git
- Branche : `deep/ai-finalization-20261101`
- Commit à suivre.
