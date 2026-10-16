# Statistical validation report

Profile: `publication_reference`. Values below come from `summary.csv`.
Each row identifies the lowest observed case coverage for that method;
Wilson bounds describe trial uncertainty for that case.

| Method | Lowest-coverage case | Covered / trials | Observed | 95% Wilson interval | Nominal |
| --- | --- | ---: | ---: | --- | ---: |
| canonical_qae | bernoulli_0.1 | 185 / 200 | 0.925 | [0.880, 0.954] | 0.950 |
| faster_ae | bernoulli_0 | 200 / 200 | 1.000 | [0.981, 1.000] | 0.950 |
| iqae_cost_aware | bernoulli_0.1 | 199 / 200 | 0.995 | [0.972, 0.999] | 0.950 |
| iqae_expected | bernoulli_0.25 | 197 / 200 | 0.985 | [0.957, 0.995] | 0.950 |
| iqae_fisher | bernoulli_0.75 | 196 / 200 | 0.980 | [0.950, 0.992] | 0.950 |
| iqae_fixed | bernoulli_0.25 | 199 / 200 | 0.995 | [0.972, 0.999] | 0.950 |
| mlqae | bernoulli_0.25 | 187 / 200 | 0.935 | [0.892, 0.962] | 0.950 |
