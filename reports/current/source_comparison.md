# Paired comparison of training sources

Values are MSE. OOF averages molecule/scaffold on train_v2. Endpoint diagnostics cover 468/597 test rows and are not the official Kaggle Score. Lower is better.

| Method | OOF Kaggle | OOF public + Kaggle | Diagnostic Kaggle | Diagnostic public + Kaggle |
|---|---:|---:|---:|---:|
| chemeleon_ridge | 7,693,582.04 | 56,006,693.70 | 2,276,487.37 | 25,336,601.51 |
| chemeleon_mlp | 9,191,511.41 | 62,869,404.78 | 3,771,186.93 | 46,765,171.29 |
| chemeleon_domain | 7,973,414.61 | 7,807,105.80 | 3,708,113.40 | 3,612,776.64 |
| catboost_direct | 5,501,602.81 | 103,831,255.14 | 4,187,780.79 | 9,927,306.08 |
| chemeleon_mlp_log1p | 5,071,066.32 | 8,404,670.38 | 2,228,047.25 | 5,650,984.88 |
| chemeleon_domain_log1p | 5,076,402.90 | 5,061,848.48 | 3,189,087.84 | 3,240,409.48 |
| catboost_log1p | 5,125,011.44 | 5,135,017.31 | 2,001,905.61 | 1,968,835.32 |

Initial comparison sent no submissions. In the separately authorized goal, domain log1p + public scored 1586333.78789 (ref 58103006). Other base methods remain unsubmitted. Historical concentration heads are outside the current protocol.
