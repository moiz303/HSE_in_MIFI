"""Post-inference diagnostics and comparison; never imported by training/inference."""
import hashlib
import json
from pathlib import Path
import pandas as pd
from src.test_diagnostics import run as diagnose

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"independent": "Независимые измерения", "features": "Молекулярные признаки",
         "censoring": "Учёт цензуры AFT", "direct_si": "Прямой SI + смесь",
         "clean_pairs": "Согласованность пар", "head_params": "Параметры двух голов"}


def eligible(score_value, baseline, fraction=.10):
    return bool(score_value < baseline*(1-fraction))


def official_score_for(path, submissions):
    """Attach an official result only to the exact evaluated submission bytes."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    matches = [row for row in submissions if row.get("competition") == "molecular-activity-prediction-hack"
               and row.get("status") == "SubmissionStatus.COMPLETE" and row.get("submission_sha256") == digest]
    return float(matches[-1]["public_score"]) if matches else None


def main():
    config = json.loads((ROOT / "configs/hypotheses.json").read_text(encoding="utf-8"))
    reports = {}
    filenames = [ROOT / config["baseline_submission"]]
    for name in config["order"]:
        report = json.loads((ROOT / f"reports/hypotheses/{name}/report.json").read_text(encoding="utf-8"))
        path = ROOT / report["submission"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != report["submission_sha256"]:
            raise ValueError(f"Changed candidate submission: {name}")
        reports[name] = report
        filenames.append(path)
    # Only this post-inference stage reads the diagnostic test endpoint columns.
    diagnostic = diagnose(ROOT / "data/raw/kaggle/test.csv", filenames, ROOT / "reports/hypotheses_test_diagnostics")
    metrics = diagnostic["results"]
    baseline = metrics[0]["mse_ratio_subset"]
    rows = [{"hypothesis": "baseline", "name": "Текущий weighted Windows", "candidate": "weighted",
        "local_score": baseline, "improvement_percent": 0., "molecule_oof_mse": None,
        "scaffold_oof_mse": None, "oof_safe": True, "eligible_submission": False,
        "n_evaluated": metrics[0]["n_evaluated"], "n_test": metrics[0]["n_test"],
        "official_kaggle_score": None, "submission": config["baseline_submission"]}]
    for (name, report), values in zip(reports.items(), metrics[1:]):
        if (values["n_evaluated"], values["n_test"]) != (metrics[0]["n_evaluated"], metrics[0]["n_test"]):
            raise AssertionError("Inconsistent diagnostic coverage")
        selected = report["selection"]["selected"]
        rows.append({"hypothesis": name, "name": NAMES[name], "candidate": selected,
            "local_score": values["mse_ratio_subset"],
            "improvement_percent": 100*(baseline-values["mse_ratio_subset"])/baseline,
            "molecule_oof_mse": report["protocols"]["molecule"][selected]["mse"],
            "scaffold_oof_mse": report["protocols"]["scaffold"][selected]["mse"],
            "oof_safe": report["selection"]["oof_safe"],
            "eligible_submission": eligible(values["mse_ratio_subset"], baseline,
                config["submission_improvement_fraction"]),
            "n_evaluated": values["n_evaluated"], "n_test": values["n_test"],
            "official_kaggle_score": None, "submission": report["submission"]})
    official_path = ROOT / "reports/hypotheses/kaggle_submissions.json"
    official = json.loads(official_path.read_text(encoding="utf-8"))["submissions"] if official_path.exists() else []
    for row in rows:
        row["official_kaggle_score"] = official_score_for(ROOT / row["submission"], official)
    qualifying = sorted([row for row in rows if row["eligible_submission"]], key=lambda r: r["local_score"])
    result = {"baseline_score": baseline, "threshold_strictly_below": baseline*(1-config["submission_improvement_fraction"]),
        "metric": "Local endpoint-ratio subset MSE; smaller is better; not official Kaggle score",
        "test_endpoint_use": "Post-inference diagnosis and user-authorized conditional submission only; no fitting or hyperparameter selection",
        "previous_verified_kaggle_score": 1967765.75828,
        "best_eligible_submission": qualifying[0]["submission"] if qualifying else None,
        "eligible_submissions": [r["submission"] for r in qualifying], "results": rows,
        "verified_kaggle_submissions": official}
    directory = ROOT / "reports/hypotheses"
    (directory / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(rows).to_csv(directory / "comparison.csv", index=False)
    lines = ["# Сравнение гипотез", "", "Score — локальный MSE по доступным точным отношениям CC50/IC50, меньше лучше.",
        f"Покрытие: {metrics[0]['n_evaluated']}/{metrics[0]['n_test']}. Это не официальный Kaggle Score.",
        "Все гипотезы — независимые изменения одной базы; внутренние кандидаты выбраны по OOF до диагностики.", "",
        "| Гипотеза | Выбранный кандидат | Локальный Score | Улучшение | OOF-проверки | Kaggle Score |",
        "|---|---|---:|---:|---|---:|"]
    for row in rows:
        official_text = f"{row['official_kaggle_score']:.5f}" if row['official_kaggle_score'] is not None else "—"
        lines.append(f"| {row['name']} | {row['candidate']} | {row['local_score']:.2f} | {row['improvement_percent']:+.2f}% | {'пройдены' if row['oof_safe'] else 'не пройдены'} | {official_text} |")
    lines.extend(["", f"Условие пользователя для экспериментальной отправки: Score < {result['threshold_strictly_below']:.2f}.",
        "OOF-проверки показаны отдельно; экспериментальная заявка не меняет основной артефакт модели.",
        "Подтверждённый ранее Kaggle Score взвешенного варианта: 1 967 765.75828; он относится к прежнему submission.",
        "Kaggle Score привязан к SHA256 фактически отправленного файла; тире означает отсутствие подтверждённой оценки этого файла.",
        "Первоначальный порог не был достигнут. Отдельным последующим указанием пользователя отправлен head_params, заявка 58090521.", "",
        "Файлы моделей: `artifacts/hypotheses/`; отчёты, OOF, разбиения и submissions — в подкаталогах `reports/hypotheses/`."])
    (directory / "comparison.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(pd.DataFrame(rows)[["hypothesis", "candidate", "local_score", "improvement_percent", "oof_safe", "eligible_submission"]].to_string(index=False))
    print(f"[conditional submission] best eligible file={result['best_eligible_submission']}")


if __name__ == "__main__":
    main()
