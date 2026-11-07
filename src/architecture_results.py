"""Post-inference score table and one-candidate threshold decision; no fitting."""
import hashlib
import json
from pathlib import Path
import pandas as pd
from src.hypothesis_results import official_score_for
from src.test_diagnostics import run as diagnose

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"chemeleon": "CheMeleon (frozen 2D)", "joint_gp": "Совместный GP/Tanimoto",
         "unimol": "Uni-Mol (frozen 3D)", "domain_heads": "Доменные головы CheMeleon"}


def submission_decision(local_score, digest, submissions, threshold):
    submitted = any(row.get("competition") == "molecular-activity-prediction-hack"
                    and row.get("submission_sha256") == digest and row.get("ref") for row in submissions)
    passes = bool(local_score < threshold)
    return {"passes_local_threshold": passes, "submitted": bool(submitted), "eligible": passes and not submitted}


def main():
    config = json.loads((ROOT / "configs/architectures.json").read_text(encoding="utf-8"))
    baselines = [("weighted", "Взвешенный XGBoost, отправленный ранее", "reports/submission_xgb_weighted.csv"),
                 ("weighted_windows", "Текущий weighted Windows", "reports/submission_xgb_tail.csv"),
                 ("head_params", "XGBoost с отдельными параметрами голов", "reports/hypotheses/head_params/submission.csv")]
    reports = {name: json.loads((ROOT / f"reports/architectures/{name}/report.json").read_text(encoding="utf-8")) for name in config["order"]}
    for report in reports.values():
        if report["status"] != "complete" or hashlib.sha256((ROOT / report["submission"]).read_bytes()).hexdigest() != report["submission_sha256"]:
            raise ValueError("Incomplete or modified architecture submission")
    files = [ROOT / row[2] for row in baselines]+[ROOT / report["submission"] for report in reports.values()]
    diagnostic = diagnose(ROOT / "data/raw/kaggle/test.csv", files, ROOT / "reports/architectures_test_diagnostics")
    official = json.loads((ROOT / "reports/hypotheses/kaggle_submissions.json").read_text(encoding="utf-8"))["submissions"]
    official += [{"competition": "molecular-activity-prediction-hack", "status": "SubmissionStatus.COMPLETE",
                  "submission_sha256": "92a0ea69f93cc5c48663ec0b84d68f853ae4c486c5d0b55ee713f5c85b787169", "public_score": 1967765.75828}]
    scored = ROOT / "reports/architectures/kaggle_submissions.json"
    if scored.exists():
        official += json.loads(scored.read_text(encoding="utf-8"))["submissions"]
    rows = []
    for (name, description, path), metric in zip(baselines, diagnostic["results"][:len(baselines)]):
        rows.append({"method": name, "name": description, "selected": "historical control", "local_subset_mse": metric["mse_ratio_subset"],
            "n_evaluated": metric["n_evaluated"], "n_test": metric["n_test"], "official_kaggle_score": official_score_for(ROOT / path, official),
            "submission": path, "eligible": False})
    for (name, report), metric in zip(reports.items(), diagnostic["results"][len(baselines):]):
        selected = report["selected"]
        rows.append({"method": name, "name": NAMES[name], "selected": selected, "local_subset_mse": metric["mse_ratio_subset"],
            "n_evaluated": metric["n_evaluated"], "n_test": metric["n_test"],
            "molecule_measured_oof_mse": report["protocols"]["molecule"][selected]["measured"]["mse"],
            "scaffold_measured_oof_mse": report["protocols"]["scaffold"][selected]["measured"]["mse"],
            "molecule_all_oof_mse": report["protocols"]["molecule"][selected]["mse"],
            "scaffold_all_oof_mse": report["protocols"]["scaffold"][selected]["mse"],
            "official_kaggle_score": official_score_for(ROOT / report["submission"], official), "submission": report["submission"],
            **submission_decision(metric["mse_ratio_subset"], report["submission_sha256"], official,
                                  config["submit_if_local_subset_mse_strictly_below"])})
    qualifying = sorted([row for row in rows if row["eligible"]], key=lambda r: r["local_subset_mse"])
    result = {"formula": "raw SI mean squared error; lower is better", "local_reference_coverage": diagnostic["coverage_by_status"],
        "can_compute_identical_official_score_for_new_predictions": False,
        "reason": "Organizer test ground truth and public mask unavailable; official scores are exact-file lookups only",
        "no_score_calibration_to_leaderboard": True, "threshold": config["submit_if_local_subset_mse_strictly_below"],
        "selection_of_internal_variants": config["selection"], "selection_used_test_endpoints": False,
        "post_inference_submission_choice_authorized_by_user": True,
        "best_new_method": min([r for r in rows if r["method"] in config["order"]], key=lambda r: r["local_subset_mse"]),
        "best_eligible": qualifying[0] if qualifying else None, "results": rows}
    directory = ROOT / "reports/architectures"
    (directory / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(rows).to_csv(directory / "comparison.csv", index=False)
    lines = ["# Сравнение архитектур", "", "Score — MSE сырого SI, меньше лучше. Локальный срез: 468/597 точных отношений.",
        "Для нового файла идентичный Kaggle Score локально вычислить нельзя: нет официальных целей и public-маски.",
        "Официальные значения ниже взяты только из завершённых заявок с совпадающим SHA256.", "",
        "| Метод | Вариант | OOF измеренных SI, molecule | OOF измеренных SI, scaffold | Локальный Score | Kaggle Score |",
        "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        oof = [f"{row[key]:.2f}" if key in row else "—" for key in ("molecule_measured_oof_mse", "scaffold_measured_oof_mse")]
        public = f"{row['official_kaggle_score']:.5f}" if row["official_kaggle_score"] is not None else "—"
        lines.append(f"| {row['name']} | {row['selected']} | {oof[0]} | {oof[1]} | {row['local_subset_mse']:.2f} | {public} |")
    lines += ["", "Внутренние варианты выбраны до диагностики тестовых endpoints, по исходным измеренным TRAIN SI.",
              "Условные SI=10000 остаются отдельной OOF-стратой. Все методы используют одинаковые группы и holdout исключён.",
              "CheMeleon и Uni-Mol проверены с замороженными авторскими весами; полного fine-tuning энкодеров в этом опыте нет.",
              "Для одной условной отправки нужен лучший из четырёх новых методов с локальным MSE строго ниже 1 900 000."]
    (directory / "comparison.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(pd.DataFrame(rows)[["method", "selected", "local_subset_mse", "official_kaggle_score", "eligible"]].to_string(index=False))
    print("[best eligible] " + json.dumps(result["best_eligible"], ensure_ascii=False))


if __name__ == "__main__":
    main()
