"""
review_predictions.py — score and review predictions after each matchday
========================================================================

Sync results, fill in actual scores, then print a matchday report.

    conda activate poet
    python review_predictions.py              # all matchdays
    python review_predictions.py 2026-06-18     # one date
    python review_predictions.py --no-sync        # skip git pull
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from prediction_log import LOG_PATH, load_log, score_predictions


def _dedupe_log(log: pd.DataFrame) -> pd.DataFrame:
    """One row per fixture: prefer scored rows, else latest prediction."""
    if log.empty:
        return log
    log = log.copy()
    log["_sort_scored"] = log["actual_result"].notna().astype(int)
    log["_sort_time"] = pd.to_datetime(log["predicted_at"], utc=True, errors="coerce")
    log = log.sort_values(["_sort_scored", "_sort_time"])
    key = ["match_date", "home_team", "away_team"]
    log = log.drop_duplicates(key, keep="last")
    return log.drop(columns=["_sort_scored", "_sort_time"])


def _goal_metrics(row: pd.Series) -> dict[str, float | bool | None]:
    if pd.isna(row.get("actual_home_score")) or pd.isna(row.get("pred_home_goals")):
        return {"exact_score": None, "home_mae": None, "away_mae": None, "total_mae": None}
    ah = float(row["actual_home_score"])
    aa = float(row["actual_away_score"])
    ph = float(row["pred_home_goals"])
    pa = float(row["pred_away_goals"])
    eh = float(row["exp_home_goals"]) if pd.notna(row.get("exp_home_goals")) else ph
    ea = float(row["exp_away_goals"]) if pd.notna(row.get("exp_away_goals")) else pa
    return {
        "exact_score": bool(ph == ah and pa == aa),
        "home_mae": abs(eh - ah),
        "away_mae": abs(ea - aa),
        "total_mae": abs(eh - ah) + abs(ea - aa),
    }


def _fmt_score(row: pd.Series) -> str:
    if pd.isna(row.get("actual_home_score")):
        return "pending"
    return f"{int(row['actual_home_score'])}-{int(row['actual_away_score'])}"


def _fmt_pred_score(row: pd.Series) -> str:
    if pd.isna(row.get("pred_home_goals")):
        return "—"
    return f"{int(row['pred_home_goals'])}-{int(row['pred_away_goals'])}"


def _print_match(row: pd.Series) -> None:
    actual = _fmt_score(row)
    pred_score = _fmt_pred_score(row)
    pick_ok = row.get("correct_pick")
    if pd.isna(pick_ok):
        result_tag = "pending"
    elif pick_ok:
        result_tag = "✓ pick"
    else:
        result_tag = "✗ pick"

    goals = _goal_metrics(row)
    if goals["exact_score"] is True:
        score_tag = "✓ score"
    elif goals["exact_score"] is False:
        score_tag = "✗ score"
    else:
        score_tag = ""

    print(
        f"  {row['home_team']} vs {row['away_team']:<22} "
        f"pick {row['pick']:<20} ({row['pick_confidence']*100:>5.1f}%)  "
        f"pred {pred_score:<5} actual {actual:<7}  {result_tag}  {score_tag}"
    )
    if goals["total_mae"] is not None:
        print(
            f"    xG {row['exp_home_goals']:.2f}-{row['exp_away_goals']:.2f}  "
            f"goal MAE {goals['total_mae']:.2f}  "
            f"draw prob {row['p_draw']*100:.1f}%"
        )


def _print_summary(title: str, df: pd.DataFrame) -> None:
    scored = df[df["actual_result"].notna()].copy()
    pending = df[df["actual_result"].isna()]

    print(f"\n{'=' * 72}")
    print(title)
    print(f"{'=' * 72}")
    print(f"matches: {len(df)}   scored: {len(scored)}   pending: {len(pending)}")

    if scored.empty:
        print("  no scored matches yet")
        return

    picks = scored["correct_pick"].astype(float)
    draws_actual = (scored["actual_result"] == "Draw").mean()
    draws_pred = scored["p_draw"].mean()

    print(f"pick accuracy      : {picks.mean():.1%}  ({int(picks.sum())}/{len(scored)})")
    print(f"mean log-loss      : {scored['log_loss'].mean():.3f}")
    print(f"draw rate (actual) : {draws_actual:.1%}")
    print(f"draw rate (predicted avg): {draws_pred:.1%}")

    goal_rows = []
    for _, row in scored.iterrows():
        m = _goal_metrics(row)
        if m["exact_score"] is not None:
            goal_rows.append(m)
    if goal_rows:
        exact = np.mean([g["exact_score"] for g in goal_rows])
        total_mae = np.mean([g["total_mae"] for g in goal_rows])
        print(f"exact scoreline hit: {exact:.1%}")
        print(f"mean total goal MAE: {total_mae:.2f}")

    for tag in ["LOCK", "LEAN", "TOSS-UP"]:
        sub = scored[scored["tag"].astype(str).str.startswith(tag)]
        if sub.empty:
            continue
        print(f"  {tag:<8} accuracy: {sub['correct_pick'].mean():.1%}  (n={len(sub)})")


def review_predictions(match_date: str | None = None, sync: bool = True) -> pd.DataFrame:
    if sync:
        from sync_results import sync_results

        print("Syncing latest results ...")
        sync_results()

    print("Scoring pending predictions ...")
    score_predictions()

    log = _dedupe_log(load_log())
    if match_date:
        log = log[log["match_date"].astype(str) == match_date]

    if log.empty:
        print(f"no predictions in log -> {LOG_PATH}")
        return log

    dates = sorted(log["match_date"].astype(str).unique())
    for day in dates:
        day_df = log[log["match_date"].astype(str) == day].sort_values("predicted_at")
        _print_summary(f"Matchday {day}", day_df)
        for _, row in day_df.iterrows():
            _print_match(row)

    if match_date is None and len(dates) > 1:
        _print_summary("Tournament total (deduped)", log)

    return log


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--no-sync"]
    sync = "--no-sync" not in sys.argv
    match_date = args[0] if args else None
    review_predictions(match_date=match_date, sync=sync)


if __name__ == "__main__":
    main()
