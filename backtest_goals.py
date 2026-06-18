"""
backtest_goals.py — evaluate Poisson goal predictions on past World Cups
=========================================================================

    conda activate poet
    python backtest_goals.py
    python backtest_goals.py --tournament 2022
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from goal_prediction import predict_goals
from player_features import normalize_team
from predict_today import ELO_BASE, build_dataset, load_results, per_team_long


def world_cup_matches(results: pd.DataFrame, year: int) -> pd.DataFrame:
    wc = results[
        results["tournament"].str.lower().eq("fifa world cup")
        & (results["date"].dt.year == year)
    ].copy()
    return wc.sort_values("date").reset_index(drop=True)


def backtest_year(results: pd.DataFrame, year: int) -> dict[str, float]:
    matches = world_cup_matches(results, year)
    if matches.empty:
        return {}

    dataset, elo_by_end = build_dataset(results)
    final_elo: dict[str, float] = {}
    for _, row in dataset.sort_values("date").iterrows():
        final_elo[row["home_team"]] = row["home_elo"]
        final_elo[row["away_team"]] = row["away_elo"]

    rows = []
    for _, m in matches.iterrows():
        asof = m["date"]
        history = results[results["date"] < asof]
        if history.empty:
            continue
        long = per_team_long(history)
        preds = predict_goals(
            long,
            m["home_team"],
            m["away_team"],
            asof,
            neutral=bool(m.get("neutral", 1)),
            home_elo=final_elo.get(m["home_team"], ELO_BASE),
            away_elo=final_elo.get(m["away_team"], ELO_BASE),
            elo=final_elo,
        )
        actual_home = int(m["home_score"])
        actual_away = int(m["away_score"])
        rows.append(
            {
                "date": asof.date(),
                "home": m["home_team"],
                "away": m["away_team"],
                "actual": f"{actual_home}-{actual_away}",
                "predicted": f"{preds['pred_home_goals']}-{preds['pred_away_goals']}",
                "exp_home": preds["exp_home_goals"],
                "exp_away": preds["exp_away_goals"],
                "actual_home": actual_home,
                "actual_away": actual_away,
                "exact": int(
                    preds["pred_home_goals"] == actual_home
                    and preds["pred_away_goals"] == actual_away
                ),
                "btts_pred": int(preds["btts_prob"] >= 0.5),
                "btts_actual": int(actual_home >= 1 and actual_away >= 1),
                "over25_pred": int(preds["over_2_5_prob"] >= 0.5),
                "over25_actual": int(actual_home + actual_away >= 3),
            }
        )

    if not rows:
        return {}

    df = pd.DataFrame(rows)
    return {
        "year": year,
        "matches": len(df),
        "exact_score_rate": float(df["exact"].mean()),
        "home_goals_mae": float((df["exp_home"] - df["actual_home"]).abs().mean()),
        "away_goals_mae": float((df["exp_away"] - df["actual_away"]).abs().mean()),
        "total_goals_mae": float(
            ((df["exp_home"] + df["exp_away"]) - (df["actual_home"] + df["actual_away"])).abs().mean()
        ),
        "btts_accuracy": float((df["btts_pred"] == df["btts_actual"]).mean()),
        "over25_accuracy": float((df["over25_pred"] == df["over25_actual"]).mean()),
        "sample": df,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest Poisson goal model on World Cups")
    parser.add_argument("--tournament", type=int, nargs="*", default=[2018, 2022])
    args = parser.parse_args()

    results = load_results(sync=False)
    results["home_team"] = results["home_team"].map(normalize_team)
    results["away_team"] = results["away_team"].map(normalize_team)

    print("Poisson goal model backtest (opponent-adjusted xG)\n")
    for year in args.tournament:
        out = backtest_year(results, year)
        if not out:
            print(f"{year}: no matches found\n")
            continue
        print(f"=== FIFA World Cup {year} ({out['matches']} matches) ===")
        print(f"  exact scoreline hit rate : {out['exact_score_rate']*100:.1f}%")
        print(f"  home goals MAE           : {out['home_goals_mae']:.2f}")
        print(f"  away goals MAE           : {out['away_goals_mae']:.2f}")
        print(f"  total goals MAE          : {out['total_goals_mae']:.2f}")
        print(f"  BTTS accuracy            : {out['btts_accuracy']*100:.1f}%")
        print(f"  Over 2.5 accuracy        : {out['over25_accuracy']*100:.1f}%")
        print("  sample predictions:")
        for _, row in out["sample"].head(5).iterrows():
            print(f"    {row['date']} {row['home']} vs {row['away']}: pred {row['predicted']}  actual {row['actual']}")
        print()


if __name__ == "__main__":
    main()
