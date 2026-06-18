"""
predict_slate.py — predict every World Cup fixture on a given date
==================================================================

Trains once, then runs all group-stage matches scheduled for that day.

    conda activate poet
    python predict_slate.py 2026-06-19
    python predict_slate.py            # defaults to today
"""

from __future__ import annotations

import sys
from datetime import date

import pandas as pd

from predict_today import (
    FIXTURES_PATH,
    MATCH_NEUTRAL,
    MATCH_WEIGHT,
    TRAIN_START,
    VAL_START,
    attach_squad_lookup,
    build_dataset,
    build_squad_feature_lookup,
    find_fixture,
    load_results,
    make_chart,
    per_team_long,
    predict_goals,
    predict_symmetric,
    split_by_date,
    tag_match,
    train_model,
)
from prediction_log import log_prediction
from player_features import load_goalscorers, load_squads, squad_match_features
from sync_player_stats import sync_player_stats


def fixtures_on_date(slate_date: str) -> list[tuple[str, str]]:
    fx = pd.read_csv(FIXTURES_PATH)
    pairs: list[tuple[str, str]] = []
    for _, row in fx.iterrows():
        if str(row.get("date_dt", "")) != slate_date:
            continue
        teams = str(row["teams"])
        if " v " not in teams:
            continue
        left, right = [p.strip() for p in teams.split(" v ")]
        if any(w in teams.lower() for w in ["winner", "runner", "third", "place", "group"]):
            continue
        pairs.append((left, right))
    return pairs


def print_match_result(m, p_home, p_draw, p_away, pick, conf, tag, goals, squad):
    print("\n" + "=" * 60)
    print(f"  {m['home_disp']} vs {m['away_disp']}")
    print(f"  {m['date']}  ·  {m['group']}  ·  {m['stadium']}")
    print("=" * 60)
    print(f"  {m['home_disp']:<22} win   {p_home*100:>5.1f}%")
    print(f"  {'Draw':<22}       {p_draw*100:>5.1f}%")
    print(f"  {m['away_disp']:<22} win   {p_away*100:>5.1f}%")
    print("-" * 60)
    print(f"  PICK: {pick}  ({conf*100:.1f}%)   [{tag}]")
    print(
        f"  Score: {m['home_disp']} {goals['pred_home_goals']}-{goals['pred_away_goals']} {m['away_disp']}  "
        f"(xG {goals['exp_home_goals']:.2f}-{goals['exp_away_goals']:.2f})"
    )
    print(
        f"  Squad goals L10: {m['home_disp']} {squad['home_squad_goals_l10']:.0f}  "
        f"{m['away_disp']} {squad['away_squad_goals_l10']:.0f}"
    )


def main() -> None:
    slate_date = sys.argv[1] if len(sys.argv) > 1 else str(date.today())
    pairs = fixtures_on_date(slate_date)
    if not pairs:
        print(f"No fixtures found for {slate_date}")
        return

    print(f"\nPredicting {len(pairs)} matches on {slate_date} ...")
    results = load_results(sync=False)
    goalscorers = load_goalscorers()
    squads = load_squads()
    club_stats = sync_player_stats()
    squad_ctx = {
        "results": results,
        "goalscorers": goalscorers,
        "squads": squads,
        "club_stats": club_stats,
    }
    dataset, final_elo = build_dataset(results)
    valid_teams = set(results["home_team"]) | set(results["away_team"])
    long = per_team_long(results)

    print(f"Training model (data up to {slate_date}) ...")
    train, val = split_by_date(dataset, TRAIN_START, VAL_START, slate_date)
    lookup = build_squad_feature_lookup(pd.concat([train, val], ignore_index=True), goalscorers)
    train = attach_squad_lookup(train, lookup)
    val = attach_squad_lookup(val, lookup)
    model, _, _, draw_boost = train_model(train, val)

    out_dir = f"predictions/{slate_date}"
    import os
    os.makedirs(out_dir, exist_ok=True)

    for team_a, team_b in pairs:
        m = find_fixture(team_a, team_b)
        if m is None:
            print(f"\nSkipping {team_a} vs {team_b} (fixture not found)")
            continue
        if m["home"] not in valid_teams or m["away"] not in valid_teams:
            print(f"\nSkipping {m['home_disp']} vs {m['away_disp']} (placeholder teams)")
            continue

        match_date = m["date"]
        p_home, p_draw, p_away = predict_symmetric(
            model, long, final_elo, m["home"], m["away"], match_date,
            MATCH_NEUTRAL, MATCH_WEIGHT, squad_ctx, draw_boost,
        )
        squad = squad_match_features(
            results, goalscorers, squads, m["home"], m["away"], match_date, club_stats=club_stats
        )
        outcomes = [(m["home_disp"], p_home), ("Draw", p_draw), (m["away_disp"], p_away)]
        pick, conf = max(outcomes, key=lambda x: x[1])
        he = final_elo.get(m["home"], 1500.0)
        ae = final_elo.get(m["away"], 1500.0)
        tag = tag_match(conf, p_home, p_away, he, ae)
        goals = predict_goals(long, m["home"], m["away"], match_date, MATCH_NEUTRAL, he, ae, final_elo)
        chart = make_chart(m, p_home, p_draw, p_away, goals, match_date, out_dir)
        print_match_result(m, p_home, p_draw, p_away, pick, conf, tag, goals, squad)
        print(f"  Chart -> {chart}")
        log_prediction(
            m, p_home, p_draw, p_away, pick, conf, tag, he, ae,
            pred_home_goals=goals["pred_home_goals"],
            pred_away_goals=goals["pred_away_goals"],
            exp_home_goals=goals["exp_home_goals"],
            exp_away_goals=goals["exp_away_goals"],
        )

    print(f"\nDone. Charts in {out_dir}/")


if __name__ == "__main__":
    main()
