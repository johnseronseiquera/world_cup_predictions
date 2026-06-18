# 2026 FIFA World Cup match predictor

Predict win / draw / loss probabilities, opponent-adjusted expected goals, and a branded chart for any 2026 World Cup match.

```
conda activate poet
pip install -r requirements.txt
python predict_today.py "Saudi Arabia" "Uruguay"
```

## Daily workflow

```bash
python sync_results.py              # pull latest match results
python sync_player_stats.py         # refresh club stats
python import_fbref.py --dir data_cache/fbref_imports   # optional FBref import
python predict_today.py "Spain" "Cabo Verde"            # single match
python predict_slate.py 2026-06-19                      # all matches on a date
python score_predictions.py           # fill in actual results in the prediction log
python review_predictions.py          # sync, score, and print matchday report
python backtest_goals.py              # validate goal model on past World Cups
```

## How it works

**Team model (XGBoost)** uses Elo ratings, recent form, head-to-head, tournament context, and squad features built from international goal scorers plus optional club-form data.

**Goal model (Poisson)** uses opponent-adjusted scoring rates (minnow blowouts discounted via Elo) to produce expected goals, most likely scorelines, over/under 2.5, and BTTS.

**Calibration** applies isotonic regression on the validation set so reported percentages better match real frequencies.

Each run saves a two-panel chart (win probabilities + opponent-adjusted xG) under `predictions/<date>/` and logs the pick to `predictions/prediction_log.csv`.

## Data setup

Clone [martj42/international_results](https://github.com/martj42/international_results) next to this repo:

```text
~/international_results/
~/world_cup_predictions/
```

`sync_results.py` pulls updates from `martj42/international_results` via git.

Squad lists live in `data_cache/squads_2026.csv` (bootstrap with `python bootstrap_squads.py`). Club stats go in `data_cache/player_club_stats_manual.csv` or import FBref CSVs with `python import_fbref.py`.

## Scripts

| Script | Purpose |
|--------|---------|
| `predict_today.py` | Predict one match |
| `predict_slate.py` | Predict all fixtures on a date (trains once) |
| `sync_results.py` | Update `results.csv` from sibling repo |
| `sync_player_stats.py` | Merge manual club stats into player file |
| `import_fbref.py` | Bulk-import FBref CSV exports |
| `bootstrap_squads.py` | Build squad list from recent intl scorers |
| `score_predictions.py` | Fill in actual results in the prediction log |
| `review_predictions.py` | Sync results, score picks, print matchday report |
| `backtest_goals.py` | Backtest Poisson goals on WC 2018/2022 |
| `inspect_data.py` | Summarize international_results CSVs |

## License

MIT
