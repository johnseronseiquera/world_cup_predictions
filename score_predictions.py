"""
score_predictions.py — update the prediction log with real match results
========================================================================

    conda activate poet
    python score_predictions.py
    python review_predictions.py   # score + print matchday report
"""

from prediction_log import score_predictions

if __name__ == "__main__":
    score_predictions()
