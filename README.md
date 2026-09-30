# CivicPulse — Citizen Sentiment Analysis for Smart Cities

An SDG 11 academic project that turns municipal feedback into sentiment, service, location and trend insights. Built with Python, Streamlit, scikit-learn, VADER and SQLite.

## Run locally

Python 3.10–3.12 recommended. From the repository directory:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m streamlit run app.py
```

On macOS/Linux, replace `.venv\Scripts\python` with `.venv/bin/python`. Open the local URL printed by Streamlit (normally http://localhost:8501).

## What you can do

- Explore 336 reproducible, clearly labeled **fictional** Bengaluru feedback records.
- Import feedback from survey, social-media or mobile-app CSV exports.
- Submit feedback through the web form; entries persist in local SQLite.
- Classify feedback into Water, Roads, Sanitation, Transport, Lighting, Parks or Other.
- Inspect positive/neutral/negative sentiment, emotion keywords and neighborhood entities.
- Filter by service, location, channel and date; search and export analyzed records.
- Map neighborhood-level response counts and inspect daily sentiment.
- Compare negative service counts across adjacent seven-day windows and review suggested actions.
- Explore a seven-day sentiment-index forecast when at least seven days are available.
- Optionally generate a planning brief with a locally installed Ollama model.

## Input format

```csv
date,text,source,location
2026-09-20,"Terrible garbage collection in Whitefield. I am frustrated!",Survey,Whitefield
2026-09-21,"Excellent bus service in Hebbal. Thank you!",Web portal,Hebbal
```

Only `text` is required (3–5,000 characters). Missing dates use today's local date; malformed or future dates are rejected. Missing sources become `CSV`. Files are limited to 5 MB and 10,000 rows. The upload screen offers a complete example download. CSV imports remain in the current session; only web-form submissions are saved to `data/feedback.sqlite3`, which is excluded from Git. Exports neutralize spreadsheet formula prefixes.

## Method and architecture

```text
CSV / web form / fictional demo
              |
       schema validation
              |
  VADER sentiment + TF-IDF/logistic service classifier
  + emotion keywords + neighborhood gazetteer
              |
  filters / map / trends / priorities / CSV export
              |
  weighted linear forecast / optional local LLM brief
```

`analysis.py` contains validation, models, aggregations and export handling. `app.py` is the dashboard and SQLite intake. `demo_data.py` creates deterministic fictional examples relative to the run date. `test_project.py` checks analysis behavior, invalid inputs, time windows, forecasting and dashboard interactions.

### NLP and machine learning

VADER assigns English sentiment using compound-score thresholds of ±0.05. The service classifier learns TF-IDF word/bigram features with logistic regression from 144 synthetic template examples across six services; maximum class probabilities below 0.40 route to Other. These probabilities are **not calibrated confidence measures**. The demo shares vocabulary with training examples and must not be used as an independent evaluation set.

Emotion detection is a keyword baseline. Location extraction is gazetteer-based entity recognition for Indiranagar, Koramangala, Whitefield, Jayanagar, Malleshwaram and Hebbal. A recognized supplied location takes precedence over a text match; unknown names remain unmapped. Map coordinates are neighborhood centroids, not incident coordinates. Basemap tiles require internet.

### Prediction and decision support

The sentiment index is `(mean VADER compound + 1) × 50`. It is a **satisfaction proxy**, not a survey-derived satisfaction score. A linear regression on up to 30 observed daily means, weighted by response counts, extrapolates seven days beyond the latest selected observation and clips results to 0–100. Missing days are not invented. The model has no validated forecast accuracy or calibrated prediction interval.

Concern ranking uses negative counts over the latest seven days of selected data and reports the difference from the prior seven days. These are count changes, not proof of deteriorating services: sampling volume and incomplete windows affect comparisons. Recommendations are service-specific templates for human review.

### Optional LLM

With Ollama running on `127.0.0.1:11434` and a model already installed, enter its name in **Decision support → Optional local LLM planning brief**. The default model name is `llama3.2`. This sends only aggregate service counts to the local server. No API key or paid service is required. The rest of the dashboard works without an LLM. Generated briefs can be incorrect and need review.

## Verification

```powershell
.venv\Scripts\python -m unittest -v
```

GitHub Actions runs the same checks on pushes and pull requests. Tests verify functionality; they do not establish model accuracy. For research evaluation, collect consented labeled feedback, separate train/test data by source or time, and report macro-F1 and confusion matrices for service and sentiment labels. Compare forecasts to a last-value baseline on held-out dates using MAE before drawing predictive conclusions.

## Scope and limitations

This is a runnable academic prototype, not a production municipal platform. Social media, surveys and mobile applications are supported through CSV exports rather than authenticated live connectors. English, the six services and the six Bengaluru locations are the initial scope. Sarcasm, negation in emotion keywords, multilingual text, mixed-service feedback and unseen locations may be misclassified.

No authentication, moderation or production access controls are included: run locally with non-sensitive data. Public deployment requires access controls, consent and retention policies, reviewed location data and validated models. Do not use automated labels to deny services or make consequential decisions about individuals. Citizen names and contact information are unnecessary for the analysis and should be removed before import.
