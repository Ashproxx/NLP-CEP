"""Run with: streamlit run app.py"""
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import streamlit as st

from analysis import LOCATIONS, analyze, build_models, concerns, daily_trend, forecast, llm_brief, safe_csv, validate_frame
from demo_data import demo_data

st.set_page_config(page_title="CivicPulse | Citizen Sentiment", page_icon="🏙️", layout="wide")
st.markdown("<style>.block-container{padding-top:2rem}h1{letter-spacing:-1.5px}[data-testid=stMetric]{border:1px solid #dce5e3;border-radius:12px;padding:18px}</style>", unsafe_allow_html=True)
st.caption("CIVICPULSE  /  SDG 11 · SUSTAINABLE CITIES")
st.title("Listen to your city.")
st.write("Turn citizen feedback into service insights and evidence for municipal planning.")


@st.cache_resource
def models():
    return build_models()


@contextmanager
def database():
    directory = Path(__file__).parent / "data"
    directory.mkdir(exist_ok=True)
    connection = sqlite3.connect(directory / "feedback.sqlite3")
    try:
        with connection:
            connection.execute("CREATE TABLE IF NOT EXISTS feedback (date TEXT, text TEXT, source TEXT, location TEXT)")
            yield connection
    finally:
        connection.close()


with st.sidebar:
    st.header("Workspace")
    dataset = st.radio("Feedback source", ["Demo data", "Upload CSV", "Saved feedback"])
    st.caption("English-language prototype · Bengaluru location gazetteer")
    st.divider()
    st.write("**How it works**")
    st.caption("VADER sentiment · TF-IDF service classifier · keyword emotions · location gazetteer · linear trend forecast")

with st.expander("Submit citizen feedback"):
    with st.form("intake", clear_on_submit=True):
        text = st.text_area("Your feedback", max_chars=5000, placeholder="Tell us about a municipal service…")
        location = st.selectbox("Location", ["Unknown", *LOCATIONS])
        submitted = st.form_submit_button("Save feedback")
    if submitted:
        try:
            row = validate_frame(pd.DataFrame([{"text": text, "location": location, "source": "Web portal"}])).iloc[0]
            with database() as connection:
                connection.execute("INSERT INTO feedback VALUES (?, ?, ?, ?)", (row["date"].date().isoformat(), row["text"], row["source"], row["location"]))
            st.success("Saved locally. Select Saved feedback in the sidebar to analyze it.")
        except ValueError as error:
            st.error(str(error))
        except sqlite3.Error:
            st.error("Could not save feedback. Check that the data directory is writable.")

if dataset == "Demo data":
    raw = demo_data()
    st.info("DEMO · 336 fictional records. These insights do not describe real citizens or city conditions.")
elif dataset == "Upload CSV":
    upload = st.file_uploader("CSV: text required; date, source, location optional", type="csv")
    st.download_button("Download example CSV", safe_csv(demo_data()), "example_feedback.csv", "text/csv")
    if upload is None:
        st.stop()
    if upload.size > 5_000_000:
        st.error("CSV must be smaller than 5 MB.")
        st.stop()
    try:
        raw = pd.read_csv(upload)
    except (ValueError, UnicodeError, pd.errors.ParserError) as error:
        st.error(f"Unable to read CSV: {error}")
        st.stop()
else:
    with database() as connection:
        raw = pd.read_sql_query("SELECT date, text, source, location FROM feedback", connection)
    if raw.empty:
        st.info("No saved feedback yet. Submit a response above.")
        st.stop()

try:
    data = analyze(raw, *models())
except ValueError as error:
    st.error(str(error))
    st.stop()

with st.sidebar:
    services = st.multiselect("Services", sorted(data["service"].unique()), default=sorted(data["service"].unique()))
    locations = st.multiselect("Locations", sorted(data["location"].unique()), default=sorted(data["location"].unique()))
    sources = st.multiselect("Channels", sorted(data["source"].unique()), default=sorted(data["source"].unique()))
    dates = st.date_input("Date range", value=(data["date"].min().date(), data["date"].max().date()))

filtered = data[data["service"].isin(services) & data["location"].isin(locations) & data["source"].isin(sources)]
if len(dates) == 2:
    filtered = filtered[filtered["date"].between(pd.Timestamp(dates[0]), pd.Timestamp(dates[1]))]
if filtered.empty:
    st.info("No feedback matches these filters.")
    st.stop()

columns = st.columns(4)
columns[0].metric("Responses", len(filtered))
columns[1].metric("Positive feedback", f"{filtered['sentiment'].eq('Positive').mean():.0%}")
columns[2].metric("Negative feedback", f"{filtered['sentiment'].eq('Negative').mean():.0%}")
columns[3].metric("Sentiment index / 100", f"{(filtered['score'].mean() + 1) * 50:.1f}")
st.caption("The sentiment index rescales VADER scores to 0–100; it is a proxy, not a measured satisfaction rating.")
overview, geography, decisions, records = st.tabs(["Overview", "City map", "Decision support", "Feedback explorer"])
with overview:
    left, right = st.columns(2)
    with left:
        st.subheader("Sentiment by service")
        st.bar_chart(pd.crosstab(filtered["service"], filtered["sentiment"]).reindex(columns=["Negative", "Neutral", "Positive"], fill_value=0), color=["#dc685d", "#a1aeb0", "#219b82"])
    with right:
        st.subheader("Daily sentiment")
        st.line_chart(daily_trend(filtered)["score"], color="#168f7a")
    st.subheader("Emotion signals")
    st.bar_chart(filtered["emotion"].value_counts(), horizontal=True)
    st.caption("Emotion labels use keywords and can miss negation, sarcasm and mixed emotions.")
with geography:
    st.subheader("Where feedback is coming from")
    mapped = filtered.dropna(subset=["lat", "lon"])
    if not mapped.empty:
        points = mapped.groupby(["location", "lat", "lon"]).size().reset_index(name="responses")
        points["size"] = points["responses"] * 15 + 80
        st.map(points, latitude="lat", longitude="lon", size="size", color="#168f7a")
        st.dataframe(points[["location", "responses"]], hide_index=True, width="stretch")
    st.caption(f"{len(mapped)} of {len(filtered)} records mapped to neighborhood centroids. Unknown locations are excluded. Basemap needs internet.")
with decisions:
    st.subheader("Service priorities")
    table = concerns(filtered)
    st.caption("Negative counts in the last 7 days of the selected data versus the preceding 7 days. Counts reflect participation as well as service quality; partial windows are not comparable.")
    if table.empty:
        st.info("No negative feedback in the latest seven-day window.")
    else:
        st.dataframe(table, width="stretch")
    st.subheader("Seven-day sentiment outlook")
    prediction = forecast(filtered)
    if prediction.empty:
        st.info("At least seven distinct observed days are needed for a forecast.")
    else:
        st.line_chart(prediction.set_index("date"), color="#168f7a")
    st.caption("Experimental linear extrapolation of up to 30 observed days, weighted by response count. No calibrated uncertainty or validated predictive accuracy; do not use it to allocate resources automatically.")
    with st.expander("Optional local LLM planning brief"):
        st.write("Requires Ollama running locally with an installed model. Only aggregate service counts are sent; citizen text stays out of the prompt.")
        model = st.text_input("Installed Ollama model", "llama3.2")
        if st.button("Generate planning brief", disabled=table.empty):
            try:
                with st.spinner("Generating draft…"):
                    st.write(llm_brief(table, model))
                st.caption("AI-generated draft. Verify every recommendation before acting.")
            except (OSError, ValueError, KeyError) as error:
                st.error(f"Local model unavailable: {error}")
with records:
    query = st.text_input("Search feedback")
    shown = filtered[filtered["text"].str.contains(query, case=False, regex=False)]
    st.dataframe(shown.drop(columns=["lat", "lon"]), hide_index=True, width="stretch")
    st.download_button("Export analyzed feedback", safe_csv(shown), "civicpulse_analysis.csv", "text/csv")
st.divider()
st.caption("CivicPulse · Academic decision-support prototype · Human review required · Avoid uploading personal or sensitive information.")
