"""Run: python -m unittest -v"""
import unittest
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from streamlit.testing.v1 import AppTest
from analysis import analyze, build_models, concerns, forecast, safe_csv, validate_frame
from demo_data import demo_data


class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = build_models()

    def test_analysis_and_validation(self):
        frame = pd.DataFrame({"text": ["Excellent bus service in Hebbal. I am delighted!", "Terrible garbage collection in Whitefield. I am angry!", "Astronomy discussion"]})
        result = analyze(frame, *self.models)
        self.assertEqual(result["sentiment"].tolist()[:2], ["Positive", "Negative"])
        self.assertEqual(result["service"].tolist(), ["Transport", "Sanitation", "Other"])
        self.assertEqual(result["location"].tolist(), ["Hebbal", "Whitefield", "Unknown"])
        self.assertEqual(result["emotion"].iloc[1], "Anger")
        for bad in (pd.DataFrame({"wrong": ["hi"]}), pd.DataFrame({"text": [None]}), pd.DataFrame({"text": ["hello"], "date": ["bad"]}), pd.DataFrame({"text": ["hello"], "date": ["2099-01-01"]})):
            with self.assertRaises(ValueError):
                validate_frame(bad)
        self.assertIn("'=1+1", safe_csv(pd.DataFrame({"text": ["=1+1"]})).decode())

    def test_forecast_and_concern_windows(self):
        data = analyze(demo_data(), *self.models)
        predicted = forecast(data)
        self.assertEqual(len(predicted), 7)
        self.assertTrue(predicted["sentiment_index"].between(0, 100).all())
        self.assertGreater(predicted["date"].min(), data["date"].max())
        self.assertTrue(forecast(data.head(8)).empty)
        end = data["date"].max()
        expected = data[(data["date"] > end - pd.Timedelta(days=7)) & data["sentiment"].eq("Negative")]
        self.assertEqual(concerns(data)["negative_last_7_days"].sum(), len(expected))

    def test_dashboard(self):
        app = AppTest.from_file("app.py", default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value, "336")
        app.text_input[1].set_value("nonexistent-search-phrase").run()
        self.assertFalse(app.exception)
        app.sidebar.multiselect[0].set_value([]).run()
        self.assertFalse(app.exception)
        self.assertTrue(any("No feedback matches" in message.value for message in app.info))

    def test_feedback_persistence(self):
        connect = sqlite3.connect
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feedback.sqlite3"
            with patch("sqlite3.connect", side_effect=lambda _: connect(path)):
                app = AppTest.from_file("app.py", default_timeout=30).run()
                app.text_area[0].set_value("Excellent bus service in Hebbal. Thank you!")
                app.selectbox[0].set_value("Hebbal")
                app.button[0].click().run()
                self.assertFalse(app.exception)
                self.assertTrue(app.success)
                app.sidebar.radio[0].set_value("Saved feedback").run()
                self.assertFalse(app.exception)
                self.assertEqual(app.metric[0].value, "1")


if __name__ == "__main__":
    unittest.main()
