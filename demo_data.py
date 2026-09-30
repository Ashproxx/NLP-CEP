"""Reproducible fictional Bengaluru feedback, never real citizen records."""
import random
import pandas as pd
from analysis import LOCATIONS, SERVICES


def demo_data():
    rng = random.Random(11)
    rows = []
    end = pd.Timestamp.today().normalize()
    for days_ago in range(42):
        for _ in range(8):
            service = rng.choice(list(SERVICES))
            location = rng.choice(list(LOCATIONS))
            phrase = rng.choice(SERVICES[service])
            tone = rng.choices(["positive", "negative", "neutral"], [3, 5 if days_ago < 7 else 3, 2])[0]
            text = {"positive": f"Excellent {phrase} in {location}. Thank you, I am happy!", "negative": f"Terrible {phrase} in {location}. I am angry and disappointed with this awful service.", "neutral": f"Please share the schedule for {phrase} in {location}."}[tone]
            rows.append({"date": end - pd.Timedelta(days=days_ago), "text": text, "source": rng.choice(["Survey", "Web portal", "Social media export", "Mobile app export"]), "location": location})
    return pd.DataFrame(rows)
