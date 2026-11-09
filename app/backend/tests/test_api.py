"""API smoke tests.  Run from backend/:  venv/Scripts/python -m tests.test_api"""

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

import database  # noqa: E402
from main import app  # noqa: E402

SAMPLES = Path(__file__).resolve().parents[1] / "samples"
HEADER = "year,infl,rGDP_pc,unemp,govdebt_GDP,CA_GDP,USDfx,REER,pop"


def post(client, text, model="RandomForest", name="t.csv"):
    return client.post("/api/predict", files={"file": (name, io.BytesIO(text.encode()), "text/csv")},
                       data={"model_name": model})


def main():
    with TestClient(app) as c:
        h = c.get("/api/health").json()
        assert h["status"] == "ok" and set(h["models_available"]) == {"RandomForest", "XGBoost", "Persistence"}, h

        m = c.get("/api/models").json()
        assert len(m["models"]) == 3 and m["benchmark"], m

        c.delete("/api/predictions")
        for f in sorted(SAMPLES.glob("sample_*.csv")):
            text = f.read_text()
            for model in ["RandomForest", "XGBoost", "Persistence"]:
                r = post(c, text, model, f.name)
                assert r.status_code == 200, (f.name, model, r.text)
                j = r.json()
                assert j["model"] == model and 0 < j["prediction"] < 90, j
                assert len(j["history"]) == 10 and j["forecast_end"] - j["forecast_start"] == 4
            print(f.name, post(c, text, "RandomForest").json()["all_predictions"])

        # the file from the original instruction PDF (synthetic numbers) must also work
        pdf = HEADER + """
2015,5.5,15000,5.2,45.0,-2.1,85.0,120.0,160000000
2016,5.8,15500,5.1,46.0,-2.3,86.0,121.0,162000000
2017,6.1,16000,5.0,47.0,-2.0,87.0,122.0,164000000
2018,5.9,16500,4.8,48.0,-1.8,88.0,123.0,166000000
2019,6.0,17000,4.5,49.0,-1.5,89.0,124.0,168000000
2020,5.2,16500,6.5,52.0,-2.5,90.0,120.0,170000000
2021,5.7,17200,6.0,53.0,-2.2,91.0,121.0,172000000
2022,6.3,17800,5.5,54.0,-2.0,92.0,122.0,174000000
2023,6.0,18300,5.2,55.0,-1.9,93.0,123.0,176000000
2024,5.9,18800,5.0,56.0,-1.8,94.0,124.0,178000000
"""
        r = post(c, pdf, "Persistence")
        assert r.json()["prediction"] == round((5.2 + 5.7 + 6.3 + 6.0 + 5.9) / 5, 2), r.json()
        assert post(c, pdf, "XGBoost").status_code == 200

        # error handling
        assert post(c, pdf, "Nope").status_code == 400
        assert post(c, "", "XGBoost").status_code == 422
        assert post(c, "a,b\n1,2\n", "XGBoost").status_code == 422
        assert post(c, HEADER + "\n2020,1,1,1,1,1,1,1,1\n", "XGBoost").status_code == 422        # too few rows
        gap = "\n".join(pdf.splitlines()[:5] + pdf.splitlines()[6:])                               # missing a year
        assert post(c, gap, "XGBoost").status_code == 422
        nan_rows = pdf.replace("5.1,46.0", ",46.0")  # a single missing value is fillable (<= 2 years)
        assert post(c, nan_rows, "XGBoost").status_code == 200
        allnan = "\n".join([HEADER] + [l.split(",")[0] + ",5,,5,5,5,5,5,5" for l in pdf.splitlines()[1:]])
        assert post(c, allnan, "XGBoost").status_code == 422
        assert post(c, pdf.replace("5.5,15000", "abc,15000"), "XGBoost").status_code == 200      # fillable (bfill)
        assert post(c, pdf.replace("2020,5.2", "2020,5.2").replace("2019,", "2019.5,"), "XGBoost").status_code == 422

        hist = c.get("/api/predictions?limit=5").json()
        assert hist and "all_predictions" in hist[0]
        assert c.delete(f"/api/predictions/{hist[0]['id']}").status_code == 200
        assert c.delete("/api/predictions/99999999").status_code == 404
        assert c.get("/api/samples").json()
        assert c.get("/api/samples/sample_USA.csv").status_code == 200
        assert c.get("/api/samples/..%2Fmain.py").status_code == 404
        c.delete("/api/predictions")
    print("ALL API TESTS PASSED")


if __name__ == "__main__":
    main()
