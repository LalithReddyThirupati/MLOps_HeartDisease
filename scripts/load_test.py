"""Send randomised valid requests to the API to populate the Grafana dashboard.

Standard library only. Usage:
    python scripts/load_test.py --url http://localhost:8000 --n 300 --rate 5
    python scripts/load_test.py --url http://localhost:8080        # Kubernetes
"""
import argparse
import json
import random
import time
import urllib.error
import urllib.request


def random_patient() -> dict:
    record = {
        "age": random.randint(29, 77), "sex": random.randint(0, 1),
        "cp": random.randint(1, 4), "trestbps": random.randint(94, 200),
        "chol": random.randint(126, 564), "fbs": random.randint(0, 1),
        "restecg": random.randint(0, 2), "thalach": random.randint(71, 202),
        "exang": random.randint(0, 1), "oldpeak": round(random.uniform(0, 6.2), 1),
        "slope": random.randint(1, 3), "ca": random.randint(0, 3),
        "thal": random.choice([3, 6, 7]),
    }
    if random.random() < 0.1:          # some requests omit optional fields
        record.pop("ca")
    if random.random() < 0.05:         # a few deliberately invalid requests (422s)
        record["age"] = -1
    return record


def main(url: str, n: int, rate: float):
    counts = {}
    for _ in range(n):
        req = urllib.request.Request(f"{url}/predict", method="POST",
                                     data=json.dumps(random_patient()).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                status = r.status
        except urllib.error.HTTPError as e:
            status = e.code
        except urllib.error.URLError as e:
            raise SystemExit(f"Cannot reach {url}: {e.reason}")
        counts[status] = counts.get(status, 0) + 1
        time.sleep(1 / rate)
    print("status counts:", counts)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--rate", type=float, default=5.0, help="requests per second")
    a = p.parse_args()
    main(a.url.rstrip("/"), a.n, a.rate)
