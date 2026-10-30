import json
import urllib.request

BASE = "http://127.0.0.1:5000"


def call(url, payload=None):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


payload = {
    "name": "Test Document Patient (fictional)",
    "age": 30,
    "language": "en",
    "complaints": "Cough and mild fever for 3 days (fictional test).",
    "document": {
        "original_text": "Complaint: Cogh and mild fever for 3 days",
        "reviewed_text": "Complaint: Cough and mild fever for 3 days",
        "pages": 1,
    },
}

saved = call(BASE + "/api/intake", payload)
print("SAVED:", saved)

documents = call(BASE + "/api/intakes/" + saved["patient_id"] + "/documents")
print("DOCUMENTS:", json.dumps(documents, indent=2))