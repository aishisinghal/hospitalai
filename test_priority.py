
import requests

BASE = "http://127.0.0.1:5000"

response = requests.get(f"{BASE}/api/intakes")
data = response.json()
patients = data["intakes"]

print("PATIENTS:", response.status_code)

for patient in patients:
    print(patient["name"], "->", patient["priority"])

if not patients:
    print("No patients found!")
    raise SystemExit

patient_id = patients[0]["patient_id"]

response = requests.patch(
    f"{BASE}/api/intakes/{patient_id}/priority",
    json={"priority": "high"}
)
print("SET HIGH:", response.status_code, response.json())

response = requests.get(f"{BASE}/api/intakes/{patient_id}")
print("READ BACK:", response.json())

response = requests.patch(
    f"{BASE}/api/intakes/{patient_id}/priority",
    json={"priority": "unassigned"}
)
print("RESET:", response.status_code, response.json())

response = requests.patch(
    f"{BASE}/api/intakes/{patient_id}/priority",
    json={"priority": "urgent"}
)
print("INVALID PRIORITY REJECTED:", response.status_code)
