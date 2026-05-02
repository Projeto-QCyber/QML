import os, json, random
import pandas as pd, requests
from dotenv import load_dotenv; load_dotenv()

API_URL = f"http://localhost:{os.getenv('FLASK_API_PORT',5000)}/analisar"
df = pd.read_csv("data/test/dados_de_teste.csv")
att = df[df.Attack_type != 0]
first3 = att.sample(3, replace=len(att)<3)
rest = df.drop(first3.index)
last2 = rest.sample(2, replace=len(rest)<2)
samples = pd.concat([first3, last2], ignore_index=True)
assert (samples.Attack_type.iloc[:3] != 0).all()

print("labels:", samples.Attack_type.tolist(),
      "flags:", ["safe" if x==0 else "attack" for x in samples.Attack_type])

payload = {
    "device_id": f"test-device-{random.randint(100,999)}",
    "samples": samples.drop(columns=['Attack_type', 'Attack_label']).to_dict("records"),
}

r = requests.post(API_URL, json=payload); r.raise_for_status()
print(json.dumps(r.json(), indent=2))
