import re
import warnings
from pathlib import Path
import joblib
import sklearn
from sklearn.exceptions import InconsistentVersionWarning

BASE_DIRS = [Path("IA/weights/traditional"), Path("/app/IA/weights/traditional")]
MODELS = {"binary": "random_forest_model_bin.joblib", "multiclass": "random_forest_model_mult.joblib"}

def find_model_path(filename):
    for base in BASE_DIRS:
        p = base / filename
        if p.exists():
            return p
    return None

def load_and_detect_version(p: Path):
    # Capture only scikit-learn’s InconsistentVersionWarning for this load
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", InconsistentVersionWarning)
        model = joblib.load(p)
    trained_version = None
    for warn in w:
        msg = str(warn.message)
        # Extract the training version from messages like:
        # "Trying to unpickle estimator DecisionTreeClassifier from version 1.6.1 when using version 1.7.1."
        # Use proper regex escapes (no double backslashes inside a raw string)
        m = re.search(r"from version\s+(\d+\.\d+\.\d+)", msg)
        if m:
            trained_version = m.group(1)
            break
    return model, trained_version, [str(w.message) for w in w]

print(f"Runtime scikit-learn: {sklearn.__version__}")
print(f"Runtime joblib:       {joblib.__version__}")

results = {}
for key, fname in MODELS.items():
    p = find_model_path(fname)
    if not p:
        print(f"[{key}] NOT FOUND: {fname}")
        continue
    model, ver, warns = load_and_detect_version(p)
    print(f"[{key}] path={p}")
    print(f"[{key}] class={model.__class__.__name__}")
    print(f"[{key}] training version (from warning): {ver}")
    if not ver and warns:
        print(f"[{key}] warnings observed (first 2):")
        for line in warns[:2]:
            print("   -", line)
    results[key] = ver

bver, mver = results.get("binary"), results.get("multiclass")
if bver and mver:
    print(f"\nComparison: binary vs multiclass -> {bver} vs {mver} -> {'SAME' if bver == mver else 'DIFFERENT'}")
else:
    print("\nComparison: Could not determine both versions.")