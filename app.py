"""Aplikasi web Flask: prediksi churn pelanggan telekomunikasi dengan Decision Tree (TA-05).

Versi ringan untuk deployment (mis. Vercel): pohon keputusan hasil pelatihan scikit-learn
diekspor ke model/tree.json dan ditelusuri dengan Python murni, sehingga scikit-learn,
pandas, dan SciPy tidak diperlukan saat aplikasi berjalan. Model aslinya tetap tersimpan
dengan Joblib (model/decision_tree.joblib) dan dilatih lewat notebook.
"""
import json
import os

from flask import Flask, render_template, request

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

YESNO = ["Yes", "No"]
ADDON = ["Yes", "No", "No internet service"]
# (nama kolom, label, tipe, pilihan)
FIELDS = [
    ("gender", "Jenis kelamin", "select", ["Female", "Male"]),
    ("SeniorCitizen", "Lansia (65+)", "select", ["0", "1"]),
    ("Partner", "Punya pasangan", "select", YESNO),
    ("Dependents", "Punya tanggungan", "select", YESNO),
    ("tenure", "Lama berlangganan (bulan)", "number", None),
    ("PhoneService", "Layanan telepon", "select", YESNO),
    ("MultipleLines", "Banyak saluran telepon", "select", ["Yes", "No", "No phone service"]),
    ("InternetService", "Layanan internet", "select", ["DSL", "Fiber optic", "No"]),
    ("OnlineSecurity", "Keamanan online", "select", ADDON),
    ("OnlineBackup", "Cadangan online", "select", ADDON),
    ("DeviceProtection", "Proteksi perangkat", "select", ADDON),
    ("TechSupport", "Dukungan teknis", "select", ADDON),
    ("StreamingTV", "Streaming TV", "select", ADDON),
    ("StreamingMovies", "Streaming film", "select", ADDON),
    ("Contract", "Jenis kontrak", "select", ["Month-to-month", "One year", "Two year"]),
    ("PaperlessBilling", "Tagihan tanpa kertas", "select", YESNO),
    ("PaymentMethod", "Metode pembayaran", "select",
     ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]),
    ("MonthlyCharges", "Tagihan bulanan (USD)", "number", None),
    ("TotalCharges", "Total tagihan (USD)", "number", None),
]
NUMERIC = {"tenure", "MonthlyCharges", "TotalCharges"}
BINARY = {"gender": "Male", "Partner": "Yes", "Dependents": "Yes", "PhoneService": "Yes", "PaperlessBilling": "Yes"}
INTERNET_ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
LABELS = {f[0]: f[1] for f in FIELDS}
EXAMPLES = {
    "loyal": {"gender": "Female", "SeniorCitizen": "0", "Partner": "Yes", "Dependents": "Yes", "tenure": "60",
              "PhoneService": "Yes", "MultipleLines": "Yes", "InternetService": "DSL", "OnlineSecurity": "Yes",
              "OnlineBackup": "Yes", "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
              "StreamingMovies": "No", "Contract": "Two year", "PaperlessBilling": "No",
              "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": "65", "TotalCharges": "3900"},
    "risk": {"gender": "Male", "SeniorCitizen": "0", "Partner": "No", "Dependents": "No", "tenure": "3",
             "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic", "OnlineSecurity": "No",
             "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
             "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
             "PaymentMethod": "Electronic check", "MonthlyCharges": "75", "TotalCharges": "225"},
}
_state = {}


def _load():
    if _state:
        return _state
    with open(os.path.join(BASE, "model", "tree.json")) as f:
        t = json.load(f)
    with open(os.path.join(BASE, "model", "metrics.json")) as f:
        m = json.load(f)
    _state.update(features=t["features"], tree=t["tree"], metrics=m)
    return _state


def encode(form):
    """Ubah isian formulir menjadi dict fitur sesuai pra-pemrosesan pada notebook."""
    v = {}
    for name, _, kind, _ in FIELDS:
        raw = form[name]
        if kind == "number":
            v[name] = float(raw)
        elif name == "SeniorCitizen":
            v[name] = float(raw)
        elif name in BINARY:
            v[name] = 1.0 if raw == BINARY[name] else 0.0
        else:  # kategorikal -> one-hot
            for feat in _load()["features"]:
                if feat.startswith(name + "_"):
                    v[feat] = 1.0 if feat == f"{name}_{raw}" else 0.0
    return v


def explain(feat, went_left, thr):
    """Terjemahkan satu simpul keputusan menjadi kalimat yang mudah dibaca."""
    for col, label in LABELS.items():
        if feat.startswith(col + "_"):
            val = feat[len(col) + 1:]
            return f"{label} = {val}" if not went_left else f"{label} bukan {val}"
    if feat in NUMERIC:
        return f"{LABELS[feat]} {'≤' if went_left else '>'} {thr:.1f}"
    return f"{LABELS.get(feat, feat)} {'≤' if went_left else '>'} {thr:.1f}"


def predict(vec):
    s = _load()
    t, feats = s["tree"], s["features"]
    node, path = 0, []
    while t["left"][node] != -1:
        f = feats[t["feature"][node]]
        thr = t["threshold"][node]
        left = vec.get(f, 0.0) <= thr
        path.append(explain(f, left, thr))
        node = t["left"][node] if left else t["right"][node]
    val = t["value"][node]
    p = val[1] / (val[0] + val[1])
    return {"proba": p, "label": 1 if p >= 0.5 else 0, "path": path, "n": int(t["n_samples"][node])}


def parse(form):
    """Validasi masukan; kembalikan (nilai_bersih, daftar_error)."""
    s, errors, clean = _load(), [], {}
    for name, label, kind, opts in FIELDS:
        raw = (form.get(name) or "").strip()
        if kind == "number":
            try:
                x = float(raw)
            except ValueError:
                errors.append(f"{label}: harus berupa angka.")
                continue
            lo, hi = s["metrics"]["ranges"][name]
            if name == "TotalCharges":
                lo = 0.0  # pelanggan baru (tenure 0) memiliki total tagihan 0
            if not (lo <= x <= hi):
                errors.append(f"{label}: harus antara {lo:g} dan {hi:g} (rentang data latih).")
                continue
            clean[name] = raw
        else:
            if raw not in opts:
                errors.append(f"{label}: pilihan tidak valid.")
                continue
            clean[name] = raw
    if not errors:  # konsistensi layanan
        if clean["InternetService"] == "No":
            for c in INTERNET_ADDONS:
                clean[c] = "No internet service"
        else:
            for c in INTERNET_ADDONS:
                if clean[c] == "No internet service":
                    clean[c] = "No"
        if clean["PhoneService"] == "No":
            clean["MultipleLines"] = "No phone service"
        elif clean["MultipleLines"] == "No phone service":
            clean["MultipleLines"] = "No"
    return clean, errors


@app.route("/", methods=["GET", "POST"])
def index():
    s = _load()
    values, errors, result = dict(EXAMPLES["loyal"]), [], None
    if request.method == "POST":
        values = {k: request.form.get(k, "") for k, *_ in FIELDS}
        clean, errors = parse(request.form)
        if not errors:
            values = clean
            result = predict(encode(clean))
    imp = sorted(s["metrics"]["importance"].items(), key=lambda kv: -kv[1])
    imp = [(k, v) for k, v in imp if v > 0]
    return render_template("index.html", fields=FIELDS, values=values, errors=errors, result=result,
                           metrics=s["metrics"], importance=imp, examples=EXAMPLES)


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
