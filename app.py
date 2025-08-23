from flask import Flask, render_template, request, send_file, redirect, url_for
import pickle
import pandas as pd
import numpy as np
import re

# ------------------------------
# دوال التنظيف
arabic_stopwords = {...}  
arabic_diacritics = re.compile(r'[\u0617-\u061A\u064B-\u0652]')

def clean_arabic_text(text: str) -> str:
    if pd.isna(text):
        return ''
    text = str(text).strip()
    text = re.sub(arabic_diacritics, '', text)
    text = re.sub(r'[إأآ]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ؤ', 'و', text)
    text = re.sub(r'ئ', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    text = text.replace('ـ', '')
    text = re.sub(r'[^ء-ي\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [t for t in text.split() if (t not in arabic_stopwords and len(t) > 1)]
    return ' '.join(tokens)

# ------------------------------
# تحميل الموديل والـ TF-IDF
with open(r"C:\Prediction_project_app\logistic_regression.pkl", "rb") as f:
    loaded_model = pickle.load(f)

with open(r"C:\Prediction_project_app\tfidf_vectorizer_subgroup.pkl", "rb") as f:
    loaded_tfidf = pickle.load(f)

cleaned_df = pd.read_csv(r"C:\Prediction_project_app\products_sample.csv")  

# ------------------------------
# دوال التنبؤ
def predict_topn(user_text, model, tfidf, cleaned_df, top_n=3):
    cleaned_text = clean_arabic_text(user_text).lower().strip()
    if not cleaned_text:
        return []
    text_tfidf = tfidf.transform([cleaned_text])
    probs = model.predict_proba(text_tfidf)[0]
    top_indices = np.argsort(probs)[::-1][:top_n]

    results = []
    for idx in top_indices:
        predicted_class = model.classes_[idx]
        prob = probs[idx]
        row = cleaned_df.loc[
            cleaned_df['department_bk'] == predicted_class,
            ['itemgroup_name_ar','itemclass_name_ar','section_name_ar','subgroup_name_ar']
        ].dropna().head(1)

        if not row.empty:
            group_name    = row['itemgroup_name_ar'].values[0]
            class_name    = row['itemclass_name_ar'].values[0]
            section_name  = row['section_name_ar'].values[0]
            subgroup_name = row['subgroup_name_ar'].values[0]
        else:
            group_name, class_name, section_name, subgroup_name = "غير معروف", "غير معروف", "غير معروف", "غير معروف"

        results.append({
            "group": group_name,
            "class": class_name,
            "section": section_name,
            "subgroup": subgroup_name,
            "probability": f"{prob*100:.1f}%"
        })
    return results


def predict_top1(user_text, model, tfidf, cleaned_df):
    cleaned_text = clean_arabic_text(user_text).lower().strip()
    if not cleaned_text:
        return {}
    text_tfidf = tfidf.transform([cleaned_text])
    probs = model.predict_proba(text_tfidf)[0]
    idx = np.argmax(probs)

    predicted_class = model.classes_[idx]
    prob = probs[idx]
    row = cleaned_df.loc[
        cleaned_df['department_bk'] == predicted_class,
        ['itemgroup_name_ar','itemclass_name_ar','section_name_ar','subgroup_name_ar']
    ].dropna().head(1)

    if not row.empty:
        group_name    = row['itemgroup_name_ar'].values[0]
        class_name    = row['itemclass_name_ar'].values[0]
        section_name  = row['section_name_ar'].values[0]
        subgroup_name = row['subgroup_name_ar'].values[0]
    else:
        group_name, class_name, section_name, subgroup_name = "غير معروف", "غير معروف", "غير معروف", "غير معروف"

    return {
        "input_text": user_text,
        "group": group_name,
        "class": class_name,
        "section": section_name,
        "subgroup": subgroup_name,
        "probability": f"{prob*100:.1f}%"
    }

# ------------------------------
# Flask App
app = Flask(__name__)
uploaded_results = None

# الصفحة الرئيسية
@app.route("/")
def home():
    return render_template("home.html")

# إدخال يدوي
@app.route("/manual", methods=["GET", "POST"])
def manual():
    results = None
    user_input = ""
    if request.method == "POST":
        user_input = request.form["user_text"]
        results = predict_topn(user_input, loaded_model, loaded_tfidf, cleaned_df, top_n=3)
    return render_template("index.html", results=results, user_input=user_input)

# رفع ملف
@app.route("/upload", methods=["GET", "POST"])
def upload():
    global uploaded_results
    results = None
    if request.method == "POST":
        file = request.files["file"]
        if file:
            if file.filename.endswith(".csv"):
                df = pd.read_csv(file)
            elif file.filename.endswith((".xls", ".xlsx")):
                df = pd.read_excel(file)
            else:
                return "❌ مسموح فقط بملفات CSV أو Excel"

            col_name = df.columns[0]
            results_list = []
            for val in df[col_name]:
                pred = predict_top1(val, loaded_model, loaded_tfidf, cleaned_df)
                results_list.append(pred)

            results = pd.DataFrame(results_list)
            uploaded_results = results
    return render_template("upload.html", results=results)

# تحميل النتائج
@app.route("/download")
def download():
    global uploaded_results
    if uploaded_results is None:
        return "❌ مفيش نتائج للتحميل"
    
    file_path = "results.xlsx"
    uploaded_results.to_excel(file_path, index=False)
    return send_file(file_path, as_attachment=True)

# ------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
