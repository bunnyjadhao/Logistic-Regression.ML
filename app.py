import os
import pickle

import pandas as pd
from flask import Flask, render_template_string, request

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load model (logistic.pkl must sit in the same folder as app.py)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = "logistic.pkl"

with open(os.path.join(BASE_DIR, MODEL_FILE), "rb") as f:
    model = pickle.load(f)

# ---------------------------------------------------------------------------
# Dropdown options -> numeric codes the model was trained on.
# IMPORTANT: the model expects numbers for these four columns. The codes below
# assume alphabetical label encoding (0, 1, 2 ...). If you encoded them in a
# different order in your notebook, edit the numbers here to match.
# ---------------------------------------------------------------------------
MAJORS = [("Arts & Humanities", 0), ("Business", 1), ("Engineering", 2),
          ("Health Sciences", 3), ("Science", 4)]
USE_CASES = [("Brainstorming", 0), ("Coding", 1), ("Research", 2),
             ("Summarizing", 3), ("Writing", 4)]
POLICIES = [("Allowed", 0), ("Limited", 1), ("Restricted", 2)]
YES_NO = [("No", 0), ("Yes", 1)]

# Form sections: each field has a type (number or select)
SECTIONS = [
    {
        "title": "Student profile",
        "fields": [
            {"name": "Major_Category", "label": "Major", "type": "select",
             "options": MAJORS, "default": 2},
            {"name": "Year_of_Study", "label": "Year of study", "type": "number",
             "min": 1, "max": 6, "step": 1, "default": 2, "hint": "1 = first year"},
            {"name": "Pre_Semester_GPA", "label": "GPA before the semester", "type": "number",
             "min": 0, "max": 10, "step": 0.01, "default": 3.2,
             "hint": "Use the same scale as your training data"},
            {"name": "Institutional_Policy", "label": "College AI policy", "type": "select",
             "options": POLICIES, "default": 0},
        ],
    },
    {
        "title": "How they use AI",
        "fields": [
            {"name": "Weekly_GenAI_Hours", "label": "Weekly AI hours", "type": "number",
             "min": 0, "max": 80, "step": 0.5, "default": 10, "hint": "Hours per week with AI tools"},
            {"name": "Primary_Use_Case", "label": "Main use of AI", "type": "select",
             "options": USE_CASES, "default": 2},
            {"name": "Prompt_Engineering_Skill", "label": "Prompting skill", "type": "number",
             "min": 1, "max": 10, "step": 1, "default": 5, "hint": "Self rating, 1 to 10"},
            {"name": "Paid_Subscription", "label": "Paid AI subscription", "type": "select",
             "options": YES_NO, "default": 0},
        ],
    },
    {
        "title": "Study habits",
        "fields": [
            {"name": "Traditional_Study_Hours", "label": "Weekly study hours without AI", "type": "number",
             "min": 0, "max": 80, "step": 0.5, "default": 12, "hint": "Books, notes, lectures"},
            {"name": "Perceived_AI_Dependency", "label": "Dependency on AI", "type": "number",
             "min": 1, "max": 10, "step": 1, "default": 5, "hint": "Self rating, 1 to 10"},
            {"name": "Skill_Retention_Score", "label": "Skill retention score", "type": "number",
             "min": 0, "max": 100, "step": 1, "default": 60, "hint": "0 to 100"},
        ],
    },
]

ALL_FIELDS = [f for s in SECTIONS for f in s["fields"]]

PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI Impact on Students</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --night: #071a3d;
      --navy: #0d2f6e;
      --blue: #2563eb;
      --cyan: #22d3ee;
      --mint: #34e0a1;
      --amber: #ffc857;
      --coral: #ff7a90;
      --ink: #0f1f45;
      --muted: #5d6b8f;
      --line: #dbe5f7;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: "Outfit", system-ui, sans-serif;
      color: var(--ink);
      min-height: 100vh;
      padding: 40px 20px 30px;
      background:
        radial-gradient(700px 420px at 90% -5%, rgba(255, 200, 87, .45), transparent 60%),
        radial-gradient(800px 500px at 0% 10%, rgba(34, 211, 238, .45), transparent 60%),
        linear-gradient(145deg, #071a3d 0%, #0d2f6e 40%, #2563eb 75%, #22d3ee 120%);
      background-attachment: fixed;
    }
    .wrap { max-width: 1120px; margin: 0 auto; }

    header { color: #fff; margin-bottom: 30px; max-width: 720px; }
    header h1 { font-size: clamp(2rem, 4.6vw, 3.2rem); font-weight: 800; letter-spacing: -.02em; line-height: 1.1; }
    header p { margin-top: 12px; font-size: 1.08rem; line-height: 1.55; color: rgba(255,255,255,.85); }

    .grid { display: grid; grid-template-columns: 1.5fr 1fr; gap: 24px; align-items: start; }
    @media (max-width: 880px) { .grid { grid-template-columns: 1fr; } }

    form.panel {
      background: rgba(255,255,255,.95);
      border-radius: 24px;
      padding: 12px 30px 30px;
      box-shadow: 0 24px 60px rgba(4, 14, 40, .45);
    }
    fieldset { border: 0; padding: 22px 0 6px; border-bottom: 1px solid var(--line); }
    fieldset:last-of-type { border-bottom: 0; }
    legend {
      font-size: 1.1rem; font-weight: 700; padding: 0 0 14px;
      background: linear-gradient(90deg, var(--blue), var(--cyan));
      -webkit-background-clip: text; background-clip: text; color: transparent;
    }
    .fields { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding-bottom: 14px; }
    @media (max-width: 560px) { .fields { grid-template-columns: 1fr; } }
    label { display: block; font-size: .92rem; font-weight: 600; margin-bottom: 6px; }
    input, select {
      width: 100%; padding: 11px 13px; font: inherit; color: var(--ink);
      background: #f5f8ff; border: 1.5px solid var(--line); border-radius: 12px;
      transition: border-color .15s, box-shadow .15s;
    }
    input:focus, select:focus {
      outline: none; border-color: var(--blue); box-shadow: 0 0 0 4px rgba(37, 99, 235, .2);
    }
    .hint { font-size: .78rem; color: var(--muted); margin-top: 5px; }

    button {
      margin-top: 22px; width: 100%; padding: 15px 20px; font: inherit;
      font-weight: 700; font-size: 1.05rem; color: #06214f; border: 0; border-radius: 14px; cursor: pointer;
      background: linear-gradient(90deg, var(--mint), var(--cyan) 50%, var(--amber));
      box-shadow: 0 10px 26px rgba(34, 211, 238, .4);
      transition: transform .12s, box-shadow .12s;
    }
    button:hover { transform: translateY(-2px); box-shadow: 0 14px 32px rgba(34, 211, 238, .55); }
    button:focus-visible { outline: 3px solid var(--amber); outline-offset: 3px; }

    .result {
      position: sticky; top: 24px; color: #fff; border-radius: 24px; padding: 30px;
      background: linear-gradient(165deg, #06122e 0%, #0d2f6e 60%, #1d4ed8 100%);
      box-shadow: 0 24px 60px rgba(4, 14, 40, .5); overflow: hidden;
    }
    .result::after {
      content: ""; position: absolute; right: -80px; top: -80px; width: 240px; height: 240px;
      border-radius: 50%; background: radial-gradient(circle, rgba(255,200,87,.4), transparent 70%);
    }
    .result > * { position: relative; z-index: 1; }
    .result small { display: block; color: rgba(255,255,255,.7); font-size: .95rem; }
    .level {
      margin-top: 8px; font-size: clamp(2.4rem, 5vw, 3.4rem); font-weight: 800; letter-spacing: -.02em;
      background: linear-gradient(90deg, var(--cyan), #fff 60%, var(--amber));
      -webkit-background-clip: text; background-clip: text; color: transparent;
    }
    .level.Low    { background: linear-gradient(90deg, var(--mint), #fff);  -webkit-background-clip: text; background-clip: text; color: transparent; }
    .level.Medium { background: linear-gradient(90deg, var(--amber), #fff); -webkit-background-clip: text; background-clip: text; color: transparent; }
    .level.High   { background: linear-gradient(90deg, var(--coral), #fff); -webkit-background-clip: text; background-clip: text; color: transparent; }

    .bars { margin-top: 22px; display: grid; gap: 14px; }
    .bar-top { display: flex; justify-content: space-between; font-size: .9rem; margin-bottom: 6px; color: rgba(255,255,255,.9); }
    .track { height: 10px; border-radius: 99px; background: rgba(255,255,255,.14); overflow: hidden; }
    .fill { height: 100%; border-radius: 99px; }
    .fill.Low    { background: linear-gradient(90deg, var(--mint), var(--cyan)); }
    .fill.Medium { background: linear-gradient(90deg, var(--amber), #ffe29a); }
    .fill.High   { background: linear-gradient(90deg, var(--coral), #ffb3c0); }

    .note { margin-top: 20px; font-size: .86rem; line-height: 1.55; color: rgba(255,255,255,.72); }
    .error { margin-top: 12px; padding: 12px 14px; border-radius: 12px; background: rgba(255,122,144,.2); color: #ffd9df; font-size: .92rem; }
    .empty { line-height: 1.6; color: rgba(255,255,255,.85); }

    footer { text-align: center; color: rgba(255,255,255,.75); font-size: .85rem; margin-top: 28px; }
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>AI Impact on Students</h1>
      <p>Describe a student's profile and how they use AI tools. The model estimates the impact level as Low, Medium or High.</p>
    </header>

    <div class="grid">
      <form class="panel" method="POST" action="/">
        {% for section in sections %}
        <fieldset>
          <legend>{{ section.title }}</legend>
          <div class="fields">
            {% for f in section.fields %}
            <div>
              <label for="{{ f.name }}">{{ f.label }}</label>
              {% if f.type == "select" %}
              <select id="{{ f.name }}" name="{{ f.name }}">
                {% for text, code in f.options %}
                <option value="{{ code }}" {% if values[f.name]|string == code|string %}selected{% endif %}>{{ text }}</option>
                {% endfor %}
              </select>
              {% else %}
              <input type="number" id="{{ f.name }}" name="{{ f.name }}"
                     min="{{ f.min }}" max="{{ f.max }}" step="{{ f.step }}"
                     value="{{ values[f.name] }}" required>
              {% endif %}
              {% if f.hint %}<div class="hint">{{ f.hint }}</div>{% endif %}
            </div>
            {% endfor %}
          </div>
        </fieldset>
        {% endfor %}
        <button type="submit">Predict impact level</button>
      </form>

      <aside class="result" aria-live="polite">
        {% if error %}
          <small>Predicted impact level</small>
          <div class="error">{{ error }}</div>
        {% elif prediction %}
          <small>Predicted impact level</small>
          <div class="level {{ prediction }}">{{ prediction }}</div>
          <div class="bars">
            {% for p in probabilities %}
            <div>
              <div class="bar-top"><span>{{ p.label }}</span><span>{{ p.percent }}%</span></div>
              <div class="track"><div class="fill {{ p.label }}" style="width: {{ p.percent }}%"></div></div>
            </div>
            {% endfor %}
          </div>
          <p class="note">The bars show how confident the model is in each level. Change any value and predict again to compare.</p>
        {% else %}
          <div class="empty">Fill in the student's details and press <b>Predict impact level</b>. The result and the model's confidence for each level will appear here.</div>
        {% endif %}
      </aside>
    </div>

    <footer>Built with Flask and scikit-learn</footer>
  </div>
</body>
</html>
"""


def default_values():
    return {f["name"]: f["default"] for f in ALL_FIELDS}


@app.route("/", methods=["GET", "POST"])
def home():
    values = default_values()
    prediction = None
    probabilities = []
    error = None

    if request.method == "POST":
        try:
            row = {}
            for f in ALL_FIELDS:
                raw = request.form.get(f["name"], "").strip()
                values[f["name"]] = raw
                row[f["name"]] = float(raw)

            # Column order must match the order the model was trained on
            df = pd.DataFrame([row])[list(model.feature_names_in_)]
            prediction = str(model.predict(df)[0])
            proba = model.predict_proba(df)[0]

            order = {"Low": 0, "Medium": 1, "High": 2}
            probabilities = sorted(
                [{"label": str(c), "percent": round(float(p) * 100, 1)}
                 for c, p in zip(model.classes_, proba)],
                key=lambda x: order.get(x["label"], 99),
            )
        except ValueError:
            error = "Please enter a valid number in every field."
        except Exception as exc:  # keep the page alive on any model error
            error = f"Could not predict: {exc}"

    return render_template_string(
        PAGE, sections=SECTIONS, values=values,
        prediction=prediction, probabilities=probabilities, error=error,
    )


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
