# 🌀 Rinnegan — AI-Powered Fake Review Detector

**Rinnegan** is a real-time, fully browser-based dashboard that analyzes product reviews and scores their authenticity using a 6-layer ML/NLP ensemble engine — no backend, no server, no data leaving your browser.

Paste in a review, hit analyze, and Rinnegan breaks down *why* it thinks the review is real, suspicious, or fake — with a live authenticity score, linguistic insights, spam-signal highlighting, and historical trend charts.

---

## ✨ Features

- **0–100 Authenticity Score** — every review is scored and classified as `REAL`, `SUSPICIOUS`, or `FAKE`
- **6-Layer Ensemble Engine** — combines rule-based heuristics, TF-IDF, Naive Bayes, sentiment analysis, linguistic scoring, and a neural network (TensorFlow.js) into a single weighted verdict
- **Live Dashboard** — quality score, honest review count, word-count trends, and animated stat counters
- **Signal Highlighting** — flags spam phrases, urgency language, excessive punctuation/caps, and generic vs. specific wording directly in the review text
- **Analysis History** — every analysis is saved locally with the ability to inspect, delete, or clear entries
- **Reports & Charts** — 7-day activity trend and REAL/SUSPICIOUS/FAKE breakdowns rendered with Chart.js, exportable as CSV
- **CSV Import** — upload a CSV of reviews and auto-load them into the analyzer
- **Sample Reviews** — one-click fake/real sample pills to demo the engine instantly
- **Light/Dark Theme** — theme preference persisted across sessions
- **Zero Backend** — runs entirely client-side with `localStorage` for persistence; deploy it as a static site anywhere

---

## 🧠 How the Ensemble Engine Works

Every submitted review is run through multiple independent scoring models, each producing its own "realness" signal. These are combined into a single weighted final score:

| Model | Weight (with Neural Net) | Weight (fallback) | What it looks at |
|---|---|---|---|
| Rule-Based Engine | 25% | 35% | Spam keywords, honest-language markers, caps ratio, exclamation density |
| TF-IDF Scoring | 20% | 25% | Term-frequency similarity against known real/fake review corpora |
| Naive Bayes Classifier | 20% | 25% | Probabilistic text classification |
| Sentiment Analysis | 10% | 15% | Sentiment polarity via `sentiment.js` — overly one-sided sentiment is a fake-review signal |
| Linguistic Scoring | 15% | — | Specificity, structure, and NLP-derived features via `compromise.js` |
| Neural Network | 10% | — | TensorFlow.js model trained on extracted text features |

> If the TensorFlow.js model fails to load, Rinnegan automatically falls back to a 4-model ensemble (Rule + TF-IDF + Naive Bayes + Sentiment) with re-balanced weights, so the app never breaks.

**Verdict thresholds:**
- `70–100` → ✅ **REAL**
- `40–69` → ⚠️ **SUSPICIOUS**
- `0–39` → ❌ **FAKE**

---

## 🖥️ Dashboard Preview

The dashboard is organized into five main sections, accessible from the sidebar:

1. **Dashboard** — live stats overview (honest review count, avg. quality score, word-count trend) with animated counters and charts
2. **Analyze Review** — the core analyzer: paste a review, watch the multi-step loading sequence (tokenizing → sentiment → linguistic → ensemble), and get a full score breakdown
3. **History** — a searchable/sortable table of every review you've analyzed, with delete/clear controls
4. **Reports** — aggregate stats (total/real/suspicious/fake counts) and a 7-day trend chart, exportable as CSV
5. **Settings** — theme, sensitivity slider, auto-analyze toggle, highlight toggle, and optional Gemini/OpenAI API key storage for future LLM-based analysis

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Markup / Styling | HTML5, CSS3 (custom properties, light/dark theming) |
| Fonts / Icons | Google Fonts (Inter), Font Awesome 6.4 |
| Charts | [Chart.js 4.4](https://www.chartjs.org/) |
| Machine Learning | [TensorFlow.js 4.10](https://www.tensorflow.org/js) |
| NLP | [Compromise 14.10](https://github.com/spencermountain/compromise) |
| Sentiment Analysis | [Sentiment.js 5.0](https://github.com/thisandagain/sentiment) |
| Persistence | Browser `localStorage` (no database, no server) |

All dependencies are loaded via CDN — there's no bundler, build step, or `npm install` required to run the app.

---

## 🚀 Getting Started

### Option 1 — Just open it
Since Rinnegan is 100% static, you can simply open `index.html` directly in your browser.

### Option 2 — Run a local server (recommended)
Some browsers restrict certain features when opening HTML files directly via `file://`. To avoid this, serve it locally:

```bash
git clone https://github.com/sujayski/Fake-Review-Detector-Using-AI.git
cd Fake-Review-Detector-Using-AI
npm start
```

`npm start` runs `npx serve .`, spinning up a static server at `http://localhost:3000` (or the next available port).

### Requirements
- Node.js (only needed for the local `serve` script — not required to run the app itself)
- A modern browser (Chrome, Edge, Firefox, Safari)

---

## 📂 Project Structure

```
Fake-Review-Detector-Using-AI/
├── index.html          # App shell, styles, and full application logic
├── css/
│   └── style.css       # Additional/extracted styles
├── js/
│   └── app.js          # Additional/extracted application logic
├── package.json         # Project metadata & npm start script
├── package-lock.json
├── LICENSE              # MIT License
└── README.md
```

> Note: the core dashboard currently ships as a single self-contained `index.html` (styles + logic inline) for zero-config portability. `css/style.css` and `js/app.js` are being used to progressively extract and modularize that logic.

---

## 📊 Usage

1. Open the app and navigate to **Analyze Review**
2. Paste a product review into the text box (or click one of the sample "fake"/"real" pills to try a demo)
3. Click **Analyze** — Rinnegan runs the full ensemble pipeline and displays:
   - Final authenticity score (0–100) and verdict badge
   - Per-model score breakdown
   - Highlighted spam/urgency/honest-language signals within the text
4. Check **History** to review past analyses, or **Reports** to see aggregate trends and export a CSV

---

## 🗺️ Roadmap

- [ ] Live Gemini / OpenAI LLM-based review analysis (API key fields are already scaffolded in Settings)
- [ ] Batch CSV analysis (multi-row processing instead of first-row only)
- [ ] Browser extension version for in-page review scanning on e-commerce sites
- [ ] Model retraining pipeline with a larger labeled dataset

---

## 🤝 Contributors

Done By:
1. Sujay H - 1CR25AD127
2. Swarnim Mishra - 1CR25AD128
3. Shravya - 1CR25CI055
4. Shubham Bhowmik - 1CR25CI056


---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](./LICENSE) for more information.

---

