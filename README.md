# Ulta Beauty Dashboard

[![Website](https://img.shields.io/badge/Live%20Demo-Click%20Here-blue)](https://ulta-dash-app-v2.wl.r.appspot.com)

##  Project Summary

This repository hosts a data analytics dashboard for **Ulta Beauty products**, visualizing product trends, review patterns, and pricing insights. The dashboard compares historical and current product data scraped from Ulta's website and highlights new product launches, shifts in customer reviews, and category-level trends. Built with **Dash + Plotly**, the app serves as an interactive platform for exploring Ulta's evolving product catalog.

---

##  Setup Instructions

### Requirements:

* Python 3.10+
* Recommended to use a virtual environment:

  ```bash
  python -m venv venv
  source venv/bin/activate  # or venv\Scripts\activate on Windows
  ```

### Install Dependencies:

```bash
pip install -r requirements.txt
```

### Run Locally:

```bash
python app.py
```

### Deployment:

Deployed on Google Cloud App Engine.

```bash
gcloud app deploy
```

---

## Project Pipeline Overview

###  1. Data Collection:

* **Source:** Ulta Beauty website.
* Historical & recent product data is scraped and stored in **Google Cloud Storage (GCS)**.
* CSVs: `cleaned_makeup_products.csv` (8 months ago from kaggle), `face_df.csv` (current-scraped).

###  2. Data Processing:

* Feature engineering (new brands, new products, review metrics).
* Computed category-level and brand-level trends.

###  3. Dashboard Visualization:

* Built using **Dash** and **Plotly**.
* Interactive charts: Bar charts, scatter plots, violin plots.
* Focus on product growth, reviews, and pricing trends.

###  4. Deployment:

* Hosted on **Google Cloud App Engine**.
* Live at: [ulta-dash-app-v2.wl.r.appspot.com](https://ulta-dash-app-v2.wl.r.appspot.com).

---

## 📁 Repository Structure

| Folder/File        | Purpose                                                              |
| ------------------ | -------------------------------------------------------------------- |
| `app.py`           | Main entry point for Dash app. Initializes pages & server.           |
| `app.yaml`         | GCP App Engine configuration (runtime, scaling).                     |
| `fetch.py`         | Loads data from Google Cloud Storage buckets (lazy loading).         |
| `pages/`           | Contains individual dashboard pages (`home.py`, `methods.py`, etc.). |
| `data/`            | (Optional local CSVs for testing, real data is fetched from GCS).    |
| `assets/`          | Static assets like CSS, images (e.g., background visuals).           |
| `requirements.txt` | All required Python packages for local & cloud deployment.           |
| `.gcloudignore`    | Files/folders ignored during GCP deploy (like .gitignore).           |

---

##  Live Website

 [https://ulta-dash-app-v2.wl.r.appspot.com](https://ulta-dash-app-v2.wl.r.appspot.com)

