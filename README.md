# Emotional Receipts
*A nation's playlist as its therapy bill.*

**DTU 02806 · Final Project · Spring 2026**

## Overview

Does what a country streams say anything about how its people feel? We took **170,614 chart entries** from Spotify's Daily Top-200 across **34 countries** (2017–2020) and lined them up against the **World Happiness Report 2019**. The result: happy countries do *not* listen to happy music — and that inversion is the story.

## Key Findings

- **Valence** (Spotify's "happiness" score) showed no meaningful correlation with national happiness (r = −0.16).
- **Acousticness** was the strongest *negative* predictor (r = −0.52).
- **Speechiness** was the strongest *positive* predictor (r = +0.47).
- Countries with high "coping ratios" (high energy, lower valence) — like Finland (#1 in WHR) — turned out to be the *happiest*, not the saddest.

## Dataset

| Metric | Value |
|---|---|
| Chart entries | 170,614 |
| Countries | 34 |
| Audio features | 21 |
| Time window | 2017–2020 |

## Project Structure

```
index.html                  # Main scrollytelling page
visualizations/
  v1_choropleth.html        # Global coping ratio map
  v2_scatter.html           # Valence × Energy scatter
  v4_genre_heatmap.html     # Genre over/under-indexing heatmap
  v5_profile.html           # Per-country audio radar
images/
  v3_correlations.svg       # Audio features vs WHR correlations
explainer_notebook.ipynb    # Full analysis notebook
country_features_full.csv   # Raw data download
```

## How to Run

Open `index.html` in a browser. All visualizations are embedded as local iframes — no server required.

## Course

DTU 02806 — Social Data Analysis and Visualization

## Group 21

| Name | Student ID |
|---|---|
| Rupak Parmeshwar Kadhare | s252684 |
| Sandesh Oli | s250126 |
| Matias Fernando Yuan | s253008 |
