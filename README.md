# StarWhisper

Astronomical Detection & Classification Workstation

## Overview

StarWhisper is a production-ready platform for exoplanet detection and classification from TESS light curves. It provides:

- FITS/CSV upload and automatic processing
- Light curve detrending, sigma clipping, and BLS transit search
- Machine learning classification with XGBoost and explainable AI (SHAP)
- Scientific sonification of light curves
- Publication-quality PDF reports
- Interactive visualization with Plotly
- Full validation metrics

## Quick Start

1. Clone the repository.
2. Copy `.env.example` to `.env` and adjust if needed.
3. Run:
   ```bash
   docker-compose up -d
