#!/bin/bash
python -m venv antenv
source antenv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run PSP_GPT_v5.py --server.port=$PORT --server.enableCORS=false
