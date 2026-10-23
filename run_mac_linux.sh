#!/usr/bin/env bash
cd "$(dirname "$0")"
python3 -m pip install -r requirements.txt
cd backend && python3 app.py
