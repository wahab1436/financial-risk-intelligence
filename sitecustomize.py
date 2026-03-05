"""
sitecustomize.py — loaded automatically by Python before any user code.
Adds the repo root to sys.path so all subpackages are importable on Streamlit Cloud.
"""
import sys
import os

repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
