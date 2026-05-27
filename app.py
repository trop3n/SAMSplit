"""Launch the SAMSplit interactive UI.

    uv run app.py
"""

from samsplit.ui import build_app

if __name__ == "__main__":
    build_app().launch()
