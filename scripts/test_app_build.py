"""Validate the Gradio app constructs (catches component/event wiring errors).

Does not launch a server or load the SAM model (segmenter is lazy).
"""

from samsplit.ui import build_app

demo = build_app()
print("app built OK:", type(demo).__name__, "| fns:", len(demo.fns))
