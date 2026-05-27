"""Confirm the Gradio server actually starts and serves (no manual interaction)."""

import urllib.request

from samsplit.ui import build_app

demo = build_app()
demo.launch(prevent_thread_lock=True, server_port=7861, show_error=True)
url = demo.local_url
print("launched at", url)
code = urllib.request.urlopen(url, timeout=15).status
print("HTTP", code)
demo.close()
assert code == 200, f"server returned {code}"
print("launch OK")
