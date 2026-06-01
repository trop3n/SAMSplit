  - Working tree is clean — the Phase 4 diff is just the 7 modified + 5 new files listed above. git add -A && git commit captures exactly the feature, nothing
  stray.
  - The app is still running at http://localhost:7860 (toggle live) so you can keep testing — kill it with pkill -f app.py when you're done.
  - A/B sheets are in outputs/ab_style_default.png and outputs/ab_style_big.png if you want another look before committing.

  One heads-up for your commit: the 6.4 MB LoRA in models/style_lora/ is gitignored (along with all model weights, ArtTraining/, and data/), so it won't be in the
  commit — it lives only on this machine. That's the conventional choice, but it means on another box you'd retrain (uv sync --extra train →
  prepare_style_dataset.py → train_style_lora.py, ~60 min) or copy the weights over. If you'd rather version it, git add -f
  models/style_lora/pytorch_lora_weights.safetensors.
