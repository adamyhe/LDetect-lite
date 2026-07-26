# Pipeline schematics

From a development checkout, the compact pipeline overview can be regenerated
with:

```bash
python3 schematics/plot_figure1_panels.py
```

The per-step pipeline schematics can be regenerated with:

```bash
uv run --extra heatmap python schematics/plot_pipeline_schematics.py
```
