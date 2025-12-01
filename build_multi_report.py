
import os, json, base64

RESULTS_DIR = "results"
SITE_DIR = "site"
os.makedirs(SITE_DIR, exist_ok=True)

multi_summary_path = os.path.join(RESULTS_DIR, "multi_summary.json")
if not os.path.exists(multi_summary_path):
    raise FileNotFoundError("results/multi_summary.json not found. Run simulate_multi.py first.")

data = json.load(open(multi_summary_path))

def b64img(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")

html = """<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Multi-Symbol Strategy Dashboard</title>
<style>
 body { font-family: Arial, sans-serif; margin:2rem; color:#222; }
 h1 { color:#0b5; }
 .card { border:1px solid #ddd; border-radius:8px; padding:1rem; margin-bottom:2rem; }
 img { max-width:100%; border:1px solid #ddd; border-radius:8px; }
 select { margin-bottom:1rem; padding:0.5rem; }
 table { border-collapse: collapse; width:100%; margin-top:1rem; }
 th, td { border:1px solid #ddd; padding:8px; text-align:left; }
 th { background:#f0f0f0; }
</style>
<script>
function showStrategy(symbol) {
    var select = document.getElementById(symbol + '-select');
    var chosen = select.value;
    var cards = document.querySelectorAll('.' + symbol + '-strategy');
    cards.forEach(card => {
        card.style.display = (card.dataset.strategy === chosen) ? 'block' : 'none';
    });
}
</script>
</head>
<body>
<h1>Multi-Symbol Strategy Dashboard</h1>
<p>Educational and simulation-only. Toggle strategies below each symbol.</p>
"""

for symbol, symdata in data.items():
    html += f"<div class='card'><h2>{symbol}</h2>"
    # Dropdown
    html += f"<label for='{symbol}-select'>Select Strategy:</label> "
    html += f"<select id='{symbol}-select' onchange='showStrategy(\"{symbol}\")'>"
    strategies = symdata.get("strategies", [])
    for strat in strategies:
        html += f"<option value='{strat}'>{strat}</option>"
    html += "</select>"

    # Strategy-specific sections
    for strat in strategies:
        img_path = os.path.join(RESULTS_DIR, f"{symbol}_{strat}_equity_curve.png")
        img_b64 = b64img(img_path)
        metrics = symdata.get(strat, {}).get("metrics", {})
        html += f"<div class='{symbol}-strategy' data-strategy='{strat}' style='display:none;'>"
        html += f"<h3>{strat.capitalize()} Strategy</h3>"
        if img_b64:
            html += f"<img src='data:image/png;base64,{img_b64}' alt='{strat} equity curve'/>"
        else:
            html += "<p><em>No equity curve available.</em></p>"
        # Metrics table
        if metrics:
            html += "<table><tbody>"
            for k,v in metrics.items():
                try:
                    v_rounded = f"{float(v):.3f}"
                except:
                    v_rounded = v
                html += f"<tr><th>{k}</th><td>{v_rounded}</td></tr>"
            html += "</tbody></table>"
        html += "</div>"
    html += "</div>"

html += "<script>document.querySelectorAll('select').forEach(sel => sel.dispatchEvent(new Event('change')));</script>"
html += "</body></html>"

with open(os.path.join(SITE_DIR, "index.html"), "w", encoding="utf-8") as f:
    f.write(html)

print("Dashboard with strategy dropdown generated → site/index.html")
