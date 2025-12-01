
# build_multi_report.py
import os, json

RESULTS_DIR = "results"
SITE_DIR = "site"
os.makedirs(SITE_DIR, exist_ok=True)

multi_summary_path = os.path.join(RESULTS_DIR, "multi_summary.json")
data = json.load(open(multi_summary_path))

html = """<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>Multi-Symbol Simulation Dashboard</title>
<style>
body { font-family: Arial; margin: 2rem; color:#222; }
h1 { color:#0b5; }
table { border-collapse: collapse; width: 100%; margin-bottom: 2rem; }
th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
th { background: #f0f0f0; }
.disclaimer { background:#fff7e6; border:1px solid #ffd48a; padding:1rem; border-radius:8px; margin-bottom:1rem; }
</style>
</head>
<body>
<h1>Multi-Symbol Simulation Dashboard</h1>
<div class="disclaimer">
<strong>Disclaimer:</strong> Educational and simulation-only. No brokerage, no live trades. Results are hypothetical.
</div>
<table>
<tr>
<th>Symbol</th>
<th>Total Return (%)</th>
<th>Sharpe</th>
<th>Max Drawdown (%)</th>
<th>Hit Rate</th>
<th>Profit Factor</th>
</tr>
"""

for symbol, summary in data.items():
    m = summary.get("mean_reversion_metrics", {})
    html += f"<tr><td>{symbol}</td><td>{m.get('total_return_pct', 0):.2f}</td><td>{m.get('daily_sharpe', 0):.2f}</td><td>{m.get('max_drawdown_pct', 0):.2f}</td><td>{m.get('hit_rate', 0):.2f}</td><td>{m.get('profit_factor', float('inf')):.2f}</td></tr>"

html += "</table><hr/><p>Generated automatically by GitHub Actions.</p></body></html>"

with open(os.path.join(SITE_DIR, "index.html"), "w", encoding="utf-8") as f:
    f.write(html)

print("Multi-symbol dashboard generated → site/index.html")
