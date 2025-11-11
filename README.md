# Media.net – MVP with Change #1 and #2
Changes:
1) Priority/Severity Fix: robust % deviation floors, optional persistence (2 intervals), and block anomalies so P1/P2 surface reliably.
2) Alerts tab: "Active Alerts" list first — shows Severity, Primary KPI, % deviation, Impact/hr, RCA, Recommendation — then detailed drilldown.

Run:
  python3 -m venv venv && source venv/bin/activate
  pip install -r requirements.txt
  streamlit run 0_📊_Dashboard.py
