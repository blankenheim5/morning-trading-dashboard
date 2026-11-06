
name: Technical Dashboard

on:
  workflow_dispatch:
  schedule:
    # GitHub schedules use UTC.
    # Both times cover Central Standard and Daylight Time.
    - cron: '40 14 * * 1-5'
    - cron: '40 15 * * 1-5'

permissions:
  contents: read

concurrency:
  group: technical-dashboard-pages
  cancel-in-progress: false

jobs:
  test-calculator:
    runs-on: ubuntu-latest

    permissions:
      contents: read
      pages: write
      id-token: write

    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}

    steps:
      - name: Check Central Time schedule
        id: timing
        env:
          EVENT_NAME: ${{ github.event_name }}
        run: |
          python - <<'PY'
          import os
          import datetime as dt
          from zoneinfo import ZoneInfo

          now = dt.datetime.now(ZoneInfo("America/Chicago"))
          manual = os.environ["EVENT_NAME"] == "workflow_dispatch"

          # Allow delayed scheduled runs during the same
          # trading morning, but reject the wrong UTC slot.
          allowed = manual or (
              now.weekday() < 5
              and now.hour == 8
              and now.minute >= 40
          )

          with open(os.environ["GITHUB_OUTPUT"], "a") as f:
              f.write(f"run={'true' if allowed else 'false'}\n")

          print("Central time:", now.isoformat())
          print("Run calculator:", allowed)
          PY

      - name: Check out repository
        if: steps.timing.outputs.run == 'true'
        uses: actions/checkout@v4

      - name: Set up Python
        if: steps.timing.outputs.run == 'true'
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        if: steps.timing.outputs.run == 'true'
        run: |
          python -m pip install --upgrade pip
          pip install pandas numpy yfinance requests pandas_market_calendars

      - name: Run technical dashboard calculator
        if: steps.timing.outputs.run == 'true'
        run: |
          mkdir -p "${RUNNER_TEMP}/dashboard"
          python technical_dashboard.py --current-open --output "${RUNNER_TEMP}/dashboard/dashboard.csv"

      - name: Verify output and prepare website
        if: steps.timing.outputs.run == 'true'
        env:
          DASHBOARD_DIR: ${{ runner.temp }}/dashboard
        run: |
          python - <<'PY'
          import json
          import os
          import pathlib
          import datetime as dt
          from zoneinfo import ZoneInfo

          folder = pathlib.Path(os.environ["DASHBOARD_DIR"])
          source = folder / "dashboard.json"
          assert source.is_file(), "Missing dashboard.json"

          data = json.loads(source.read_text())
          required = {
              "SCHG", "URA", "COPX", "PDBC",
              "CEG", "VST", "CIBR", "SMH", "SPY"
          }

          assert data.get("status") == "validated"
          rows = data.get("tickers")
          assert isinstance(rows, list) and len(rows) == 9

          names = {r.get("ticker") for r in rows}
          assert names == required, "Missing or duplicate tickers"

          checks = data.get("validation", {})
          assert checks.get("minimum_sessions", 0) >= 220

          asof = dt.date.fromisoformat(data["asof"])
          today = dt.datetime.now(
              ZoneInfo("America/New_York")
          ).date()

          # During a weekday morning, the latest completed
          # session should be before the current trading day.
          assert asof < today, "Unexpected completed-session date"
          assert (today - asof).days <= 4, "Stale market data"

          site = folder / "site"
          site.mkdir(exist_ok=True)
          (site / "dashboard.json").write_bytes(source.read_bytes())

          html = """<!doctype html>
          <html lang="en">
          <head><meta charset="utf-8">
          <title>Technical Dashboard</title></head>
          <body>
          <h1>Technical Dashboard</h1>
          <p>Validated market data:</p>
          <a href="dashboard.json">View dashboard.json</a>
          </body></html>"""
          (site / "index.html").write_text(html)

          print("PUBLICATION READY:", len(rows),
                "tickers, asof", asof)
          PY

      - name: Save diagnostic results
        if: always() && steps.timing.outputs.run == 'true'
        uses: actions/upload-artifact@v4
        with:
          name: technical-dashboard-diagnostics
          path: ${{ runner.temp }}/dashboard/dashboard.csv.errors.json
          if-no-files-found: warn

      - name: Configure GitHub Pages
        if: steps.timing.outputs.run == 'true'
        uses: actions/configure-pages@v5

      - name: Upload validated website
        if: steps.timing.outputs.run == 'true'
        uses: actions/upload-pages-artifact@v3
        with:
          path: ${{ runner.temp }}/dashboard/site

      - name: Publish validated website
        if: steps.timing.outputs.run == 'true'
        id: deployment
        uses: actions/deploy-pages@v4
