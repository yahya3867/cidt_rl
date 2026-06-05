from __future__ import annotations

import argparse
import json
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from src.incident_agent import generate_incident_report
from src.knowledge_db import initialize_database


class OpsHandler(BaseHTTPRequestHandler):
    db_path: Path
    agent_model: str
    slm_base_url: str

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        if parsed.path == "/":
            self._send_html(_dashboard_html())
        elif parsed.path == "/api/health":
            self._send_json({"ok": True, "db_path": str(self.db_path), "agent_model": self.agent_model})
        elif parsed.path == "/api/stats":
            self._send_json(_stats(self.db_path))
        elif parsed.path == "/api/anomalies":
            self._send_json(_anomalies(self.db_path, limit=_int_param(params, "limit", 25)))
        elif parsed.path == "/api/log-entries":
            self._send_json(_log_entries(self.db_path, limit=_int_param(params, "limit", 50)))
        elif parsed.path == "/api/observations":
            self._send_json(_observations(self.db_path, limit=_int_param(params, "limit", 25)))
        elif parsed.path == "/api/incidents":
            self._send_json(_incident_reports(self.db_path, limit=_int_param(params, "limit", 25)))
        elif parsed.path == "/api/feedback":
            self._send_json(_feedback_rows(self.db_path, limit=_int_param(params, "limit", 50)))
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        body = self._read_json()
        try:
            if parsed.path == "/api/incidents/generate":
                instance = str(body.get("instance", "")).strip() or None
                log_entry_id = body.get("log_entry_id")
                log_entry_id = int(log_entry_id) if log_entry_id not in (None, "") else None
                if not instance and log_entry_id is None:
                    self._send_json({"error": "instance or log_entry_id is required"}, status=400)
                    return
                report = generate_incident_report(
                    self.db_path,
                    instance,
                    log_entry_id=log_entry_id,
                    agent_model=self.agent_model,
                    base_url=self.slm_base_url,
                )
                self._send_json(report)
            elif parsed.path == "/api/feedback":
                self._send_json(_insert_feedback(self.db_path, body))
            else:
                self._send_json({"error": "not found"}, status=404)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _read_json(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_json(self, payload: object, *, status: int = 200) -> None:
        data = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self, html: str) -> None:
        data = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve CIDT anomaly operations API and dashboard.")
    parser.add_argument("--db-path", default="results/knowledge/cidt_anomaly_knowledge.db")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--agent-model", default="google/gemma-3-4b")
    parser.add_argument("--slm-base-url", default="http://127.0.0.1:1234/v1")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    initialize_database(db_path)
    OpsHandler.db_path = db_path
    OpsHandler.agent_model = args.agent_model
    OpsHandler.slm_base_url = args.slm_base_url

    server = ThreadingHTTPServer((args.host, args.port), OpsHandler)
    print(f"CIDT ops server running at http://{args.host}:{args.port}")
    print(f"Using DB: {db_path.resolve()}")
    print(f"Incident agent model: {args.agent_model}")
    server.serve_forever()


def _latest_run_id(conn: sqlite3.Connection) -> int | None:
    row = conn.execute("SELECT id FROM model_runs ORDER BY id DESC LIMIT 1").fetchone()
    return int(row[0]) if row else None


def _best_model(conn: sqlite3.Connection, run_id: int | None) -> str:
    row = conn.execute(
        "SELECT model FROM model_scores WHERE run_id = ? ORDER BY host_mean_score_auc DESC, f1 DESC LIMIT 1",
        (run_id,),
    ).fetchone()
    return str(row[0]) if row else ""


def _anomalies(db_path: Path, *, limit: int) -> list[dict[str, object]]:
    conn = _connect(db_path)
    try:
        run_id = _latest_run_id(conn)
        model = _best_model(conn, run_id)
        rows = conn.execute(
            """
            SELECT hs.*, h.rack, h.note, h.assigned_user
            FROM host_scores hs
            LEFT JOIN hosts h ON h.instance = hs.instance
            WHERE hs.run_id = ? AND hs.model = ?
            ORDER BY hs.alert_host DESC, hs.mean_likelihood DESC
            LIMIT ?
            """,
            (run_id, model, limit),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def _observations(db_path: Path, *, limit: int) -> list[dict[str, object]]:
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT * FROM observations
            ORDER BY severity = 'critical' DESC, confidence DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def _log_entries(db_path: Path, *, limit: int) -> list[dict[str, object]]:
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT id, run_id, entry_type, timestamp, instance, severity, source, message, created_at
            FROM log_entries
            ORDER BY
                CASE severity WHEN 'critical' THEN 0 WHEN 'high' THEN 1 WHEN 'warning' THEN 2 ELSE 3 END,
                id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def _incident_reports(db_path: Path, *, limit: int) -> list[dict[str, object]]:
    conn = _connect(db_path)
    try:
        rows = conn.execute("SELECT * FROM incident_reports ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def _feedback_rows(db_path: Path, *, limit: int) -> list[dict[str, object]]:
    conn = _connect(db_path)
    try:
        rows = conn.execute("SELECT * FROM feedback ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def _stats(db_path: Path) -> dict[str, object]:
    conn = _connect(db_path)
    try:
        run_id = _latest_run_id(conn)
        severity_rows = conn.execute(
            "SELECT severity, COUNT(*) AS count FROM log_entries GROUP BY severity"
        ).fetchall()
        entry_type_rows = conn.execute(
            "SELECT entry_type, COUNT(*) AS count FROM log_entries GROUP BY entry_type"
        ).fetchall()
        latest_entry = conn.execute(
            "SELECT timestamp, created_at FROM log_entries ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return {
            "run_id": run_id,
            "best_model": _best_model(conn, run_id),
            "log_entries": _count(conn, "log_entries"),
            "observations": _count(conn, "observations"),
            "telemetry_points": _count(conn, "telemetry_points"),
            "incident_reports": _count(conn, "incident_reports"),
            "feedback": _count(conn, "feedback"),
            "entry_metadata": _count_where(conn, "observations", "observation_type = 'entry_metadata'"),
            "entry_model_outputs": _count(conn, "entry_model_outputs"),
            "entry_slm_extractions": _count(conn, "entry_slm_extractions"),
            "severity_counts": {str(row["severity"] or "info"): int(row["count"]) for row in severity_rows},
            "entry_type_counts": {str(row["entry_type"] or "unknown"): int(row["count"]) for row in entry_type_rows},
            "latest_entry_timestamp": str(latest_entry["timestamp"] or latest_entry["created_at"]) if latest_entry else "",
        }
    finally:
        conn.close()


def _count(conn: sqlite3.Connection, table: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return int(row[0]) if row else 0


def _count_where(conn: sqlite3.Connection, table: str, where_clause: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {where_clause}").fetchone()
    return int(row[0]) if row else 0


def _insert_feedback(db_path: Path, body: dict[str, object]) -> dict[str, object]:
    instance = str(body.get("instance", "")).strip()
    log_entry_id = body.get("log_entry_id")
    log_entry_id = int(log_entry_id) if log_entry_id not in (None, "") else None
    if not instance and log_entry_id is not None:
        instance = _instance_for_log_entry(db_path, log_entry_id)
    feedback_type = str(body.get("feedback_type", "")).strip()
    if not instance or not feedback_type:
        raise ValueError("instance and feedback_type are required")
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(
            "INSERT INTO feedback(log_entry_id, instance, model, feedback_type, note, user) VALUES (?, ?, ?, ?, ?, ?)",
            (
                log_entry_id,
                instance,
                str(body.get("model", "")),
                feedback_type,
                str(body.get("note", "")),
                str(body.get("user", "")),
            ),
        )
        conn.commit()
        return {"id": int(cursor.lastrowid), "ok": True}
    finally:
        conn.close()


def _instance_for_log_entry(db_path: Path, log_entry_id: int) -> str:
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT instance FROM log_entries WHERE id = ?", (log_entry_id,)).fetchone()
        return str(row["instance"]) if row else ""
    finally:
        conn.close()


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _int_param(params: dict[str, list[str]], name: str, default: int) -> int:
    try:
        return int(params.get(name, [str(default)])[0])
    except ValueError:
        return default


def _dashboard_html() -> str:
    return """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>CIDT Ops</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {
      color-scheme: dark;
      --bg: #0b0d10;
      --surface: #11151a;
      --panel: #171c23;
      --panel-2: #1f2630;
      --line: #2e3744;
      --line-strong: #455160;
      --text: #f2f5f8;
      --muted: #98a5b5;
      --soft: #c7d0db;
      --accent: #62d0a7;
      --accent-2: #8fb8ff;
      --warn: #f3bd63;
      --danger: #ff7b7b;
      --info: #7db7ff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "Segoe UI", Arial, sans-serif;
      letter-spacing: 0;
    }
    header {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 18px;
      padding: 16px 22px;
      border-bottom: 1px solid var(--line);
      background: #0e1217;
      position: sticky;
      top: 0;
      z-index: 2;
    }
    h1, h2, h3, p { margin: 0; }
    h1 { font-size: 19px; font-weight: 750; }
    h2 { font-size: 14px; font-weight: 750; margin-bottom: 4px; }
    h3 { font-size: 13px; margin: 13px 0 7px; color: var(--soft); }
    .subtle { color: var(--muted); font-size: 13px; margin-top: 3px; }
    .header-actions { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; justify-content: flex-end; }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      padding: 6px 10px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: var(--panel-2);
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent); }
    main { padding: 18px 22px 28px; }
    .stats {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(132px, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }
    .stat {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 11px 12px;
      background: linear-gradient(180deg, #171d24, #14191f);
      min-height: 72px;
    }
    .stat .label { color: var(--muted); font-size: 12px; }
    .stat .value { font-size: 24px; font-weight: 800; margin-top: 7px; }
    .stat .hint { color: var(--muted); font-size: 11px; margin-top: 3px; }
    .layout {
      display: grid;
      grid-template-columns: minmax(430px, 1.02fr) minmax(420px, 0.98fr);
      gap: 16px;
      align-items: start;
    }
    section {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      overflow: hidden;
    }
    .section-head {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 12px;
      padding: 14px 14px 11px;
      border-bottom: 1px solid var(--line);
      background: #151a21;
    }
    .section-body { padding: 14px; }
    .toolbar {
      display: grid;
      grid-template-columns: minmax(180px, 1fr) 150px 150px;
      gap: 9px;
      align-items: center;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      background: #12171d;
    }
    button, select, input {
      min-height: 34px;
      padding: 7px 10px;
      background: #222832;
      color: var(--text);
      border: 1px solid #3a4350;
      border-radius: 6px;
      font: inherit;
      font-size: 13px;
    }
    button { cursor: pointer; font-weight: 650; }
    button.primary { background: #244b3d; border-color: #35745d; color: #eafff7; }
    button.secondary { background: #242b36; }
    button.full { width: 100%; }
    button:hover { border-color: #5a6676; }
    input { width: 100%; }
    .queue { max-height: 620px; overflow: auto; }
    .entry {
      display: grid;
      grid-template-columns: 58px 1fr auto;
      gap: 12px;
      padding: 12px 14px;
      border-top: 1px solid var(--line);
      cursor: pointer;
    }
    .entry:first-child { border-top: 0; }
    .entry:hover, .entry.active { background: #202733; }
    .entry.active { box-shadow: inset 3px 0 0 var(--accent); }
    .entry-id { color: var(--muted); font-size: 12px; }
    .entry-title { font-size: 13px; line-height: 1.35; }
    .entry-meta { color: var(--muted); font-size: 12px; margin-top: 5px; }
    .badge {
      display: inline-flex;
      align-items: center;
      height: 24px;
      padding: 0 8px;
      border-radius: 999px;
      border: 1px solid var(--line);
      font-size: 12px;
      white-space: nowrap;
    }
    .badge.neutral { color: var(--muted); background: #151a21; }
    .badge.critical, .badge.high { color: var(--danger); border-color: #6a363b; background: #2a171b; }
    .badge.warning { color: var(--warn); border-color: #70552c; background: #2a2114; }
    .badge.info { color: var(--info); border-color: #2f4b70; background: #152132; }
    .detail-grid { display: grid; gap: 12px; }
    .detail-box {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #12161d;
    }
    .agent-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .kv {
      display: grid;
      grid-template-columns: 116px 1fr;
      gap: 6px 10px;
      font-size: 13px;
    }
    .kv span:nth-child(odd) { color: var(--muted); }
    .summary-list { display: grid; gap: 10px; max-height: 340px; overflow: auto; }
    .summary-item {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 11px;
      background: #12161d;
      font-size: 13px;
      line-height: 1.4;
    }
    .summary-item strong { display: block; margin-bottom: 5px; }
    .timeline {
      display: grid;
      gap: 8px;
      max-height: 270px;
      overflow: auto;
    }
    .timeline-item {
      padding: 10px 11px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #12161d;
      font-size: 13px;
    }
    .timeline-item .meta { color: var(--muted); font-size: 12px; margin-top: 5px; }
    pre {
      white-space: pre-wrap;
      word-break: break-word;
      background: #0a0d11;
      border: 1px solid var(--line);
      padding: 12px;
      border-radius: 8px;
      max-height: 360px;
      overflow: auto;
      font-size: 12px;
      line-height: 1.45;
    }
    .split { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .empty { color: var(--muted); padding: 18px; text-align: center; }
    .notice {
      border: 1px solid #2f4b70;
      background: #121d2a;
      color: #cfe2ff;
      border-radius: 8px;
      padding: 10px 12px;
      font-size: 13px;
      margin-bottom: 12px;
    }
    @media (max-width: 980px) {
      header { align-items: flex-start; flex-direction: column; }
      .stats { grid-template-columns: repeat(2, minmax(130px, 1fr)); }
      .layout { grid-template-columns: 1fr; }
      .split { grid-template-columns: 1fr; }
      .agent-grid { grid-template-columns: 1fr; }
      .toolbar { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>CIDT Anomaly Operations</h1>
      <p class="subtle">Entry-based incident triage over alerts, anomaly scores, SLM summaries, and feedback.</p>
    </div>
    <div class="header-actions">
      <span class="pill"><span class="dot"></span><span id="healthText">Checking API</span></span>
      <span class="pill" id="modelText">Model pending</span>
      <button class="secondary" onclick="refreshAll()">Refresh</button>
    </div>
  </header>
  <main>
    <div class="stats">
      <div class="stat"><div class="label">Log entries</div><div class="value" id="statEntries">-</div><div class="hint">full DB count</div></div>
      <div class="stat"><div class="label">High priority</div><div class="value" id="statCritical">-</div><div class="hint">critical plus high</div></div>
      <div class="stat"><div class="label">Prometheus points</div><div class="value" id="statPoints">-</div><div class="hint">raw samples</div></div>
      <div class="stat"><div class="label">SLM extractions</div><div class="value" id="statSlmExtractions">-</div><div class="hint">one per log entry</div></div>
      <div class="stat"><div class="label">ML outputs</div><div class="value" id="statModelOutputs">-</div><div class="hint">one per log entry</div></div>
      <div class="stat"><div class="label">Incidents</div><div class="value" id="statIncidents">-</div><div class="hint">generated reports</div></div>
      <div class="stat"><div class="label">Feedback</div><div class="value" id="statFeedback">-</div><div class="hint">training labels</div></div>
      <div class="stat"><div class="label">Best ML model</div><div class="value" id="statModel" style="font-size: 17px;">-</div><div class="hint">latest run</div></div>
    </div>
    <div class="layout">
    <section>
      <div class="section-head">
        <div>
          <h2>Entry Queue</h2>
          <p class="subtle">Select a log entry to investigate or attach feedback.</p>
        </div>
        <span class="pill" id="queueCount">0 shown</span>
      </div>
      <div class="toolbar">
        <input id="searchBox" placeholder="Search message, instance, source" oninput="renderEntries()">
        <select id="entryTypeFilter" onchange="renderEntries()">
          <option value="">All entry types</option>
          <option value="slm_observation">SLM observations</option>
          <option value="alert">Alerts</option>
          <option value="anomaly_score">Anomaly scores</option>
        </select>
        <select id="severityFilter" onchange="renderEntries()">
          <option value="">All severities</option>
          <option value="critical">Critical</option>
          <option value="high">High</option>
          <option value="warning">Warning</option>
          <option value="info">Info</option>
        </select>
      </div>
      <div id="logEntries" class="queue"></div>
    </section>

    <div class="detail-grid">
    <section>
      <div class="section-head">
        <div>
          <h2>Selected Entry</h2>
          <p class="subtle">Generate an incident from the entry, then record engineer feedback.</p>
        </div>
      </div>
      <div class="section-body">
        <div class="notice">This workflow is log-entry based. Pick one entry, generate an incident report for that exact event, then use feedback to teach the next training loop.</div>
        <div id="selectedEntry" class="detail-box empty">No entry selected.</div>
        <div class="split" style="margin-top: 12px;">
          <button class="primary" onclick="generateIncident()">Analyze Entry</button>
          <button class="secondary" onclick="copyEntryId()">Copy Entry ID</button>
        </div>
      </div>
    </section>

    <section>
      <div class="section-head">
        <h2>Incident Agent</h2>
        <span class="pill">Gemma report</span>
      </div>
      <div class="section-body">
        <pre id="incident">Generate an incident from a selected entry.</pre>
      </div>
    </section>

    <div class="agent-grid">
      <section>
        <div class="section-head">
          <h2>Feedback</h2>
          <span class="pill">Entry scoped</span>
        </div>
        <div class="section-body">
          <div class="split">
            <select id="fbType">
              <option>true_positive</option>
              <option>false_positive</option>
              <option>known_maintenance</option>
              <option>duplicate</option>
              <option>resolved</option>
            </select>
            <button class="primary" onclick="submitFeedback()">Submit</button>
          </div>
          <input id="fbNote" placeholder="Add a short note for future retraining" style="margin-top: 8px;">
          <pre id="feedback">Feedback will appear here.</pre>
        </div>
      </section>

      <section>
        <div class="section-head">
          <h2>Recent Reports</h2>
          <span class="pill" id="reportCount">0</span>
        </div>
        <div class="section-body timeline" id="incidentReports"></div>
      </section>
    </div>

    <section>
      <div class="section-head">
        <h2>SLM Observations</h2>
        <button class="secondary" onclick="loadObservations()">Refresh</button>
      </div>
      <div id="observations" class="section-body summary-list"></div>
    </section>
    </div>
    </div>
  </main>
<script>
const state = { entries: [], observations: [], feedback: [], incidents: [], stats: {}, selected: null };

async function j(url, opts) {
  const r = await fetch(url, opts);
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || 'Request failed');
  return data;
}

function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function severityClass(value) {
  const v = String(value || 'info').toLowerCase();
  return ['critical','high','warning'].includes(v) ? v : 'info';
}

function setStats() {
  const severity = state.stats.severity_counts || {};
  document.getElementById('statEntries').textContent = state.stats.log_entries ?? state.entries.length;
  document.getElementById('statCritical').textContent = (severity.critical || 0) + (severity.high || 0);
  document.getElementById('statPoints').textContent = state.stats.telemetry_points ?? '-';
  document.getElementById('statSlmExtractions').textContent = state.stats.entry_slm_extractions ?? '-';
  document.getElementById('statModelOutputs').textContent = state.stats.entry_model_outputs ?? '-';
  document.getElementById('statIncidents').textContent = state.stats.incident_reports ?? state.incidents.length;
  document.getElementById('statFeedback').textContent = state.stats.feedback ?? state.feedback.length;
  document.getElementById('statModel').textContent = state.stats.best_model || '-';
}

async function refreshAll() {
  document.getElementById('healthText').textContent = 'Refreshing';
  const health = await j('/api/health');
  state.stats = await j('/api/stats');
  state.entries = await j('/api/log-entries?limit=120');
  state.observations = await j('/api/observations?limit=25');
  state.feedback = await j('/api/feedback?limit=50');
  state.incidents = await j('/api/incidents?limit=10');
  document.getElementById('healthText').textContent = 'Online';
  document.getElementById('modelText').textContent = health.agent_model;
  renderEntries();
  renderObservations();
  renderIncidents();
  setStats();
}

async function loadLogEntries() {
  state.entries = await j('/api/log-entries?limit=120');
  renderEntries();
  setStats();
}

function renderEntries() {
  const type = document.getElementById('entryTypeFilter').value;
  const severity = document.getElementById('severityFilter').value;
  const search = document.getElementById('searchBox').value.toLowerCase().trim();
  const rows = state.entries.filter(e => {
    const haystack = `${e.message || ''} ${e.instance || ''} ${e.source || ''} ${e.entry_type || ''}`.toLowerCase();
    return (!type || e.entry_type === type) && (!severity || e.severity === severity) && (!search || haystack.includes(search));
  });
  const root = document.getElementById('logEntries');
  document.getElementById('queueCount').textContent = `${rows.length} shown`;
  if (!rows.length) {
    root.innerHTML = '<div class="empty">No matching entries.</div>';
    return;
  }
  root.innerHTML = rows.map(row => `
    <div class="entry ${state.selected && state.selected.id === row.id ? 'active' : ''}" onclick="selectEntry(${row.id})">
      <div class="entry-id">#${esc(row.id)}<br>${esc(row.entry_type)}</div>
      <div>
        <div class="entry-title">${esc(row.message)}</div>
        <div class="entry-meta">${esc(row.instance)} ${row.timestamp ? '&middot; ' + esc(row.timestamp) : ''}</div>
      </div>
      <span class="badge ${severityClass(row.severity)}">${esc(row.severity || 'info')}</span>
    </div>
  `).join('');
}

function selectEntry(id) {
  state.selected = state.entries.find(e => e.id === id) || null;
  renderEntries();
  renderSelected();
}

function renderSelected() {
  const el = document.getElementById('selectedEntry');
  const row = state.selected;
  if (!row) {
    el.className = 'detail-box empty';
    el.innerHTML = 'No entry selected.';
    return;
  }
  el.className = 'detail-box';
  el.innerHTML = `
    <div class="kv">
      <span>Entry ID</span><strong>${esc(row.id)}</strong>
      <span>Type</span><strong>${esc(row.entry_type)}</strong>
      <span>Severity</span><strong>${esc(row.severity || 'info')}</strong>
      <span>Source</span><strong>${esc(row.source || '')}</strong>
      <span>Instance</span><strong>${esc(row.instance)}</strong>
      <span>Timestamp</span><strong>${esc(row.timestamp || '')}</strong>
      <span>Created</span><strong>${esc(row.created_at || '')}</strong>
    </div>
    <h3>Message</h3>
    <p class="subtle">${esc(row.message)}</p>
  `;
  document.getElementById('fbNote').value = '';
}

async function loadObservations() {
  state.observations = await j('/api/observations?limit=25');
  renderObservations();
  setStats();
}

function renderObservations() {
  const root = document.getElementById('observations');
  if (!state.observations.length) {
    root.innerHTML = '<div class="empty">No observations.</div>';
    return;
  }
  root.innerHTML = state.observations.slice(0, 10).map(row => `
    <div class="summary-item">
      <strong>${esc(row.instance)} <span class="badge ${severityClass(row.severity)}">${esc(row.severity)}</span></strong>
      <div>${esc(row.summary)}</div>
      <p class="subtle">${esc(row.recommendation)}</p>
    </div>
  `).join('');
}

function renderIncidents() {
  const root = document.getElementById('incidentReports');
  document.getElementById('reportCount').textContent = state.incidents.length;
  if (!state.incidents.length) {
    root.innerHTML = '<div class="empty">No incident reports yet.</div>';
    return;
  }
  root.innerHTML = state.incidents.map(row => `
    <div class="timeline-item">
      <strong>${esc(row.title || 'Untitled incident')}</strong>
      <div class="meta">#${esc(row.id)} ${row.log_entry_id ? '&middot; entry #' + esc(row.log_entry_id) : ''} &middot; ${esc(row.instance || '')}</div>
      <div class="meta">${esc(row.severity || '')} &middot; ${esc(row.created_at || '')}</div>
    </div>
  `).join('');
}

async function generateIncident() {
  if (!state.selected) {
    document.getElementById('incident').textContent = 'Select a log entry first.';
    return;
  }
  document.getElementById('incident').textContent = 'Generating incident report...';
  try {
    const data = await j('/api/incidents/generate', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({log_entry_id: state.selected.id})
    });
    const recommendations = safeJson(data.recommendations, []);
    document.getElementById('incident').textContent =
      `${data.title}\\n\\nSeverity: ${data.severity}\\nRoot cause: ${data.likely_root_cause}\\n\\n${data.summary}\\n\\nRecommendations:\\n- ${recommendations.join('\\n- ')}`;
    state.incidents = await j('/api/incidents?limit=10');
    state.stats = await j('/api/stats');
    renderIncidents();
    setStats();
  } catch (err) {
    document.getElementById('incident').textContent = err.message;
  }
}

function safeJson(value, fallback) {
  try { return JSON.parse(value); } catch { return fallback; }
}

async function submitFeedback() {
  if (!state.selected) {
    document.getElementById('feedback').textContent = 'Select a log entry first.';
    return;
  }
  const body = {
    log_entry_id: state.selected.id,
    instance: state.selected.instance,
    feedback_type: document.getElementById('fbType').value,
    note: document.getElementById('fbNote').value,
    user: 'local'
  };
  try {
    const data = await j('/api/feedback', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
    document.getElementById('feedback').textContent = `Saved feedback #${data.id} for entry #${state.selected.id}.`;
    state.feedback = await j('/api/feedback?limit=50');
    state.stats = await j('/api/stats');
    setStats();
  } catch (err) {
    document.getElementById('feedback').textContent = err.message;
  }
}

async function copyEntryId() {
  if (!state.selected) return;
  await navigator.clipboard.writeText(String(state.selected.id));
}

refreshAll();
</script>
</body>
</html>"""


if __name__ == "__main__":
    main()
