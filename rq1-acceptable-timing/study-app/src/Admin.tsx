import { useEffect, useMemo, useState } from "react";
import { DbRow, dbRowsToCSV, fetchResults } from "./api";
import { download } from "./study";

interface SessionGroup {
  sessionId: string;
  judge: string;
  mode: string;
  count: number;
  last: string;
}

export default function Admin() {
  const [token, setToken] = useState(() => sessionStorage.getItem("admin-token") || "");
  const [rows, setRows] = useState<DbRow[] | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function load(t: string) {
    setLoading(true);
    setError("");
    try {
      const data = await fetchResults(t);
      setRows(data);
      sessionStorage.setItem("admin-token", t);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setRows(null);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (token) load(token);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const sessions = useMemo<SessionGroup[]>(() => {
    if (!rows) return [];
    const map = new Map<string, SessionGroup>();
    for (const r of rows) {
      const g = map.get(r.session_id) ?? {
        sessionId: r.session_id, judge: r.judge ?? "", mode: r.mode ?? "",
        count: 0, last: r.created_at,
      };
      g.count++;
      if (r.created_at > g.last) g.last = r.created_at;
      map.set(r.session_id, g);
    }
    return Array.from(map.values()).sort((a, b) => b.last.localeCompare(a.last));
  }, [rows]);

  if (rows === null) {
    return (
      <div className="wrap">
        <h1>Beheer</h1>
        <p className="lede">Voer het beheerderswachtwoord in om opgeslagen sessies te bekijken.</p>
        <div className="card" style={{ maxWidth: 420 }}>
          <label>Wachtwoord
            <input type="password" value={token} onChange={(e) => setToken(e.target.value)}
                   onKeyDown={(e) => e.key === "Enter" && load(token)} />
          </label>
          <div style={{ marginTop: 12 }}>
            <button className="btn-primary" onClick={() => load(token)} disabled={loading || !token}>
              {loading ? "Bezig" : "Inloggen"}
            </button>
          </div>
          {error && <p style={{ color: "var(--bad)", marginTop: 10 }}>{error}</p>}
        </div>
      </div>
    );
  }

  return (
    <div className="wrap">
      <div className="runbar">
        <h1 style={{ margin: 0 }}>Beheer ({rows.length} regels)</h1>
        <div style={{ display: "flex", gap: 10 }}>
          <button className="btn-secondary" onClick={() => load(token)}>Vernieuwen</button>
          <button className="btn-primary" onClick={() => download(`alle_resultaten_${Date.now()}.csv`, dbRowsToCSV(rows), "text/csv")}>Download CSV</button>
        </div>
      </div>

      <div className="card">
        <h3>Sessies ({sessions.length})</h3>
        <table>
          <thead><tr><th>Sessie</th><th>Deelnemer</th><th>Modus</th><th>Proeven</th><th>Laatste</th><th>Export</th></tr></thead>
          <tbody>
            {sessions.map((s) => {
              const sessionRows = rows.filter((r) => r.session_id === s.sessionId);
              return (
                <tr key={s.sessionId}>
                  <td>{s.sessionId}</td>
                  <td>{s.judge}</td>
                  <td>{s.mode}</td>
                  <td>{s.count}</td>
                  <td>{new Date(s.last).toLocaleString("nl-NL")}</td>
                  <td>
                    <button className="link" onClick={() => download(`sessie_${s.sessionId}.csv`, dbRowsToCSV(sessionRows), "text/csv")}>CSV</button>
                    {" · "}
                    <button className="link" onClick={() => download(`sessie_${s.sessionId}.json`, JSON.stringify(sessionRows, null, 2), "application/json")}>JSON</button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="card">
        <h3>Laatste regels</h3>
        <table>
          <thead><tr><th>Tijd</th><th>Sessie</th><th>Overgang</th><th>Offset</th><th>Trigger</th><th>Oordeel</th></tr></thead>
          <tbody>
            {rows.slice(0, 200).map((r) => (
              <tr key={r.id}>
                <td>{new Date(r.created_at).toLocaleTimeString("nl-NL")}</td>
                <td>{r.session_id}</td>
                <td>{r.transition}</td>
                <td>{r.offset_s == null ? "" : `${r.offset_s >= 0 ? "+" : ""}${r.offset_s.toFixed(2)}`}</td>
                <td>{r.trigger_offset == null ? "" : `${r.trigger_offset >= 0 ? "+" : ""}${r.trigger_offset.toFixed(2)}`}</td>
                <td>{r.rating ?? ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
