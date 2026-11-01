import { ResultRow } from "./study";

// Best-effort save to the backend DB. localStorage remains the guaranteed
// local backup, so a failed POST never loses data.
export async function postTrial(row: ResultRow): Promise<void> {
  try {
    await fetch("/api/results", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(row),
    });
  } catch {
    /* ignore: offline / local dev without functions */
  }
}

export interface DbRow {
  id: number;
  session_id: string;
  judge: string | null;
  mode: string | null;
  transition: string | null;
  cut_time: number | null;
  base_adjust: number | null;
  ordering: number | null;
  offset_s: number | null;
  trigger_offset: number | null;
  rating: string | null;
  rating_value: number | null;
  note: string | null;
  client_iso: string | null;
  ip: string | null;
  created_at: string;
}

export async function fetchResults(token: string): Promise<DbRow[]> {
  const res = await fetch("/api/results", { headers: { Authorization: `Bearer ${token}` } });
  if (!res.ok) throw new Error(res.status === 401 ? "Onjuist wachtwoord." : `Serverfout (${res.status}).`);
  const data = await res.json();
  return data.rows as DbRow[];
}

export function dbRowsToCSV(rows: DbRow[]): string {
  const cols: (keyof DbRow)[] = [
    "id", "session_id", "judge", "mode", "transition", "cut_time", "base_adjust",
    "ordering", "offset_s", "trigger_offset", "rating", "rating_value", "note",
    "client_iso", "created_at",
  ];
  const esc = (v: unknown) => {
    const s = v === null || v === undefined ? "" : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return cols.join(",") + "\n" + rows.map((r) => cols.map((c) => esc(r[c])).join(",")).join("\n") + "\n";
}
