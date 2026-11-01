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
