import { neon } from "@neondatabase/serverless";
import type { VercelRequest, VercelResponse } from "@vercel/node";

const sql = neon(process.env.DATABASE_URL!);

export default async function handler(req: VercelRequest, res: VercelResponse) {
  try {
    if (req.method === "POST") {
      const r = typeof req.body === "string" ? JSON.parse(req.body) : req.body;
      if (!r || !r.sessionId) return res.status(400).json({ error: "missing sessionId" });
      // Idempotent per (session, trial): a revised answer replaces the prior row
      // for the same ordering instead of appending a duplicate. Atomic via a
      // data-modifying CTE.
      await sql`
        WITH del AS (
          DELETE FROM trials WHERE session_id = ${r.sessionId} AND ordering = ${r.order ?? -1}
        )
        INSERT INTO trials
          (session_id, judge, mode, transition, cut_time, base_adjust, ordering,
           offset_s, trigger_offset, rating, rating_value, note, client_iso)
        VALUES
          (${r.sessionId}, ${r.judge ?? null}, ${r.mode ?? null}, ${r.transition ?? null},
           ${r.cutTime ?? null}, ${r.baseAdjust ?? null}, ${r.order ?? null},
           ${r.offset ?? null}, ${r.triggerOffset ?? null}, ${r.rating ?? null},
           ${r.ratingValue ?? null}, ${r.note ?? ""}, ${r.isoTime ?? null})`;
      return res.status(200).json({ ok: true });
    }

    return res.status(405).json({ error: "method not allowed" });
  } catch (e) {
    return res.status(500).json({ error: String(e instanceof Error ? e.message : e) });
  }
}
