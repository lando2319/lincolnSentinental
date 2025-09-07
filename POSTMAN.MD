How to use it (quick flow)

Health → send “API • Health”. Should return { ok: true }.

Count → send “Qdrant • Count Points”. If count is 0, the ingest didn’t index (often because the PDF is image-only — we can add OCR if needed).

Scroll Sample → see a few chunks & confirm your payload text looks right.

Embed + Search:

Send “API • Debug Embed” with text: "defog windshield".

Copy the vector array from the response.

Open “Qdrant • Vector Search (paste vector)” and paste it into vector: [ ... ]. Send it.

You should get top matches with filename, page, and preview text in payload.

If vector search finds relevant chunks but /ask still says no_context, your gate is too strict. In server.mjs either:

remove score_threshold temporarily, or

lower it (e.g., RELEVANCE_MIN = 0.35) and keep score_threshold: RELEVANCE_MIN.

If Count > 0 but Scroll shows empty or junk text, your PDF is likely scans (no selectable text). Say the word and I’ll drop an OCR add-on for the ingest script.