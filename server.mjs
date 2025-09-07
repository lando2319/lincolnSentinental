import 'dotenv/config';
import express from 'express';
import { QdrantClient } from '@qdrant/js-client-rest';
import { pipeline, env } from '@xenova/transformers';

const PORT = Number(process.env.PORT || 8010);
const COLLECTION = process.env.COLLECTION || 'lincoln_docs';
const QDRANT_URL = process.env.QDRANT_URL || 'http://127.0.0.1:6333';

const LLM_MODE = (process.env.LLM_MODE || 'OLLAMA').toUpperCase();
const OLLAMA_URL = process.env.OLLAMA_URL || 'http://127.0.0.1:11434/api/chat';
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || 'gemma3:4b';
const LCPP_URL = process.env.LLM_URL || 'http://127.0.0.1:8080/v1/chat/completions';
const LCPP_MODEL = process.env.LLM_MODEL || 'gemma-3-4b-it';

const TOP_K = 12;
const RELEVANCE_MIN = 0.62;      // cosine similarity gate
const MAX_CTX_CHUNKS = 8;

env.allowLocalModels = true;
env.localModelPath = process.env.MODEL_DIR || './models';
if (process.env.HF_OFFLINE === '1') env.remoteModels = false;

let embedder;
async function embed(texts) {
  if (!embedder) embedder = await pipeline('feature-extraction', process.env.EMBED_MODEL || 'Xenova/bge-small-en-v1.5');
  const out = await embedder(texts, { pooling: 'mean', normalize: true });
  const arrs = Array.isArray(out) ? out : [out];
  return arrs.map(t => Array.from(t.data));
}

const qc = new QdrantClient({ url: QDRANT_URL });
const app = express();
app.use(express.json({ limit: '1mb' }));

function buildPrompt(question, contexts) {
  const sys = 'You are a concise in-car assistant for a 1967 Lincoln Continental. If context is provided, answer ONLY from it and add bracketed citations like [filename p.X]. If no context is provided, say you have no supporting documents and avoid guessing.';
  if (!contexts?.length) {
    return [
      { role: 'system', content: sys },
      { role: 'user', content: `Question: ${question}\n\n(No context retrieved above threshold.)\nAnswer:` },
    ];
  }
  const ctx = contexts.map((c, i) => `### ${i + 1} [${c.filename} p.${c.page}]\n${c.text}`).join('\n\n');
  return [
    { role: 'system', content: sys },
    { role: 'user', content: `Question: ${question}\n\nContext:\n${ctx}\n\nAnswer:` },
  ];
}

async function callLLM(messages) {
  if (LLM_MODE === 'LLAMACPP') {
    const r = await fetch(LCPP_URL, {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model: LCPP_MODEL, messages, temperature: 0.2, max_tokens: 512 })
    });
    const j = await r.json(); if (!r.ok) throw new Error(JSON.stringify(j));
    return j.choices?.[0]?.message?.content ?? '';
  } else {
    const r = await fetch(OLLAMA_URL, {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model: OLLAMA_MODEL, messages,
        options: { num_ctx: 8192, num_predict: 512, temperature: 0.2 }, stream: false })
    });
    const j = await r.json(); if (!r.ok) throw new Error(JSON.stringify(j));
    return j.message?.content ?? '';
  }
}

app.post('/ask', async (req, res) => {
  try {
    const q = String(req.body?.question || '').trim();
    if (!q) return res.status(400).json({ error: 'question required' });

    // 1) embed the query
    const [qvec] = await embed([q]);

    // 2) pull a wider set, no early gate
    const raw = await qc.search(COLLECTION, {
      vector: qvec,
      limit: 24,               // pull more, we'll filter
      with_payload: true
    });

    // normalize hits
    const hitsRaw = (raw || []).map(h => ({
      text: h.payload.text || '',
      filename: h.payload.filename,
      page: h.payload.page,
      score: h.score
    }));

    if (!hitsRaw.length) {
      const messages = buildPrompt(q, []);
      const answer = await callLLM(messages);
      return res.json({ answer, routed: 'no_context', citations: [], used: [] });
    }

    // 3) keep only same-file as best hit, drop low scores & off-topic
    const top = hitsRaw[0];
    const SYNS = ['defog','defrost','demist']; // quick keyword gate helps with HVAC questions
    const qwords = Array.from(new Set(
      q.toLowerCase().match(/[a-z0-9]+/g) || []
    ));
    const kw = (qwords.includes('defog') || qwords.includes('defrost') || qwords.includes('demist'))
      ? SYNS
      : qwords.filter(w => w.length >= 4);

    let hits = hitsRaw
      .filter(h => h.filename === top.filename)         // same manual/page family
      .filter(h => h.score >= 0.45)                     // bump if you want stricter
      .filter(h => kw.length ? kw.some(w => (h.text.toLowerCase().includes(w))) : true)
      .sort((a,b) => b.score - a.score)
      .slice(0, 6);                                     // keep it tight

    // fallback: if filtering became too strict, use just the top hit
    if (!hits.length) hits = [top];

    // 4) build prompt + dedup citations
    const messages = buildPrompt(q, hits);
    const answer = await callLLM(messages);

    const citations = [];
    for (const h of hits) {
      if (!citations.some(c => c.filename === h.filename && c.page === h.page)) {
        citations.push({ filename: h.filename, page: h.page });
        if (citations.length >= 3) break;
      }
    }

    res.json({ answer, routed: 'retrieval', citations, used: hits });
  } catch (e) {
    res.status(500).json({ error: String(e) });
  }
});

app.post('/debug/embed', async (req, res) => {
  try {
    const text = String(req.body?.text || '').trim();
    if (!text) return res.status(400).json({ error: 'text required' });
    const [vec] = await embed([text]);
    res.json({ dim: vec.length, vector: vec });
  } catch (e) {
    res.status(500).json({ error: String(e) });
  }
});

app.get('/healthz', (_req, res) => res.json({ ok: true }));
app.listen(PORT, () => console.log(`RAG server on :${PORT}`));
