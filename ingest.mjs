// ingest.mjs — Poppler(300 DPI) + Tesseract OCR + cleaned text → Qdrant (with robust batching fix)
import 'dotenv/config';
import fs from 'fs';
import path from 'path';
import os from 'os';
import { spawnSync, spawn } from 'node:child_process';
import crypto from 'node:crypto';
import Tesseract from 'tesseract.js';
import { QdrantClient } from '@qdrant/js-client-rest';
import { pipeline, env } from '@xenova/transformers';

const DOCS_DIR    = path.resolve('./docs');
const QDRANT_URL  = process.env.QDRANT_URL || 'http://127.0.0.1:6333';
const COLLECTION  = process.env.COLLECTION || 'lincoln_docs';
const EMBED_MODEL = process.env.EMBED_MODEL || 'Xenova/bge-small-en-v1.5';
const EMBED_DIM   = 384;

// chunking tuned for manuals
const CHUNK_CHARS   = 900;
const CHUNK_OVERLAP = 120;
const BATCH         = 24;

env.allowLocalModels = true;
env.localModelPath   = process.env.MODEL_DIR || './models';
if (process.env.HF_OFFLINE === '1') env.remoteModels = false;

const qc = new QdrantClient({ url: QDRANT_URL });

// ---------- helpers ----------
function cleanText(t) {
  if (!t) return '';
  return t
    .replace(/\r/g, '\n')
    .replace(/=\s*=/g, ' ')
    .replace(/[—–]+/g, '-')
    .replace(/([A-Za-z])-\s*\n\s*([A-Za-z])/g, '$1$2')
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{2,}/g, '\n')
    .replace(/[“”]/g, '"').replace(/[‘’]/g, "'")
    .replace(/[^\S\n]+/g, ' ')
    .replace(/\s{2,}/g, ' ')
    .trim();
}

function chunkText(s, size = CHUNK_CHARS, overlap = CHUNK_OVERLAP) {
  const out = [];
  let i = 0;
  while (i < s.length) {
    let end = Math.min(i + size, s.length);
    const dot = s.lastIndexOf('.', end);
    if (dot > i + size * 0.5) end = dot + 1;
    out.push(s.slice(i, end).trim());
    i = Math.max(end - overlap, end);
  }
  return out.filter(Boolean);
}

function getPageCount(pdfPath) {
  const r = spawnSync('pdfinfo', [pdfPath], { encoding: 'utf8' });
  if (r.status !== 0) throw new Error(`pdfinfo failed: ${r.stderr}`);
  const m = r.stdout.match(/Pages:\s+(\d+)/);
  if (!m) throw new Error('Could not parse page count');
  return parseInt(m[1], 10);
}

function extractPageTextWithPdftotext(pdfPath, p) {
  const r = spawnSync(
    'pdftotext',
    ['-f', String(p), '-l', String(p), '-raw', '-enc', 'UTF-8', '-nopgbrk', pdfPath, '-'],
    { encoding: 'utf8' }
  );
  if (r.status !== 0) return '';
  return (r.stdout || '').replace(/\u0000/g, '').trim();
}

async function rasterizeAndOCR(pdfPath, p) {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'ppm-'));
  const prefix = path.join(tmp, `p${p}`);
  await new Promise((resolve, reject) => {
    const proc = spawn(
      'pdftoppm',
      ['-png', '-singlefile', '-rx', '300', '-ry', '300', '-f', String(p), '-l', String(p), pdfPath, prefix],
      { stdio: 'ignore' }
    );
    proc.on('close', code => (code === 0 ? resolve() : reject(new Error('pdftoppm failed: ' + code))));
  });
  const png = `${prefix}.png`;
  const { data } = await Tesseract.recognize(png, 'eng', { tessedit_pageseg_mode: 6 });
  return (data.text || '').trim();
}

async function ensureCollection() {
  try {
    await qc.createCollection(COLLECTION, { vectors: { size: EMBED_DIM, distance: 'Cosine' } });
    console.log(`Created collection ${COLLECTION} (size=${EMBED_DIM}, Cosine)`);
  } catch {
    console.log(`Collection ${COLLECTION} exists.`);
  }
}

// ---- robust embedder (splits batch tensor into per-text vectors) ----
let embedFn;
async function getEmbedder() {
  if (embedFn) return embedFn;
  console.log('Loading embedder:', EMBED_MODEL);
  const pipe = await pipeline('feature-extraction', EMBED_MODEL);
  embedFn = async (texts) => {
    const out = await pipe(texts, { pooling: 'mean', normalize: true });
    // Case 1: library returns array of tensors (per input)
    if (Array.isArray(out)) {
      return out.map(t => Array.from(t.data));
    }
    // Case 2: single tensor for the whole batch
    const { data, dims } = out; // dims e.g. [batch, dim] after pooling
    if (!dims || dims.length === 1) {
      // Single vector
      return [Array.from(data)];
    }
    const batch = dims[0];
    const dim = dims[dims.length - 1];
    const result = new Array(batch);
    for (let i = 0; i < batch; i++) {
      const start = i * dim;
      const end = start + dim;
      result[i] = Array.from(data.slice(start, end));
    }
    return result;
  };
  return embedFn;
}

// ---------- ingest ----------
async function ingestFile(pdfPath) {
  const filename = path.basename(pdfPath);
  const docId    = filename.replace(/\.[^.]+$/, '');
  const pages    = getPageCount(pdfPath);
  console.log(`\nIngesting ${filename} • pages: ${pages}`);

  const pageTexts = [];
  for (let p = 1; p <= pages; p++) {
    let text = extractPageTextWithPdftotext(pdfPath, p);
    let usedOCR = false;
    if (!text || text.length < 30) {
      usedOCR = true;
      text = await rasterizeAndOCR(pdfPath, p);
    }
    text = cleanText(text);
    pageTexts.push({ page: p, text });
    console.log(`  page ${p}: ${text.length} chars${usedOCR ? ' (OCR)' : ''}`);
  }

  const chunks = [];
  for (const p of pageTexts) {
    for (const part of chunkText(p.text)) {
      chunks.push({ page: p.page, text: part });
    }
  }
  console.log(`- total chunks: ${chunks.length}`);
  if (!chunks.length) { console.warn('! no chunks; skipping'); return; }

  const embed = await getEmbedder();
  for (let i = 0; i < chunks.length; i += BATCH) {
    const batch = chunks.slice(i, i + BATCH);
    const vecs  = await embed(batch.map(b => b.text));
    if (vecs.length !== batch.length) {
      throw new Error(`embedder returned ${vecs.length} vectors for ${batch.length} texts`);
    }
    const points = batch.map((b, j) => ({
      id: crypto.randomUUID(),
      vector: vecs[j],
      payload: { doc_id: docId, filename, page: b.page, text: b.text }
    }));
    await qc.upsert(COLLECTION, { points });
    process.stdout.write(`  upserted ${Math.min(i + BATCH, chunks.length)}/${chunks.length}\r`);
  }
  process.stdout.write('\nDone.\n');
}

// ---------- run ----------
(async () => {
  await ensureCollection();

  if (!fs.existsSync(DOCS_DIR)) {
    console.error('Docs dir not found:', DOCS_DIR);
    process.exit(1);
  }
  const pdfs = fs.readdirSync(DOCS_DIR).filter(f => f.toLowerCase().endsWith('.pdf'));
  console.log('Found PDFs:', pdfs.length);
  if (!pdfs.length) return;

  for (const f of pdfs) {
    await ingestFile(path.join(DOCS_DIR, f));
  }

  // show total count
  const r = await fetch(`${QDRANT_URL}/collections/${COLLECTION}/points/count`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ exact: true })
  });
  const j = await r.json();
  console.log('Collection point count:', j?.result?.count ?? j);
})();
