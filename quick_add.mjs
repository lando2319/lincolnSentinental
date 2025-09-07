import 'dotenv/config';
import { QdrantClient } from '@qdrant/js-client-rest';
import { pipeline, env } from '@xenova/transformers';
import crypto from 'node:crypto'; // for randomUUID()

const COLLECTION = process.env.COLLECTION || 'lincoln_docs';
const qc = new QdrantClient({ url: process.env.QDRANT_URL || 'http://127.0.0.1:6333' });

env.allowLocalModels = true;
env.localModelPath = process.env.MODEL_DIR || './models';
const modelName = process.env.EMBED_MODEL || 'Xenova/bge-small-en-v1.5';

const embedder = await pipeline('feature-extraction', modelName);

const text =
  'To defog the windshield, set the heater/AC to DEFROST, direct airflow to the windshield, ' +
  'set temperature to warm, and increase blower speed as needed.';

const out = await embedder([text], { pooling: 'mean', normalize: true });
const vec = Array.from((Array.isArray(out) ? out[0] : out).data);

// Use a VALID ID: either a UUID or a numeric ID
const id = crypto.randomUUID();          // <-- valid UUID
// const id = Date.now();                // <-- or a numeric ID

await qc.upsert(COLLECTION, {
  points: [{ id, vector: vec, payload: { filename: 'test.txt', page: 1, text } }],
});

console.log('Seeded one test chunk with id:', id);
