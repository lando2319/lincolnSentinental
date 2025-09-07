// embedder.mjs
import { pipeline, env } from '@xenova/transformers';
env.allowLocalModels = true;
env.localModelPath = './models';           // cached weights live here

let extractor;
export async function embed(texts) {
  if (!extractor) extractor = await pipeline('feature-extraction', process.env.EMBED_MODEL);
  const out = await extractor(texts, { pooling: 'mean', normalize: true });
  const arrs = Array.isArray(out) ? out : [out];
  return arrs.map(t => Array.from(t.data));
}
