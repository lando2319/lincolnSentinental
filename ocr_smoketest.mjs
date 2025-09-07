import { spawn } from 'node:child_process';
import fs from 'fs';
import path from 'path';
import os from 'os';
import Tesseract from 'tesseract.js';

const pdfPath = process.argv[2];
if (!pdfPath || !fs.existsSync(pdfPath)) {
  console.error('File not found:', pdfPath);
  process.exit(1);
}

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'ppm-'));
const prefix = path.join(tmp, 'page'); // pdftoppm will create page-1.png

// Render **page 1** of the PDF to PNG with Poppler
await new Promise((resolve, reject) => {
  const p = spawn('pdftoppm', ['-png', '-singlefile', '-f', '1', '-l', '1', pdfPath, prefix], { stdio: 'inherit' });
  p.on('close', code => code === 0 ? resolve() : reject(new Error('pdftoppm failed: ' + code)));
});

const imgPath = `${prefix}.png`;
if (!fs.existsSync(imgPath)) {
  console.error('Rasterized PNG not found:', imgPath);
  process.exit(1);
}

// OCR the PNG
console.log('[OCR] starting tesseract…');
const { data } = await Tesseract.recognize(fs.readFileSync(imgPath), 'eng', {
  logger: m => m.progress != null && console.log('[OCR]', Math.round(m.progress * 100) + '%', m.status)
});

console.log('\n[RESULT] OCR (first 500 chars):\n');
console.log((data.text || '').slice(0, 500));
