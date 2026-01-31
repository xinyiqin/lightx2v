/**
 * Express server that serves static files and provides lightx2v result_url proxy API.
 * Railway runs npm start without a shell; PORT from env.
 */
import express from 'express';
import { createServer } from 'http';
import handler from 'serve-handler';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
const port = process.env.PORT || '3000';

/**
 * LightX2V result_url proxy: GET /api/lightx2v/result_url?task_id=...&output_name=...
 * Forwards to LightX2V backend (cloud or local) and returns { url: "..." }.
 * Query params: task_id, output_name, is_cloud (optional, default from LIGHTX2V_CLOUD_URL presence)
 */
app.get('/api/lightx2v/result_url', async (req, res) => {
  const taskId = req.query.task_id;
  const outputName = req.query.output_name || req.query.name;
  const isCloud = req.query.is_cloud === 'true' || req.query.is_cloud === '1';

  if (!taskId || !outputName) {
    return res.status(400).json({ error: 'task_id and output_name are required' });
  }

  const baseUrl = isCloud
    ? (process.env.LIGHTX2V_CLOUD_URL || 'https://x2v.light-ai.top').trim()
    : (process.env.LIGHTX2V_URL || '').trim();

  if (!baseUrl) {
    return res.status(400).json({
      error: isCloud
        ? 'LIGHTX2V_CLOUD_URL not configured'
        : 'LIGHTX2V_URL not configured (use is_cloud=true for cloud)'
    });
  }

  const token = isCloud
    ? (process.env.LIGHTX2V_CLOUD_TOKEN || '').trim()
    : (process.env.LIGHTX2V_TOKEN || '').trim();

  const targetUrl = `${baseUrl.replace(/\/$/, '')}/api/v1/task/result_url?task_id=${encodeURIComponent(taskId)}&name=${encodeURIComponent(outputName)}`;
  const headers = {
    Accept: 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {})
  };

  try {
    const proxyRes = await fetch(targetUrl, { method: 'GET', headers });
    const data = await proxyRes.json().catch(() => ({}));

    if (!proxyRes.ok) {
      return res.status(proxyRes.status).json(data || { error: 'LightX2V result_url failed' });
    }
    if (!data.url) {
      return res.status(502).json({ error: 'LightX2V response missing url' });
    }
    return res.json(data);
  } catch (err) {
    console.error('[lightx2v result_url proxy]', err);
    return res.status(502).json({ error: String(err.message || err) });
  }
});

// Serve static files from dist
app.use((req, res) => {
  return handler(req, res, {
    public: path.join(__dirname, 'dist'),
    cleanUrls: true,
    rewrites: [{ source: '**', destination: '/index.html' }]
  });
});

const server = createServer(app);
server.listen(parseInt(port, 10) || 3000, '0.0.0.0', () => {
  console.log(`Server listening on port ${port}`);
});
server.on('error', (err) => {
  console.error('Server error:', err);
  process.exit(1);
});
