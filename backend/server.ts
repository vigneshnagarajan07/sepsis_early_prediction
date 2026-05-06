/**
 * server.ts  –  Sepsis AI  ·  Express dev-server
 *
 * Responsibilities
 *   1.  Serve the Vite dev / prod frontend
 *   2.  Proxy /api/* → Python FastAPI backend (port 8000)
 *
 * Run:  npx tsx server.ts
 * Requires python backend running:  cd ../sepsis-backend && bash start.sh
 */

import express                      from "express";
import { createProxyMiddleware }    from "http-proxy-middleware";
import { createServer as createViteServer } from "vite";
import path                         from "path";
import { fileURLToPath }            from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

const PORT         = 3000;
const BACKEND_URL  = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

async function startServer() {
  const app = express();

  // ── /api  →  Python FastAPI ────────────────────────────────────────────────
  app.use(
    "/api",
    createProxyMiddleware({
      target:       BACKEND_URL,
      changeOrigin: true,
      on: {
        error: (err, req, res) => {
          console.error(`[proxy] ${req.method} ${req.url} → ${err.message}`);
          if (!res.headersSent) {
            (res as express.Response).status(502).json({
              error:   "Python backend unreachable",
              detail:  err.message,
              hint:    `Make sure the FastAPI server is running: cd sepsis-backend && bash start.sh`,
            });
          }
        },
        proxyReq: (proxyReq, req) => {
          console.log(`[proxy] ${req.method} ${req.url} → ${BACKEND_URL}`);
        },
      },
    }),
  );

  // ── Vite middleware or static dist ─────────────────────────────────────────
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(__dirname, "dist");
    app.use(express.static(distPath));
    app.get("*", (_req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`\n  Sepsis AI  →  http://localhost:${PORT}`);
    console.log(`  API proxy  →  ${BACKEND_URL}/api/*\n`);
  });
}

startServer();
