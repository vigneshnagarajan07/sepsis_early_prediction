import express from "express";
import { createServer as createViteServer } from "vite";
import { createProxyMiddleware } from "http-proxy-middleware";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const FASTAPI_URL = process.env.FASTAPI_URL || "http://localhost:8000";

async function startServer() {
  const app = express();
  const PORT = 3000;

  // ── Proxy all /api/* requests → FastAPI ML backend ──────────────────────────
  // Mount at '/' with a filter so Express does NOT strip the /api prefix.
  // FastAPI defines routes as /api/health, /api/predict — full path must arrive.
  app.use(
    createProxyMiddleware({
      target: FASTAPI_URL,
      changeOrigin: true,
      pathFilter: "/api/**",        // only forward /api/* paths
      on: {
        proxyReq: (_proxyReq, req) => {
          console.log(
            `[PROXY] ${req.method} ${req.url}  →  ${FASTAPI_URL}${req.url}`
          );
        },
        error: (err, _req, res: any) => {
          console.error("[PROXY] FastAPI unreachable:", err.message);
          res.status(502).json({
            error: "FastAPI ML backend is not running.",
            hint: "Start it with: uvicorn main:app --host 0.0.0.0 --port 8000 --reload",
            detail: err.message,
          });
        },
      },
    })
  );

  // ── Vite SPA middleware (dev) or static dist (prod) ─────────────────────────
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (_req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(
      `\n🩺  Early Prediction of Sepsis — Frontend  →  http://localhost:${PORT}`
    );
    console.log(
      `🤖  API requests proxied to FastAPI backend  →  ${FASTAPI_URL}\n`
    );
  });
}

startServer();
