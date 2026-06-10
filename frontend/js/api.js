/**
 * Shared API client — used by app.js and dashboard.js
 */
(function () {
  const API_PORT = "5000";
  const BACKEND_CANDIDATES = [
    "http://127.0.0.1:5000",
    "http://localhost:5000",
  ];

  const isServedByBackend =
    window.location.port === API_PORT &&
    window.location.protocol.startsWith("http");

  let API_BASE = isServedByBackend
    ? window.location.origin
    : BACKEND_CANDIDATES[0];

  async function probeApi(base) {
    try {
      const res = await fetch(`${base}/health`, { mode: "cors" });
      if (!res.ok) return false;
      const data = await res.json();
      return data && data.status === "ok";
    } catch {
      return false;
    }
  }

  async function resolveApiBase() {
    if (isServedByBackend) {
      API_BASE = window.location.origin;
      return true;
    }
    for (const base of BACKEND_CANDIDATES) {
      if (await probeApi(base)) {
        API_BASE = base;
        return true;
      }
    }
    return false;
  }

  async function parseJsonResponse(response) {
    const contentType = response.headers.get("content-type") || "";
    const text = await response.text();
    if (!text) return {};
    const trimmed = text.trim();
    const looksLikeJson =
      trimmed.startsWith("{") ||
      trimmed.startsWith("[") ||
      contentType.includes("application/json");
    if (looksLikeJson) {
      try {
        return JSON.parse(text);
      } catch (err) {
        return { error: `Invalid JSON response: ${err.message}`, raw: text };
      }
    }
    return { error: text };
  }

  async function fetchApi(path, options = {}) {
    let res;
    try {
      res = await fetch(`${API_BASE}${path}`, { ...options, mode: "cors" });
    } catch {
      throw new Error(
        isServedByBackend
          ? "Network error — could not reach the server."
          : `Cannot connect to API at ${BACKEND_CANDIDATES[0]}. Start the backend first (double-click start.bat or run python backend/app.py).`,
      );
    }
    const payload = await parseJsonResponse(res);
    if (!res.ok) {
      throw new Error(
        payload.error ||
          payload.message ||
          `API request failed: ${res.status} ${res.statusText}`,
      );
    }
    return payload;
  }

  window.SpamAPI = {
    get base() {
      return API_BASE;
    },
    get isServedByBackend() {
      return isServedByBackend;
    },
    resolveApiBase,
    fetchApi,
    probeApi,
  };
})();
