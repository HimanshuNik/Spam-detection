/**
 * SpamShield AI — Core Application Logic
 * ========================================
 * Handles API communication, DOM updates, toast notifications,
 * history/stats rendering, and user interactions.
 */

// API client loaded from api.js (SpamAPI)

// ---------------------------------------------------------------------------
// DOM helpers
// ---------------------------------------------------------------------------
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

let messageInput;
let charCount;
let btnDetect;
let btnText;
let btnLoading;
let resultCard;
let verdictEl;
let verdictIcon;
let verdictLabel;
let verdictDesc;
let confidenceVal;
let confidenceFill;
let keywordsWrap;
let keywordsList;
let reasonWrap;
let reasonText;
let historyTbody;
let historyEmpty;
let historyCount;
let btnExport;
let btnClear;
let btnCopy;
let btnSampleSpam;
let btnSampleHam;
let lastResult = null;

// ---------------------------------------------------------------------------
// Toast Notification System
// ---------------------------------------------------------------------------
function showToast(message, type = "info") {
  const container = $("#toast-container");
  const icons = { success: "✅", error: "❌", info: "ℹ️" };

  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  const iconSpan = document.createElement("span");
  iconSpan.className = "toast-icon";
  iconSpan.textContent = icons[type] || icons.info;

  const msgSpan = document.createElement("span");
  msgSpan.textContent = message;

  toast.appendChild(iconSpan);
  toast.appendChild(msgSpan);
  container.appendChild(toast);

  setTimeout(() => {
    toast.classList.add("removing");
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

const fetchApi = (...args) => SpamAPI.fetchApi(...args);

// ---------------------------------------------------------------------------
// Character Counter
// ---------------------------------------------------------------------------
function bindCharacterCounter() {
  messageInput.addEventListener("input", () => {
    const len = messageInput.value.length;
    charCount.textContent = `${len.toLocaleString()} / 10,000`;
    charCount.style.color = len > 9500 ? "var(--danger)" : "";
  });
}

// ---------------------------------------------------------------------------
// Detect Spam
// ---------------------------------------------------------------------------
function bindDetectButton() {
  btnDetect.addEventListener("click", async () => {
    const message = messageInput.value.trim();

    if (!message) {
      showToast("Please enter a message to analyze.", "error");
      messageInput.focus();
      return;
    }

    btnText.style.display = "none";
    btnLoading.style.display = "inline-flex";
    btnDetect.disabled = true;
    hideResult();

    try {
      const data = await fetchApi("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });
      lastResult = data;
      displayResult(data);
      refreshHistory();
      refreshStats();
      showToast(
        data.label === "spam"
          ? "Spam detected! Be careful with this message."
          : "This message looks legitimate.",
        data.label === "spam" ? "error" : "success",
      );
    } catch (err) {
      showToast(err.message || "Failed to analyze message.", "error");
      console.error("Prediction error:", err);
    } finally {
      btnText.style.display = "inline-flex";
      btnLoading.style.display = "none";
      btnDetect.disabled = false;
    }
  });
}

function hideResult() {
  resultCard.classList.remove("is-visible");
  resultCard.hidden = true;
}

// ---------------------------------------------------------------------------
// Display Result
// ---------------------------------------------------------------------------
function displayResult(data) {
  const isSpam = data.label === "spam";

  resultCard.hidden = false;
  resultCard.classList.add("is-visible");
  resultCard.style.opacity = "1";
  resultCard.style.transform = "none";

  verdictEl.className = `result-verdict ${data.label}`;
  verdictIcon.textContent = isSpam ? "🚫" : "✅";
  verdictLabel.textContent = isSpam ? "Spam Detected" : "Not Spam";
  verdictDesc.textContent = isSpam
    ? "This message shows characteristics of spam or phishing content."
    : "This message appears to be legitimate and safe.";

  confidenceVal.textContent = `${data.confidence}%`;
  confidenceFill.className = `confidence-fill ${data.label}`;
  confidenceFill.style.width = "0%";
  requestAnimationFrame(() => {
    setTimeout(() => {
      confidenceFill.style.width = `${data.confidence}%`;
    }, 100);
  });

  if (data.keywords && data.keywords.length > 0) {
    keywordsWrap.style.display = "block";
    keywordsList.innerHTML = data.keywords
      .map(
        (kw, i) =>
          `<span class="keyword-tag" style="animation-delay:${i * 0.05}s">${escapeHtml(kw)}</span>`,
      )
      .join("");
  } else {
    keywordsWrap.style.display = "none";
  }

  if (data.reason) {
    reasonWrap.style.display = "block";
    reasonText.textContent = data.reason;
  } else {
    reasonWrap.style.display = "none";
  }

  resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ---------------------------------------------------------------------------
// Copy Result
// ---------------------------------------------------------------------------
function bindCopyButton() {
  btnCopy.addEventListener("click", () => {
    if (!lastResult) return;

    const text = [
      `Result: ${lastResult.label === "spam" ? "SPAM" : "NOT SPAM"}`,
      `Confidence: ${lastResult.confidence}%`,
      lastResult.keywords?.length
        ? `Suspicious keywords: ${lastResult.keywords.join(", ")}`
        : "",
      `Message: ${lastResult.message}`,
    ]
      .filter(Boolean)
      .join("\n");

    navigator.clipboard
      .writeText(text)
      .then(() => showToast("Result copied to clipboard!", "success"))
      .catch(() => showToast("Failed to copy result.", "error"));
  });
}

// ---------------------------------------------------------------------------
// Sample Messages
// ---------------------------------------------------------------------------
async function loadSamples() {
  try {
    const samples = await fetchApi("/samples");

    btnSampleSpam.addEventListener("click", () => {
      const idx = Math.floor(Math.random() * samples.spam.length);
      messageInput.value = samples.spam[idx];
      messageInput.dispatchEvent(new Event("input"));
      showToast("Sample spam message loaded.", "info");
    });

    btnSampleHam.addEventListener("click", () => {
      const idx = Math.floor(Math.random() * samples.ham.length);
      messageInput.value = samples.ham[idx];
      messageInput.dispatchEvent(new Event("input"));
      showToast("Sample legitimate message loaded.", "info");
    });
  } catch {
    btnSampleSpam.addEventListener("click", () => {
      messageInput.value =
        "WINNER!! You have been selected to receive a prize reward! Call now to claim!";
      messageInput.dispatchEvent(new Event("input"));
    });
    btnSampleHam.addEventListener("click", () => {
      messageInput.value =
        "Hey, are you coming to the party tonight? Let me know so I can pick you up.";
      messageInput.dispatchEvent(new Event("input"));
    });
  }
}

// ---------------------------------------------------------------------------
// History
// ---------------------------------------------------------------------------
async function refreshHistory() {
  try {
    const history = await fetchApi("/history");

    if (!Array.isArray(history) || history.length === 0) {
      historyTbody.innerHTML = "";
      historyEmpty.style.display = "block";
      historyCount.textContent = "0 predictions";
      return;
    }

    historyEmpty.style.display = "none";
    historyCount.textContent = `${history.length} prediction${history.length !== 1 ? "s" : ""}`;

    historyTbody.innerHTML = history
      .map((item) => {
        const preview =
          item.message.length > 80
            ? item.message.substring(0, 80) + "…"
            : item.message;
        const isSpam = item.label === "spam";
        const time = formatTime(item.timestamp);

        return `
                <tr>
                    <td><span class="msg-preview" title="${escapeHtml(item.message)}">${escapeHtml(preview)}</span></td>
                    <td><span class="label-badge ${item.label}">${isSpam ? "Spam" : "Ham"}</span></td>
                    <td>${item.confidence}%</td>
                    <td><span class="history-time">${time}</span></td>
                </tr>
            `;
      })
      .join("");
  } catch (err) {
    console.error("Failed to load history:", err);
  }
}

function bindHistoryButtons() {
  btnExport.addEventListener("click", () => {
    window.location.href = `${SpamAPI.base}/export`;
    showToast("Downloading history as CSV…", "info");
  });

  btnClear.addEventListener("click", async () => {
    if (!confirm("Are you sure you want to clear all prediction history?")) return;

    try {
      await fetchApi("/history", { method: "DELETE" });
      refreshHistory();
      refreshStats();
      showToast("History cleared.", "success");
    } catch {
      showToast("Failed to clear history.", "error");
    }
  });
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------
async function refreshStats() {
  try {
    const stats = await fetchApi("/stats");

    animateCounter("stat-total", stats.total);
    animateCounter("stat-spam", stats.spam);
    animateCounter("stat-ham", stats.ham);
    animateCounter("stat-accuracy", stats.accuracy, "%");
  } catch (err) {
    console.error("Failed to load stats:", err);
  }
}

function animateCounter(id, target, suffix = "") {
  const el = document.getElementById(id);
  if (!el) return;

  const start = parseInt(el.textContent) || 0;
  const end = typeof target === "number" ? target : parseFloat(target);
  const duration = 800;
  const startTime = performance.now();

  function step(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = Math.round(start + (end - start) * eased);

    el.textContent = suffix ? `${current}${suffix}` : current;

    if (progress < 1) {
      requestAnimationFrame(step);
    }
  }

  requestAnimationFrame(step);
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------
function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function formatTime(timestamp) {
  if (!timestamp) return "—";
  const date = new Date(timestamp + "Z");
  const now = new Date();
  const diff = now - date;

  if (diff < 60000) return "Just now";
  if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
  return date.toLocaleDateString();
}

// ---------------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------------
function updateActiveNav() {
  const sections = ["stats", "history", "detect", "hero"];
  const scrollY = window.scrollY + 150;

  for (const id of sections) {
    const section = document.getElementById(id);
    if (section && section.offsetTop <= scrollY) {
      $$(".nav-link").forEach((link) => {
        link.classList.toggle("active", link.getAttribute("href") === `#${id}`);
      });
      break;
    }
  }
}

function bindNavigation() {
  window.addEventListener("scroll", updateActiveNav);

  window.addEventListener("scroll", () => {
    const navbar = $("#navbar");
    if (window.scrollY > 50) {
      navbar.classList.add("scrolled");
    } else {
      navbar.classList.remove("scrolled");
    }
  });

  const mobileMenuBtn = $("#mobile-menu-btn");
  const navLinks = $("#nav-links");

  mobileMenuBtn.addEventListener("click", () => {
    mobileMenuBtn.classList.toggle("active");
    navLinks.classList.toggle("open");
  });

  $$(".nav-link").forEach((link) => {
    link.addEventListener("click", () => {
      mobileMenuBtn.classList.remove("active");
      navLinks.classList.remove("open");
    });
  });
}

function bindKeyboardShortcut() {
  messageInput.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      btnDetect.click();
    }
  });
}

// ---------------------------------------------------------------------------
// Initialisation
// ---------------------------------------------------------------------------
function initDom() {
  messageInput = $("#message-input");
  charCount = $("#char-count");
  btnDetect = $("#btn-detect");
  btnText = btnDetect.querySelector(".btn-text");
  btnLoading = btnDetect.querySelector(".btn-loading");
  resultCard = $("#result-card");
  verdictEl = $("#result-verdict");
  verdictIcon = $("#verdict-icon");
  verdictLabel = $("#verdict-label");
  verdictDesc = $("#verdict-desc");
  confidenceVal = $("#confidence-value");
  confidenceFill = $("#confidence-fill");
  keywordsWrap = $("#result-keywords");
  keywordsList = $("#keywords-list");
  reasonWrap = $("#result-reason");
  reasonText = $("#reason-text");
  historyTbody = $("#history-tbody");
  historyEmpty = $("#history-empty");
  historyCount = $("#history-count");
  btnExport = $("#btn-export");
  btnClear = $("#btn-clear");
  btnCopy = $("#btn-copy-result");
  btnSampleSpam = $("#btn-sample-spam");
  btnSampleHam = $("#btn-sample-ham");

  bindCharacterCounter();
  bindDetectButton();
  bindCopyButton();
  bindHistoryButtons();
  bindNavigation();
  bindKeyboardShortcut();
}

document.addEventListener("DOMContentLoaded", async () => {
  initDom();
  const apiReady = await SpamAPI.resolveApiBase();
  if (!apiReady) {
    showToast(
      "Backend not running. Double-click start.bat or run: python backend/app.py",
      "error",
    );
    return;
  }
  loadSamples();
  refreshHistory();
  refreshStats();
});
