/**
 * ML Dashboard — Data Visualization & Metrics Rendering
 * ======================================================
 * Fetches model comparison metrics and renders interactive charts
 * using Chart.js for accuracy, precision, recall, F1, and confusion matrices.
 */

(function () {
  let chartInstances = [];
  let metricsData = null;

  // Color scheme matching the training pipeline
  const MODEL_COLORS = {
    "Naive Bayes": "#8b5cf6",
    "Logistic Regression": "#3b82f6",
    "Random Forest": "#06b6d4",
    "Support Vector Machine": "#10b981",
  };

  // Metric colors for grouped charts
  const METRIC_COLORS = {
    precision: "#8b5cf6",
    recall: "#3b82f6",
    f1: "#10b981",
  };

  /**
   * Destroy all existing Chart.js instances to prevent conflicts
   */
  function destroyCharts() {
    chartInstances.forEach((chart) => {
      if (chart && typeof chart.destroy === "function") {
        chart.destroy();
      }
    });
    chartInstances = [];
  }

  /**
   * Get Chart.js options for consistent styling
   */
  function getChartOptions(type = "bar") {
    const isDark =
      document.documentElement.getAttribute("data-theme") !== "light";
    const textColor = isDark ? "#f0f0f5" : "#1a1a2e";
    const gridColor = isDark ? "#333355" : "#e0e0e5";
    const bgColor = isDark ? "#141432" : "#ffffff";

    const baseOptions = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: {
            color: textColor,
            font: { family: "'Inter', sans-serif", size: 12, weight: "500" },
            padding: 15,
            boxWidth: 12,
            boxPadding: 8,
          },
        },
        tooltip: {
          backgroundColor: bgColor,
          titleColor: textColor,
          bodyColor: textColor,
          borderColor: gridColor,
          borderWidth: 1,
          padding: 12,
          displayColors: true,
          titleFont: { weight: "600" },
          bodyFont: { size: 13 },
          cornerRadius: 8,
        },
      },
      scales: {
        x: {
          ticks: {
            color: textColor,
            font: { size: 11 },
            padding: 8,
          },
          grid: {
            color: gridColor,
            drawBorder: false,
          },
        },
        y: {
          beginAtZero: true,
          max: 1,
          ticks: {
            color: textColor,
            font: { size: 11 },
            callback: (value) => (value * 100).toFixed(0) + "%",
            padding: 8,
          },
          grid: {
            color: gridColor,
            drawBorder: false,
          },
        },
      },
    };

    return baseOptions;
  }

  /**
   * Create accuracy comparison bar chart
   */
  function renderAccuracyChart(data) {
    const ctx = document.getElementById("chart-accuracy");
    if (!ctx) return;

    const names = data.models.map((m) => m.name);
    const accuracies = data.models.map((m) => m.accuracy);
    const colors = names.map((n) => MODEL_COLORS[n] || "#8b5cf6");

    const options = {
      ...getChartOptions("bar"),
      plugins: {
        ...getChartOptions().plugins,
        tooltip: {
          ...getChartOptions().plugins.tooltip,
          callbacks: {
            label: (context) =>
              `Accuracy: ${(context.parsed.y * 100).toFixed(2)}%`,
          },
        },
      },
    };

    const chart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: names,
        datasets: [
          {
            label: "Accuracy",
            data: accuracies,
            backgroundColor: colors,
            borderRadius: 8,
            borderSkipped: false,
            hoverBackgroundColor: colors.map((c) =>
              adjustColorBrightness(c, 20)
            ),
          },
        ],
      },
      options,
    });

    chartInstances.push(chart);
  }

  /**
   * Create grouped metrics chart (Precision, Recall, F1)
   */
  function renderMetricsChart(data) {
    const ctx = document.getElementById("chart-metrics");
    if (!ctx) return;

    const names = data.models.map((m) => m.name);
    const metrics = ["precision", "recall", "f1"];
    const datasets = metrics.map((metric) => ({
      label: metric.charAt(0).toUpperCase() + metric.slice(1),
      data: data.models.map((m) => m[metric]),
      backgroundColor: METRIC_COLORS[metric],
      borderRadius: 6,
      hoverBackgroundColor: adjustColorBrightness(METRIC_COLORS[metric], 20),
    }));

    const options = {
      ...getChartOptions("bar"),
      plugins: {
        ...getChartOptions().plugins,
        tooltip: {
          ...getChartOptions().plugins.tooltip,
          callbacks: {
            label: (context) =>
              `${context.dataset.label}: ${(context.parsed.y * 100).toFixed(2)}%`,
          },
        },
      },
    };

    const chart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: names,
        datasets,
      },
      options,
    });

    chartInstances.push(chart);
  }

  /**
   * Create spam vs ham distribution pie chart
   */
  function renderDistributionChart(data) {
    const ctx = document.getElementById("chart-distribution");
    if (!ctx) return;

    const isDark =
      document.documentElement.getAttribute("data-theme") !== "light";
    const textColor = isDark ? "#f0f0f5" : "#1a1a2e";

    const spam = data.dataset.spam;
    const ham = data.dataset.ham;
    const total = spam + ham;
    const spamPct = ((spam / total) * 100).toFixed(1);
    const hamPct = ((ham / total) * 100).toFixed(1);

    const options = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "bottom",
          labels: {
            color: textColor,
            font: { family: "'Inter', sans-serif", size: 12, weight: "500" },
            padding: 15,
            boxWidth: 12,
            boxPadding: 8,
          },
        },
        tooltip: {
          backgroundColor: isDark ? "#141432" : "#ffffff",
          titleColor: textColor,
          bodyColor: textColor,
          borderColor: isDark ? "#333355" : "#e0e0e5",
          borderWidth: 1,
          padding: 12,
          titleFont: { weight: "600" },
          bodyFont: { size: 13 },
          cornerRadius: 8,
          callbacks: {
            label: (context) => {
              const label = context.label || "";
              const value = context.parsed || 0;
              const pct = ((value / total) * 100).toFixed(1);
              return `${label}: ${value} (${pct}%)`;
            },
          },
        },
      },
    };

    const chart = new Chart(ctx, {
      type: "doughnut",
      data: {
        labels: [`Ham (${hamPct}%)`, `Spam (${spamPct}%)`],
        datasets: [
          {
            data: [ham, spam],
            backgroundColor: ["#10b981", "#ef4444"],
            borderColor: isDark ? "#141432" : "#ffffff",
            borderWidth: 2,
            hoverOffset: 8,
          },
        ],
      },
      options,
    });

    chartInstances.push(chart);
  }

  /**
   * Render confusion matrices for all models
   */
  function renderConfusionMatrices(data) {
    const container = document.getElementById("confusion-grid");
    if (!container) return;

    container.innerHTML = "";

    data.models.forEach((model) => {
      const card = document.createElement("div");
      card.className = "confusion-matrix-card";

      const title = document.createElement("div");
      title.className = "confusion-title";
      title.textContent = `${model.name}`;

      const canvasWrap = document.createElement("div");
      canvasWrap.className = "confusion-matrix-wrap";

      const canvas = document.createElement("canvas");
      canvas.id = `confusion-${model.id}`;
      canvasWrap.appendChild(canvas);

      card.appendChild(title);
      card.appendChild(canvasWrap);
      container.appendChild(card);

      // Render confusion matrix heatmap
      renderConfusionMatrix(canvas, model);
    });
  }

  /**
   * Render individual confusion matrix heatmap
   */
  function renderConfusionMatrix(canvas, model) {
    const isDark =
      document.documentElement.getAttribute("data-theme") !== "light";
    const textColor = isDark ? "#f0f0f5" : "#1a1a2e";

    const [[tn, fp], [fn, tp]] = model.confusion_matrix;
    const total = tn + fp + fn + tp;

    // Normalize values
    const data = [
      [tn / total, fp / total],
      [fn / total, tp / total],
    ];

    const ctx = canvas.getContext("2d");
    const cellSize = 60;
    const padding = 40;
    const width = cellSize * 2 + padding * 2;
    const height = cellSize * 2 + padding * 2;

    canvas.width = width;
    canvas.height = height;

    // Draw background
    ctx.fillStyle = isDark ? "#141432" : "#ffffff";
    ctx.fillRect(0, 0, width, height);

    // Draw cells
    const colors = ["#10b981", "#3b82f6", "#f59e0b", "#ef4444"];
    const labels = [
      ["TN", "FP"],
      ["FN", "TP"],
    ];
    const values = [
      [tn, fp],
      [fn, tp],
    ];

    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        const x = padding + j * cellSize;
        const y = padding + i * cellSize;

        // Cell background
        const intensity = data[i][j];
        const color = interpolateColor(colors[(i * 2 + j) % 4], intensity);
        ctx.fillStyle = color;
        ctx.fillRect(x, y, cellSize, cellSize);

        // Cell border
        ctx.strokeStyle = isDark ? "#333355" : "#e0e0e5";
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, cellSize, cellSize);

        // Label
        ctx.fillStyle = textColor;
        ctx.font = "bold 12px 'Inter', sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillText(labels[i][j], x + cellSize / 2, y + 8);

        // Value
        ctx.font = "bold 16px 'Outfit', sans-serif";
        ctx.fillText(values[i][j], x + cellSize / 2, y + 26);
      }
    }

    // Draw axis labels
    ctx.fillStyle = textColor;
    ctx.font = "11px 'Inter', sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText("Predicted Ham", padding - 10, padding + cellSize / 2);
    ctx.fillText("Predicted Spam", padding - 10, padding + cellSize * 1.5);

    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText("Actual Ham", padding + cellSize / 2, padding - 10);
    ctx.fillText("Actual Spam", padding + cellSize * 1.5, padding - 10);
  }

  /**
   * Interpolate color based on intensity (0-1)
   */
  function interpolateColor(hexColor, intensity) {
    const r = parseInt(hexColor.slice(1, 3), 16);
    const g = parseInt(hexColor.slice(3, 5), 16);
    const b = parseInt(hexColor.slice(5, 7), 16);

    const isDark =
      document.documentElement.getAttribute("data-theme") !== "light";
    const baseColor = isDark ? [20, 20, 50] : [245, 245, 250];

    const newR = Math.round(baseColor[0] + (r - baseColor[0]) * intensity);
    const newG = Math.round(baseColor[1] + (g - baseColor[1]) * intensity);
    const newB = Math.round(baseColor[2] + (b - baseColor[2]) * intensity);

    return `rgb(${newR}, ${newG}, ${newB})`;
  }

  /**
   * Adjust color brightness
   */
  function adjustColorBrightness(hex, percent) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);

    const newR = Math.min(255, Math.round(r + (255 - r) * (percent / 100)));
    const newG = Math.min(255, Math.round(g + (255 - g) * (percent / 100)));
    const newB = Math.min(255, Math.round(b + (255 - b) * (percent / 100)));

    return `rgb(${newR}, ${newG}, ${newB})`;
  }

  /**
   * Populate highlight cards
   */
  function renderHighlights(data) {
    const best = data.highlights.best_accuracy;
    const fastest = data.highlights.fastest;
    const f1 = data.highlights.highest_f1;

    document.getElementById("hl-best-name").textContent = best.model;
    document.getElementById("hl-best-detail").textContent =
      (best.value * 100).toFixed(2) + "% Accuracy";

    document.getElementById("hl-fast-name").textContent = fastest.model;
    document.getElementById("hl-fast-detail").textContent =
      fastest.value.toFixed(4) + "s Total";

    document.getElementById("hl-f1-name").textContent = f1.model;
    document.getElementById("hl-f1-detail").textContent =
      (f1.value * 100).toFixed(2) + "% F1-Score";
  }

  /**
   * Render preprocessing pipeline steps
   */
  function renderPipelineSteps(data) {
    const container = document.getElementById("pipeline-steps");
    if (!container) return;

    container.innerHTML = "";

    data.preprocessing.forEach((step, index) => {
      const div = document.createElement("div");
      div.className = "pipeline-step";
      div.setAttribute("data-index", `${index + 1}/${data.preprocessing.length}`);
      div.textContent = step;
      container.appendChild(div);
    });
  }

  /**
   * Render model explanation
   */
  function renderExplanation(data) {
    const banner = document.getElementById("selected-model-banner");
    const text = document.getElementById("explanation-text");

    if (banner && data.selected_model) {
      banner.innerHTML = `
        <div class="selected-model-name">
          🏆 ${data.selected_model.name}
        </div>
        <div class="selected-model-accuracy">
          Accuracy: ${(data.selected_model.accuracy * 100).toFixed(2)}%
        </div>
      `;
    }

    if (text && data.selected_model) {
      text.textContent = data.selected_model.reason;
    }
  }

  /**
   * Render dataset statistics
   */
  function renderDatasetStats(data) {
    const container = document.getElementById("dataset-stats");
    if (!container) return;

    const stats = [
      {
        label: "Total Messages",
        value: data.dataset.total,
      },
      {
        label: "Spam Messages",
        value: data.dataset.spam,
      },
      {
        label: "Ham Messages",
        value: data.dataset.ham,
      },
      {
        label: "Training Set",
        value: data.dataset.train_size,
      },
      {
        label: "Test Set",
        value: data.dataset.test_size,
      },
      {
        label: "Test Split",
        value: `${(data.dataset.test_split * 100).toFixed(0)}%`,
      },
    ];

    container.innerHTML = stats
      .map(
        (stat) =>
          `<div class="stat-item">
        <span class="stat-label">${stat.label}</span>
        <span class="stat-value">${stat.value}</span>
      </div>`
      )
      .join("");
  }

  /**
   * Update dashboard metadata
   */
  function renderMetadata(data) {
    const meta = document.getElementById("dashboard-meta");
    if (!meta) return;

    const generated = new Date(data.generated_at);
    const formattedDate = generated.toLocaleDateString() + " " + generated.toLocaleTimeString();
    const modelCount = data.models.length;

    meta.innerHTML = `Generated: ${formattedDate} • ${modelCount} Models Compared • ${data.dataset.total} Messages`;
  }

  /**
   * Fetch and render complete dashboard
   */
  async function loadDashboard() {
    const loading = document.getElementById("dashboard-loading");
    const error = document.getElementById("dashboard-error");
    const content = document.getElementById("dashboard-content");

    try {
      // Resolve API base
      const apiReady = await window.SpamAPI.resolveApiBase();
      if (!apiReady) {
        throw new Error(
          "Cannot connect to backend. Ensure the API server is running."
        );
      }

      // Fetch metrics
      metricsData = await window.SpamAPI.fetchApi("/ml-metrics");

      if (!metricsData.models || metricsData.models.length === 0) {
        throw new Error("No model metrics available. Run train_model.py first.");
      }

      // Destroy old charts
      destroyCharts();

      // Render all components
      renderMetadata(metricsData);
      renderPipelineSteps(metricsData);
      renderHighlights(metricsData);
      renderAccuracyChart(metricsData);
      renderDistributionChart(metricsData);
      renderMetricsChart(metricsData);
      renderConfusionMatrices(metricsData);
      renderExplanation(metricsData);
      renderDatasetStats(metricsData);

      // Load live analytics
      await loadLiveAnalytics();

      // Show content, hide loading/error
      loading.hidden = true;
      error.hidden = true;
      content.hidden = false;

      console.log("[OK] Dashboard loaded successfully");
    } catch (err) {
      console.error("[!] Dashboard load error:", err);
      loading.hidden = true;
      error.hidden = false;
      error.querySelector("p").innerHTML =
        `<strong>${err.message}</strong><br>Visit the <a href="index.html" class="btn-link">main detector</a> while the training completes.`;
    }
  }

  /**
   * Theme change listener
   */
  function setupThemeListener() {
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (
          mutation.type === "attributes" &&
          mutation.attributeName === "data-theme"
        ) {
          if (metricsData) {
            destroyCharts();
            renderAccuracyChart(metricsData);
            renderDistributionChart(metricsData);
            renderMetricsChart(metricsData);
            renderConfusionMatrices(metricsData);
            loadLiveAnalytics(); // Re-render live charts with new theme
          }
        }
      });
    });

    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });
  }

  /**
   * Setup scroll reveal animation for .reveal elements
   */
  function setupScrollReveal() {
    const reveals = document.querySelectorAll(".reveal");

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("visible");
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.05, rootMargin: "100px 0px 100px 0px" }
    );

    reveals.forEach((el) => observer.observe(el));
    
    // Immediately make elements visible if they're already in viewport on load
    setTimeout(() => {
      reveals.forEach((el) => {
        const rect = el.getBoundingClientRect();
        if (rect.top < window.innerHeight && rect.bottom > 0) {
          el.classList.add("visible");
        }
      });
    }, 100);
  }

  /**
   * Load Live Analytics (Trends, Stats, Confidence, History)
   */
  async function loadLiveAnalytics() {
    try {
      // 1. Stats
      const stats = await window.SpamAPI.fetchApi("/api/dashboard/stats");
      document.getElementById("live-stat-total").textContent = stats.total_predictions.toLocaleString();
      document.getElementById("live-stat-spam").textContent = stats.spam_predictions.toLocaleString();
      document.getElementById("live-stat-ham").textContent = stats.ham_predictions.toLocaleString();

      // 2. Trends Chart
      const trends = await window.SpamAPI.fetchApi("/api/dashboard/trends");
      renderTrendsChart(trends);

      // 3. Confidence Chart
      const conf = await window.SpamAPI.fetchApi("/api/dashboard/confidence");
      renderConfidenceChart(conf);

      // 4. Keywords Chart
      const keywords = await window.SpamAPI.fetchApi("/api/dashboard/keywords");
      renderKeywordsChart(keywords);

      // 5. History Feed
      const history = await window.SpamAPI.fetchApi("/api/dashboard/history");
      renderHistoryFeed(history);
    } catch (e) {
      console.warn("Live analytics load error:", e);
    }
  }

  function renderTrendsChart(data) {
    const ctx = document.getElementById("chart-trends");
    if (!ctx) return;
    const isDark = document.documentElement.getAttribute("data-theme") !== "light";
    const chart = new Chart(ctx, {
      type: "line",
      data: {
        labels: data.dates,
        datasets: [
          {
            label: "Spam",
            data: data.spam,
            borderColor: "#ef4444",
            backgroundColor: "rgba(239, 68, 68, 0.1)",
            fill: true,
            tension: 0.4
          },
          {
            label: "Safe (Ham)",
            data: data.ham,
            borderColor: "#10b981",
            backgroundColor: "rgba(16, 185, 129, 0.1)",
            fill: true,
            tension: 0.4
          }
        ]
      },
      options: {
        ...getChartOptions("line"),
        scales: {
          x: { ...getChartOptions("line").scales.x },
          y: { ...getChartOptions("line").scales.y, max: null } // auto-scale y
        }
      }
    });
    chartInstances.push(chart);
  }

  function renderConfidenceChart(data) {
    const ctx = document.getElementById("chart-confidence");
    if (!ctx) return;
    const labels = Object.keys(data.distribution);
    const values = Object.values(data.distribution);
    
    const chart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [{
          label: "Predictions Count",
          data: values,
          backgroundColor: "#3b82f6",
          borderRadius: 4
        }]
      },
      options: {
        ...getChartOptions("bar"),
        scales: {
          x: { ...getChartOptions("bar").scales.x },
          y: { ...getChartOptions("bar").scales.y, max: null }
        }
      }
    });
    chartInstances.push(chart);
  }

  function renderKeywordsChart(data) {
    const ctx = document.getElementById("chart-keywords");
    if (!ctx) return;
    const labels = data.top_keywords.map(k => k.keyword);
    const values = data.top_keywords.map(k => k.count);

    const chart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [{
          label: "Frequency",
          data: values,
          backgroundColor: "#8b5cf6",
          borderRadius: 4
        }]
      },
      options: {
        ...getChartOptions("bar"),
        indexAxis: 'y', // horizontal bar chart
        scales: {
          x: { ...getChartOptions("bar").scales.x },
          y: { ...getChartOptions("bar").scales.y, max: null }
        }
      }
    });
    chartInstances.push(chart);
  }

  function renderHistoryFeed(history) {
    const feed = document.getElementById("history-feed");
    if (!feed) return;
    feed.innerHTML = history.map(item => {
      const isSpam = item.label === "spam";
      const icon = isSpam ? "🚫" : "✅";
      const color = isSpam ? "var(--danger)" : "var(--success)";
      const time = new Date(item.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
      return `
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px; padding-bottom: 15px; border-bottom: 1px solid var(--border);">
          <div style="font-size: 1.5rem;">${icon}</div>
          <div style="flex: 1; min-width: 0;">
            <div style="font-size: 0.9rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${item.message.replace(/</g, "&lt;")}</div>
            <div style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 4px;">
              <span style="color: ${color}; font-weight: 600;">${item.confidence}%</span> • ${time}
            </div>
          </div>
        </div>
      `;
    }).join("");
  }

  /**
   * Initialize dashboard on DOM ready
   */
  document.addEventListener("DOMContentLoaded",  () => {
    setupScrollReveal();
    loadDashboard();
    setupThemeListener();
    
    // Auto-refresh live analytics every 30 seconds
    setInterval(loadLiveAnalytics, 30000);
  });

  // Reload dashboard when user presses refresh
  window.addEventListener("beforeunload", destroyCharts);
})();
