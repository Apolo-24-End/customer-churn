const CHART_DEFAULTS = {
  color: {
    accent:  "#D91023",
    danger:  "#D91023",
    success: "#34D399",
    warning: "#C89B3C",
    navy:    "#1B365D",
    steel:   "#4A779E",
    slate:   "#64748B",
    teal:    "#0F766E",
    gold:    "#C89B3C",
    grid:    "rgba(226, 229, 234, 0.8)",
    text:    "#64748B",
  },
  // Ordered palette for multi-segment charts (pie, donut, multi-line, ROC)
  palette: ["#1B365D", "#4A779E", "#D91023", "#0F766E", "#C89B3C", "#64748B"],
  font: { family: "Inter, -apple-system, sans-serif", size: 11 },
};

Chart.defaults.color = CHART_DEFAULTS.color.text;
Chart.defaults.font.family = CHART_DEFAULTS.font.family;
Chart.defaults.font.size = CHART_DEFAULTS.font.size;

const activeCharts = {};

function destroyChart(id) {
  if (activeCharts[id]) { activeCharts[id].destroy(); delete activeCharts[id]; }
}

function gradientBar(ctx, color1, color2) {
  const g = ctx.createLinearGradient(0, 0, ctx.canvas.width, 0);
  g.addColorStop(0, color1);
  g.addColorStop(1, color2);
  return g;
}

function renderDonut(id, labels, values) {
  destroyChart(id);
  const ctx = document.getElementById(id).getContext("2d");
  const bg = labels.length <= 2
    ? [CHART_DEFAULTS.color.success, CHART_DEFAULTS.color.danger]
    : CHART_DEFAULTS.palette.slice(0, labels.length);
  activeCharts[id] = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: bg,
        borderWidth: 0,
        hoverOffset: 8,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "70%",
      plugins: {
        legend: { position: "bottom", labels: { padding: 16, usePointStyle: true } },
        tooltip: { callbacks: { label: (c) => ` ${c.label}: ${c.raw.toLocaleString()}` } },
      },
    },
  });
}

function renderBar(id, labels, values, label = "", color = CHART_DEFAULTS.color.accent) {
  destroyChart(id);
  const ctx = document.getElementById(id).getContext("2d");
  activeCharts[id] = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{ label, data: values, backgroundColor: color + "cc", borderColor: color, borderWidth: 1, borderRadius: 4 }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: CHART_DEFAULTS.color.grid }, ticks: { maxRotation: 30 } },
        y: { grid: { color: CHART_DEFAULTS.color.grid }, beginAtZero: true },
      },
    },
  });
}

function renderHorizontalBar(id, labels, values) {
  destroyChart(id);
  const ctx = document.getElementById(id).getContext("2d");
  const g = ctx.createLinearGradient(0, 0, ctx.canvas.width || 400, 0);
  g.addColorStop(0, CHART_DEFAULTS.color.navy);
  g.addColorStop(1, CHART_DEFAULTS.color.steel);
  activeCharts[id] = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{ data: values, backgroundColor: g, borderRadius: 4, borderSkipped: false }],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: CHART_DEFAULTS.color.grid }, beginAtZero: true },
        y: { grid: { display: false }, ticks: { font: { size: 11 } } },
      },
    },
  });
}

function renderLine(id, labels, datasets) {
  destroyChart(id);
  const ctx = document.getElementById(id).getContext("2d");
  const colors = CHART_DEFAULTS.palette;
  activeCharts[id] = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: datasets.map((d, i) => ({
        label: d.label,
        data: d.data,
        borderColor: colors[i],
        backgroundColor: colors[i] + "18",
        borderWidth: 2,
        pointRadius: 0,
        fill: true,
        tension: 0.4,
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { position: "bottom", labels: { usePointStyle: true, padding: 16 } },
      },
      scales: {
        x: { grid: { color: CHART_DEFAULTS.color.grid } },
        y: { grid: { color: CHART_DEFAULTS.color.grid }, min: 0, max: 1 },
      },
    },
  });
}

function renderHistogram(id, labels, values) {
  renderBar(id, labels, values, "Count", CHART_DEFAULTS.color.accent);
}

function renderROC(id, datasets) {
  destroyChart(id);
  const ctx = document.getElementById(id).getContext("2d");
  const palette = CHART_DEFAULTS.palette;

  const diag = { label: "Random (AUC=0.5)", data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], borderColor: CHART_DEFAULTS.color.grid, borderWidth: 1, borderDash: [4, 4], pointRadius: 0, fill: false, tension: 0 };

  activeCharts[id] = new Chart(ctx, {
    type: "line",
    data: {
      datasets: [
        ...datasets.map((d, i) => ({
          label: d.label,
          data: d.data,
          borderColor: palette[i % palette.length],
          backgroundColor: palette[i % palette.length] + "18",
          borderWidth: 2,
          pointRadius: 0,
          fill: false,
          tension: 0.2,
        })),
        diag,
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "nearest", intersect: false },
      plugins: {
        legend: {
          position: "bottom",
          labels: { usePointStyle: true, padding: 16, filter: item => item.text !== "Random (AUC=0.5)" || true },
        },
        tooltip: {
          callbacks: {
            title: (items) => `FPR: ${items[0].raw.x.toFixed(3)}`,
            label: (item) => ` ${item.dataset.label}: TPR ${item.raw.y.toFixed(3)}`,
          },
        },
      },
      scales: {
        x: {
          type: "linear",
          title: { display: true, text: "False Positive Rate", color: CHART_DEFAULTS.color.text, font: { size: 11 } },
          grid: { color: CHART_DEFAULTS.color.grid },
          min: 0, max: 1,
          ticks: { maxTicksLimit: 6 },
        },
        y: {
          type: "linear",
          title: { display: true, text: "True Positive Rate", color: CHART_DEFAULTS.color.text, font: { size: 11 } },
          grid: { color: CHART_DEFAULTS.color.grid },
          min: 0, max: 1,
          ticks: { maxTicksLimit: 6 },
        },
      },
    },
  });
}

function renderCorrelationBar(id, features, correlations) {
  destroyChart(id);
  const ctx = document.getElementById(id).getContext("2d");
  const colors = correlations.map(v => v >= 0 ? CHART_DEFAULTS.color.danger + "cc" : CHART_DEFAULTS.color.accent + "cc");
  activeCharts[id] = new Chart(ctx, {
    type: "bar",
    data: {
      labels: features,
      datasets: [{ data: correlations, backgroundColor: colors, borderRadius: 4 }],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: CHART_DEFAULTS.color.grid } },
        y: { grid: { display: false }, ticks: { font: { size: 10 } } },
      },
    },
  });
}
