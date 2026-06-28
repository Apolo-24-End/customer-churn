// ── Navigation ──────────────────────────────────────────────
function navigate(pageId) {
  document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
  document.querySelectorAll(".nav-item").forEach(n => n.classList.remove("active"));
  document.getElementById(`page-${pageId}`)?.classList.add("active");
  document.querySelector(`[data-page="${pageId}"]`)?.classList.add("active");
  PAGE_LOADERS[pageId]?.();
}

// ── Utilities ────────────────────────────────────────────────
function fmt(n) { return typeof n === "number" ? n.toLocaleString() : n; }
function fmtPct(n) { return `${n}%`; }
function loading(el) { el.innerHTML = `<div class="loading"><div class="spinner"></div> Loading data...</div>`; }

function riskBadge(level) {
  const cls = { High: "badge-high", Medium: "badge-medium", Low: "badge-low" }[level] || "badge-low";
  return `<span class="badge ${cls}">${level}</span>`;
}

function decileBadge(d) {
  const cls = d >= 9 ? "badge-high" : d >= 6 ? "badge-medium" : "badge-low";
  return `<span class="badge ${cls}">D${d}</span>`;
}

function errorBanner(message) {
  const clean = message.replace(/</g, "&lt;").replace(/>/g, "&gt;");
  return `
    <div class="error-banner">
      <div class="error-banner-icon">⚠</div>
      <div>
        <div class="error-banner-title">Error al cargar datos</div>
        <div class="error-banner-msg">${clean}</div>
      </div>
    </div>`;
}

function downsampleROC(fpr, tpr, nPoints = 250) {
  if (fpr.length <= nPoints) return { fpr, tpr };
  const step = Math.ceil(fpr.length / nPoints);
  const sf = [], st = [];
  for (let i = 0; i < fpr.length; i += step) { sf.push(fpr[i]); st.push(tpr[i]); }
  sf.push(fpr[fpr.length - 1]); st.push(tpr[tpr.length - 1]);
  return { fpr: sf, tpr: st };
}

// ── Page: Overview ──────────────────────────────────────────
async function loadOverview() {
  try {
    const [eda, eval_] = await Promise.all([API.eda.all(), API.model.results()]);
    const ov = eda.overview;
    const best = eval_.best_model;
    const bm = eval_[best];

    document.getElementById("kpi-total").textContent = fmt(ov.total_customers);
    document.getElementById("kpi-churn").textContent = fmt(ov.churn_count);
    document.getElementById("kpi-rate").textContent = fmtPct(ov.churn_rate);
    document.getElementById("kpi-auc").textContent = bm.auc;
    document.getElementById("kpi-best").textContent = best.replace("_", " ").toUpperCase().replace("LIGHTGBM", "LightGBM").replace("XGBOOST", "XGBoost").replace("LOGISTIC REGRESSION", "Log. Reg.");

    renderDonut("chart-churn-dist", eda.churn_distribution.labels, eda.churn_distribution.values);
    renderBar("chart-contract", eda.churn_by_contract.labels, eda.churn_by_contract.churn_rates, "Churn Rate %", "#58a6ff");
    renderBar("chart-tenure", eda.churn_by_tenure_group.labels, eda.churn_by_tenure_group.churn_rates, "Churn Rate %", "#bc8cff");
    renderBar("chart-satisfaction", ["No Churn", "Churn"], [eda.avg_satisfaction_by_churn.values[0], eda.avg_satisfaction_by_churn.values[1]], "Avg Satisfaction", "#3fb950");
  } catch (e) {
    console.error(e);
    document.getElementById("overview-error").innerHTML = errorBanner(e.message);
  }
}

// ── Page: EDA ───────────────────────────────────────────────
async function loadEDA() {
  const container = document.getElementById("eda-content");
  loading(container);
  try {
    const eda = await API.eda.all();
    container.innerHTML = buildEDAHTML();

    renderDonut("eda-churn-donut", eda.churn_distribution.labels, eda.churn_distribution.values);
    renderHistogram("eda-age-hist", eda.age_distribution.labels, eda.age_distribution.values);
    renderHistogram("eda-income-hist", eda.income_distribution.labels, eda.income_distribution.values);
    renderHistogram("eda-charges-hist", eda.monthly_charges_distribution.labels, eda.monthly_charges_distribution.values);
    renderBar("eda-contract-bar", eda.churn_by_contract.labels, eda.churn_by_contract.churn_rates, "Churn %", "#f85149");
    renderBar("eda-gender-bar", eda.churn_by_gender.labels, eda.churn_by_gender.churn_rates, "Churn %", "#58a6ff");
    renderBar("eda-payment-bar", eda.churn_by_payment.labels, eda.churn_by_payment.churn_rates, "Churn %", "#bc8cff");
    renderBar("eda-tenure-bar", eda.churn_by_tenure_group.labels, eda.churn_by_tenure_group.churn_rates, "Churn %", "#e3b341");
    renderCorrelationBar("eda-corr-bar", eda.correlation_with_churn.features.slice(0, 12), eda.correlation_with_churn.correlations.slice(0, 12));
  } catch (e) {
    container.innerHTML = errorBanner(e.message);
  }
}

function buildEDAHTML() {
  return `
    <div class="charts-grid">
      <div class="card"><div class="card-title">Churn Distribution</div><div class="chart-container"><canvas id="eda-churn-donut"></canvas></div></div>
      <div class="card"><div class="card-title">Age Distribution</div><div class="chart-container"><canvas id="eda-age-hist"></canvas></div></div>
      <div class="card"><div class="card-title">Annual Income Distribution</div><div class="chart-container"><canvas id="eda-income-hist"></canvas></div></div>
      <div class="card"><div class="card-title">Monthly Charges Distribution</div><div class="chart-container"><canvas id="eda-charges-hist"></canvas></div></div>
      <div class="card"><div class="card-title">Churn Rate by Contract Type</div><div class="chart-container"><canvas id="eda-contract-bar"></canvas></div></div>
      <div class="card"><div class="card-title">Churn Rate by Gender</div><div class="chart-container"><canvas id="eda-gender-bar"></canvas></div></div>
      <div class="card"><div class="card-title">Churn Rate by Payment Method</div><div class="chart-container"><canvas id="eda-payment-bar"></canvas></div></div>
      <div class="card"><div class="card-title">Churn Rate by Tenure Group</div><div class="chart-container"><canvas id="eda-tenure-bar"></canvas></div></div>
    </div>
    <div class="card"><div class="card-title">Feature Correlation with Churn</div><div class="chart-container" style="height:340px"><canvas id="eda-corr-bar"></canvas></div></div>
  `;
}

// ── Page: Models ─────────────────────────────────────────────
async function loadModels() {
  const container = document.getElementById("models-content");
  loading(container);
  try {
    const [cmp, roc, feats] = await Promise.all([
      API.model.comparison(),
      API.model.rocCurves(),
      API.model.features(),
    ]);

    container.innerHTML = buildModelsHTML(cmp, feats);

    // ROC curves — each model uses its own {x,y} coordinates to avoid misalignment
    const rocDatasets = Object.entries(roc).map(([name, curve]) => {
      const s = downsampleROC(curve.fpr, curve.tpr);
      return { label: name.replace(/_/g, " "), data: s.fpr.map((x, i) => ({ x, y: s.tpr[i] })) };
    });
    renderROC("roc-chart", rocDatasets);

    // Feature importance table
    renderFeatureTable("fi-table-body", feats.features.slice(0, 15), feats.importances.slice(0, 15));

    // Confusion matrix for best model
    const best = cmp.best_model;
    const cm = cmp.metrics[best]?.confusion_matrix;
    if (cm) renderConfusionMatrix("cm-grid", cm);

  } catch (e) {
    container.innerHTML = errorBanner(e.message);
  }
}

function buildModelsHTML(cmp, feats) {
  const models = cmp.models;
  const best = cmp.best_model;
  const rows = models.map(m => {
    const d = cmp.metrics[m] || {};
    const isBest = m === best;
    const modelLabel = m.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
    return `<tr class="${isBest ? "best-row" : ""}">
      <td>${modelLabel} ${isBest ? '<span class="badge badge-best">Best</span>' : ""}</td>
      <td><strong>${d.auc || "-"}</strong></td>
      <td>${d.f1 || "-"}</td>
      <td>${d.precision || "-"}</td>
      <td>${d.recall || "-"}</td>
      <td>${d.accuracy || "-"}</td>
      <td>${d.cv_auc_mean ? `${d.cv_auc_mean} ± ${d.cv_auc_std}` : "-"}</td>
    </tr>`;
  }).join("");

  const bestLabel = best.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());

  return `
    <div class="card">
      <div class="card-title">Model Comparison</div>
      <div class="table-container">
        <table class="metrics-table">
          <thead><tr><th>Model</th><th>AUC-ROC</th><th>F1</th><th>Precision</th><th>Recall</th><th>Accuracy</th><th>CV AUC (5-fold)</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </div>
    <div class="models-row">
      <div class="card">
        <div class="card-title">ROC Curves — All Models</div>
        <div class="chart-container" style="height:300px"><canvas id="roc-chart"></canvas></div>
      </div>
      <div class="card">
        <div class="card-title">Confusion Matrix — ${bestLabel}</div>
        <div id="cm-grid" class="confusion-grid"></div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">Feature Importance (Top 15) — Feature Selector (RF)</div>
      <div class="table-container">
        <table class="fi-table">
          <thead><tr><th>#</th><th>Feature</th><th style="min-width:180px">Importance</th><th>Score</th></tr></thead>
          <tbody id="fi-table-body"></tbody>
        </table>
      </div>
    </div>
  `;
}

function renderFeatureTable(tbodyId, features, importances) {
  const max = Math.max(...importances);
  const tbody = document.getElementById(tbodyId);
  tbody.innerHTML = features.map((f, i) => {
    const pct = ((importances[i] / max) * 100).toFixed(1);
    const score = (importances[i] * 100).toFixed(3);
    const label = f.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
    return `<tr>
      <td class="fi-rank">${i + 1}</td>
      <td>${label}</td>
      <td class="fi-bar-cell"><div class="fi-track"><div class="fi-fill" style="width:${pct}%"></div></div></td>
      <td class="fi-pct">${score}%</td>
    </tr>`;
  }).join("");
}

function renderConfusionMatrix(containerId, cm) {
  const [[tn, fp], [fn, tp]] = cm;
  document.getElementById(containerId).innerHTML = `
    <div class="cm-cell tn"><div class="cm-count">${fmt(tn)}</div><div class="cm-label">True Neg.</div></div>
    <div class="cm-cell fp"><div class="cm-count">${fmt(fp)}</div><div class="cm-label">False Pos.</div></div>
    <div class="cm-cell fn"><div class="cm-count">${fmt(fn)}</div><div class="cm-label">False Neg.</div></div>
    <div class="cm-cell tp"><div class="cm-count">${fmt(tp)}</div><div class="cm-label">True Pos.</div></div>
  `;
}

// ── Page: Predict ────────────────────────────────────────────
async function runPrediction() {
  const form = document.getElementById("predict-form");
  const resultEl = document.getElementById("predict-result");
  const btn = document.getElementById("predict-btn");

  const data = {
    age: +form.age.value,
    gender: form.gender.value,
    annual_income: +form.annual_income.value,
    education: form.education.value,
    marital_status: form.marital_status.value,
    dependents: +form.dependents.value,
    tenure: +form.tenure.value,
    contract: form.contract.value,
    payment_method: form.payment_method.value,
    paperless_billing: form.paperless_billing.value,
    senior_citizen: +form.senior_citizen.value,
    monthlycharges: +form.monthlycharges.value,
    totalcharges: +form.totalcharges.value,
    num_services: +form.num_services.value,
    has_phone_service: +form.has_phone_service.value,
    has_internet_service: +form.has_internet_service.value,
    has_online_security: +form.has_online_security.value,
    has_online_backup: +form.has_online_backup.value,
    has_device_protection: +form.has_device_protection.value,
    has_tech_support: +form.has_tech_support.value,
    has_streaming_tv: +form.has_streaming_tv.value,
    has_streaming_movies: +form.has_streaming_movies.value,
    customer_satisfaction: +form.customer_satisfaction.value,
    num_complaints: +form.num_complaints.value,
    num_service_calls: +form.num_service_calls.value,
    late_payments: +form.late_payments.value,
    avg_monthly_gb: +form.avg_monthly_gb.value,
    days_since_last_interaction: +form.days_since_last_interaction.value,
    credit_score: form.credit_score.value ? +form.credit_score.value : null,
  };

  btn.disabled = true;
  btn.innerHTML = `<div class="spinner"></div> Predicting...`;
  resultEl.innerHTML = `<div class="loading"><div class="spinner"></div> Analyzing...</div>`;

  try {
    const result = await API.predict.single(data);
    const prob = result.churn_probability;
    const level = result.risk_level;
    const levelClass = level.toLowerCase();
    const pct = (prob * 100).toFixed(1);

    resultEl.innerHTML = `
      <div style="text-align:center">
        <div class="risk-label ${levelClass}" style="font-size:48px;margin-bottom:4px">${pct}%</div>
        <div class="risk-sublabel">Churn Probability</div>
        <div class="prob-bar" style="margin:20px 0 8px"><div class="prob-fill" style="width:${pct}%"></div></div>
        <div style="display:flex;justify-content:space-between;font-size:11px;color:var(--text-muted);margin-bottom:20px"><span>Low</span><span>Medium</span><span>High</span></div>
        <div style="margin-bottom:16px">${riskBadge(level)}</div>
        <div style="font-size:12px;color:var(--text-secondary);line-height:1.6;background:var(--bg-secondary);padding:12px;border-radius:8px">${result.interpretation}</div>
      </div>
    `;
  } catch (e) {
    resultEl.innerHTML = `<div style="color:var(--danger);font-size:13px;padding:16px">${e.message}</div>`;
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<span>⚡</span> Predict Churn`;
  }
}

// ── Page: Customer Deciles ───────────────────────────────────
let _decileGroups = null;

function downloadDecileCSV(decile) {
  const group = _decileGroups?.find(g => g.decile === decile);
  if (!group) return;
  const rows = [["customer_id", "churn_probability"]];
  group.customers.forEach(c => rows.push([c.customer_id, c.churn_probability.toFixed(4)]));
  const csv = rows.map(r => r.join(",")).join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `decil_${decile}_clientes.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

async function loadChurners() {
  const container = document.getElementById("churners-content");
  loading(container);
  try {
    const groups = await API.predict.decileGroups(20);
    _decileGroups = groups;

    const cards = groups.map(g => {
      const pct = (g.avg_probability * 100).toFixed(1);
      const rows = g.customers.map((c, i) => `
        <tr>
          <td>${i + 1}</td>
          <td><code style="font-size:11px;color:var(--text-secondary)">${c.customer_id}</code></td>
          <td>
            <div class="prob-mini-bar">
              <div class="prob-mini-track"><div class="prob-mini-fill" style="width:${(c.churn_probability * 100).toFixed(0)}%"></div></div>
              <span style="font-size:12px;font-weight:600">${(c.churn_probability * 100).toFixed(1)}%</span>
            </div>
          </td>
        </tr>`).join("");

      return `
        <div class="card" style="margin-bottom:16px">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
            <div style="display:flex;align-items:center;gap:12px">
              ${decileBadge(g.decile)}
              <div>
                <span style="font-weight:700;font-size:15px">Decil ${g.decile}</span>
                <span style="color:var(--text-muted);font-size:12px;margin-left:10px">
                  ${fmt(g.customer_count)} clientes · Prob. promedio: <strong>${pct}%</strong>
                  · Rango: ${(g.min_probability * 100).toFixed(1)}% – ${(g.max_probability * 100).toFixed(1)}%
                </span>
              </div>
            </div>
            <button onclick="downloadDecileCSV(${g.decile})" class="btn-download" title="Descargar CSV">
              &#8681; CSV
            </button>
          </div>
          <div class="table-container">
            <table class="churners-table">
              <thead><tr><th>#</th><th>Customer ID</th><th>Churn Probability</th></tr></thead>
              <tbody>${rows}</tbody>
            </table>
          </div>
        </div>`;
    }).join("");

    container.innerHTML = cards;
  } catch (e) {
    container.innerHTML = errorBanner(e.message);
  }
}

// ── Page loader map ──────────────────────────────────────────
const PAGE_LOADERS = {
  overview: loadOverview,
  eda: loadEDA,
  models: loadModels,
  churners: loadChurners,
};

function showError(id, msg) {
  const el = document.getElementById(id);
  if (el) el.innerHTML = errorBanner(msg);
}

// ── Init ─────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  navigate("overview");
});
