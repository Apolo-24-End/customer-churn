const API_BASE = window.location.origin;

async function fetchJSON(path) {
  const res = await fetch(API_BASE + path);
  if (!res.ok) throw new Error(`API error ${res.status}: ${await res.text()}`);
  return res.json();
}

async function postJSON(path, body) {
  const res = await fetch(API_BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`API error ${res.status}: ${await res.text()}`);
  return res.json();
}

const API = {
  eda: {
    all: () => fetchJSON("/eda/all"),
    overview: () => fetchJSON("/eda/overview"),
    correlation: () => fetchJSON("/eda/correlation"),
    churnByTenure: () => fetchJSON("/eda/churn-by-tenure"),
    churnBy: (cat) => fetchJSON(`/eda/churn-by/${cat}`),
    distribution: (col) => fetchJSON(`/eda/distributions/${col}`),
  },
  model: {
    results: () => fetchJSON("/model/results"),
    comparison: () => fetchJSON("/model/comparison"),
    rocCurves: () => fetchJSON("/model/roc-curves"),
    features: () => fetchJSON("/model/features"),
    confusionMatrix: (name) => fetchJSON(`/model/confusion-matrix/${name}`),
  },
  predict: {
    single: (data) => postJSON("/predict/single", data),
    decileGroups: (topN = 20) => fetchJSON(`/predict/decile-groups?top_n=${topN}`),
  },
};
