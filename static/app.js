const form = document.querySelector("#analyzeForm");
const fileInput = document.querySelector("#fileInput");
const fileName = document.querySelector("#fileName");
const textInput = document.querySelector("#srsText");
const toast = document.querySelector("#toast");
const filterSelect = document.querySelector("#filterSelect");
const exportButton = document.querySelector("#exportButton");
const resultsBody = document.querySelector("#resultsBody");

let latestRows = [];

const statusLabels = {
  trained: "Trained",
  "metadata-only": "Needs training",
  heuristic: "Heuristic",
  "bert-tiny": "BERT-tiny",
  "sklearn-fallback": "Fallback",
  unavailable: "Unavailable",
  transformer: "Transformer",
};

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("show");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => toast.classList.remove("show"), 3600);
}

function pct(value) {
  return `${Math.round((Number(value) || 0) * 100)}%`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function loadStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    document.querySelector("#model1Status").textContent = statusLabels[data.model1.status] || data.model1.status;
    document.querySelector("#model2Status").textContent = statusLabels[data.model2.status] || data.model2.status;
    document.querySelector("#model3Status").textContent = statusLabels[data.model3.status] || data.model3.status;
  } catch {
    showToast("Could not read model status. Start the backend server first.");
  }
}

function updateMetrics(summary) {
  document.querySelector("#metricCandidates").textContent = summary.candidates || 0;
  document.querySelector("#metricRequirements").textContent = summary.requirements || 0;
  document.querySelector("#metricFunctional").textContent = summary.functional || 0;
  document.querySelector("#metricNfr").textContent = summary.nonFunctional || 0;

  const stack = document.querySelector("#typeStack");
  const entries = Object.entries(summary.nfrTypes || {});
  if (!entries.length) {
    stack.innerHTML = `<div class="empty-state">No NFR categories detected yet.</div>`;
    return;
  }
  const max = Math.max(...entries.map(([, value]) => value), 1);
  stack.innerHTML = entries
    .sort((a, b) => b[1] - a[1])
    .map(([name, value]) => {
      const width = Math.max(8, Math.round((value / max) * 100));
      return `
        <div class="type-row">
          <header><span>${escapeHtml(name)}</span><strong>${value}</strong></header>
          <div class="bar"><span style="width:${width}%"></span></div>
        </div>`;
    })
    .join("");
}

function filteredRows() {
  const filter = filterSelect.value;
  if (filter === "all") return latestRows;
  return latestRows.filter((row) => {
    if (filter === "FR" || filter === "NFR") return row.class.code === filter;
    return row.nfrType && row.nfrType.name === filter;
  });
}

function renderRows() {
  const rows = filteredRows();
  if (!rows.length) {
    resultsBody.innerHTML = `<tr><td colspan="4" class="empty-table">No rows match this view.</td></tr>`;
    return;
  }
  resultsBody.innerHTML = rows
    .map((row) => {
      const isNfr = row.class.code === "NFR";
      const type = row.nfrType
        ? `<span class="badge type">${escapeHtml(row.nfrType.name)}</span>`
        : `<span class="empty-state">Not applicable</span>`;
      return `
        <tr>
          <td><div class="requirement-text">${escapeHtml(row.text)}</div></td>
          <td><span class="badge ${isNfr ? "nfr" : "fr"}">${escapeHtml(row.class.name)}</span></td>
          <td>${type}</td>
          <td>
            <div class="confidence">${pct(row.classConfidence)}</div>
            <small class="empty-state">Requirement ${pct(row.requirementConfidence)}</small>
          </td>
        </tr>`;
    })
    .join("");
}

function exportCsv() {
  if (!latestRows.length) {
    showToast("Run an analysis before exporting.");
    return;
  }
  const headers = ["Requirement", "FR/NFR", "NFR Type", "Requirement Confidence", "Class Confidence"];
  const lines = [headers.join(",")];
  for (const row of filteredRows()) {
    const values = [
      row.text,
      row.class.name,
      row.nfrType ? row.nfrType.name : "",
      row.requirementConfidence,
      row.classConfidence,
    ].map((value) => `"${String(value).replaceAll('"', '""')}"`);
    lines.push(values.join(","));
  }
  const blob = new Blob([lines.join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "srs_requirement_analysis.csv";
  link.click();
  URL.revokeObjectURL(url);
}

fileInput.addEventListener("change", () => {
  fileName.textContent = fileInput.files[0] ? fileInput.files[0].name : "No file selected";
});

filterSelect.addEventListener("change", renderRows);
exportButton.addEventListener("click", exportCsv);

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = form.querySelector("button[type='submit']");
  const data = new FormData();
  if (fileInput.files[0]) data.append("file", fileInput.files[0]);
  data.append("text", textInput.value);
  button.disabled = true;
  button.innerHTML = `<span class="button-icon">…</span>Analyzing`;
  try {
    const res = await fetch("/api/analyze", { method: "POST", body: data });
    const payload = await res.json();
    if (!res.ok) throw new Error(payload.detail || "Analysis failed.");
    latestRows = payload.requirements || [];
    updateMetrics(payload.summary || {});
    filterSelect.value = "all";
    renderRows();
    showToast(`Analysis complete: ${latestRows.length} requirements found.`);
    await loadStatus();
  } catch (error) {
    showToast(error.message || "Analysis failed.");
  } finally {
    button.disabled = false;
    button.innerHTML = `<span class="button-icon">↗</span>Analyze`;
  }
});

loadStatus();
