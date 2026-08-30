// Public evaluation view.
import { apiGet, apiPost } from "./api.js";
import { el, makeSection, showToast } from "./dom-utils.js";

const DEFAULT_EVAL_WORKERS = 4;

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, ch => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;",
    "'": "&#39;",
  }[ch]));
}

function renderStatCard(label, value) {
  return `
    <div class="stat-card">
      <div class="stat-label">${escapeHtml(label)}</div>
      <div class="stat-value">${escapeHtml(value)}</div>
    </div>
  `;
}

function reasoningOptions(selected = "medium") {
  const options = [
    ["none", "None"],
    ["low", "Low"],
    ["medium", "Medium"],
    ["high", "High"],
  ];
  return options.map(([value, label]) =>
    `<option value="${value}"${value === selected ? " selected" : ""}>${label}</option>`
  ).join("");
}

function formatRate(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function formatValue(value, maxLen = 90) {
  if (value === undefined) return "—";
  if (value === null) return "null";
  const text = typeof value === "object" ? JSON.stringify(value) : String(value);
  return text.length > maxLen ? `${text.slice(0, maxLen)}...` : text;
}

function inlineMarkdown(text) {
  let html = escapeHtml(text);
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  return html;
}

function splitTableRow(line) {
  return line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map(cell => cell.trim());
}

function isTableSeparator(line) {
  return /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line);
}

function renderMarkdown(markdown) {
  const lines = String(markdown || "").replace(/\r\n/g, "\n").split("\n");
  const html = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();
    if (!trimmed) {
      i += 1;
      continue;
    }

    if (trimmed.startsWith("```")) {
      const code = [];
      i += 1;
      while (i < lines.length && !lines[i].trim().startsWith("```")) {
        code.push(lines[i]);
        i += 1;
      }
      i += 1;
      html.push(`<pre class="markdown-code"><code>${escapeHtml(code.join("\n"))}</code></pre>`);
      continue;
    }

    if (trimmed.startsWith("|") && i + 1 < lines.length && isTableSeparator(lines[i + 1])) {
      const header = splitTableRow(lines[i]);
      i += 2;
      const rows = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) {
        rows.push(splitTableRow(lines[i]));
        i += 1;
      }
      html.push(`
        <table class="field-results-table markdown-table">
          <thead><tr>${header.map(cell => `<th>${inlineMarkdown(cell)}</th>`).join("")}</tr></thead>
          <tbody>${rows.map(row => `<tr>${row.map(cell => `<td>${inlineMarkdown(cell)}</td>`).join("")}</tr>`).join("")}</tbody>
        </table>
      `);
      continue;
    }

    const heading = trimmed.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      const level = Math.min(4, heading[1].length);
      html.push(`<h${level}>${inlineMarkdown(heading[2])}</h${level}>`);
      i += 1;
      continue;
    }

    if (trimmed.startsWith("- ")) {
      const items = [];
      while (i < lines.length && lines[i].trim().startsWith("- ")) {
        items.push(lines[i].trim().slice(2));
        i += 1;
      }
      html.push(`<ul>${items.map(item => `<li>${inlineMarkdown(item)}</li>`).join("")}</ul>`);
      continue;
    }

    html.push(`<p>${inlineMarkdown(trimmed)}</p>`);
    i += 1;
  }

  return `<div class="markdown-report">${html.join("")}</div>`;
}

function overviewMarkdown(markdown) {
  const text = String(markdown || "");
  const marker = "\n## Errors";
  const index = text.indexOf(marker);
  if (index === -1) return text;
  return `${text.slice(0, index)}\n\n## Errors\n\nPer-item errors are available by clicking rows in the item table below.`;
}

function stateValue(state, path) {
  if (!state || typeof state !== "object") return undefined;
  if (Object.prototype.hasOwnProperty.call(state, path)) return state[path];
  const match = String(path).match(/^(\w+)\[(\d+|\?gt\d+)]\.(.+)$/);
  if (!match) return undefined;
  const [, group, idxToken, field] = match;
  const idx = idxToken.startsWith("?gt") ? Number(idxToken.slice(3)) : Number(idxToken);
  const rows = state[group];
  if (!Array.isArray(rows) || !rows[idx] || typeof rows[idx] !== "object") return undefined;
  return rows[idx][field];
}

function goldValue(groundTruth, qid) {
  const fromState = stateValue(groundTruth?.gold_resulting_state, qid);
  if (fromState !== undefined) return fromState;
  const entry = groundTruth?.fields?.[qid];
  if (entry && typeof entry === "object") return entry.expected_summary ?? entry.expected;
  const match = String(qid).match(/^(\w+)\[(\d+|\?gt\d+)]\.(.+)$/);
  if (!match) return undefined;
  const [, group, idxToken, field] = match;
  const gtIndex = idxToken.startsWith("?gt") ? Number(idxToken.slice(3)) : Number(idxToken);
  const instances = groundTruth?.repeat_groups?.[group]?.instances || [];
  const instance = instances.find(row => row.ground_truth_index === gtIndex);
  const fieldEntry = instance?.fields?.[field];
  return fieldEntry?.expected_summary ?? fieldEntry?.expected;
}

function summarizeUtteranceRow(utterance, runItemsById) {
  const itemId = utterance?.scenario_id || "";
  const views = utterance?.metric_views || {};
  const allFields = views.all_fields || {};
  const changed = views.changed_fields || {};
  const exact = views.whole_record_exact_match || {};
  const dv = utterance?.derived_variables || {};
  const runItem = runItemsById.get(itemId) || {};
  return {
    item_id: itemId,
    questionnaire_id: utterance.questionnaire || runItem.questionnaire_id || "",
    operation_count: runItem.operation_count ?? "",
    exact_match: exact.exact_match,
    state_condition: dv.prior_state_condition || utterance.state || "",
    all_accuracy: allFields.accuracy,
    changed_f1: changed.strict?.f1,
    errors: (allFields.partial || 0) + (allFields.incorrect || 0),
    utterance,
  };
}

function fallbackItemRow(row) {
  return {
    item_id: row.item_id || "",
    questionnaire_id: row.questionnaire_id || "",
    operation_count: row.operation_count ?? "",
    exact_match: row.exact_match,
    state_condition: "",
    all_accuracy: null,
    changed_f1: null,
    errors: row.evaluation_summary
      ? (row.evaluation_summary.partially_correct || 0) + (row.evaluation_summary.incorrect || 0)
      : "",
    utterance: null,
  };
}

function buildRunItemRows(run, summary) {
  const runItemsById = new Map((run.items || []).map(row => [row.item_id, row]));
  if (summary?.utterances?.length) {
    return summary.utterances.map(utterance => summarizeUtteranceRow(utterance, runItemsById));
  }
  return (run.items || []).map(fallbackItemRow);
}

function renderOperationList(operations) {
  if (!Array.isArray(operations) || !operations.length) {
    return `<p class="muted">No operations recorded.</p>`;
  }
  return `
    <table class="field-results-table">
      <thead><tr><th>Op</th><th>Path</th><th>Value</th></tr></thead>
      <tbody>
        ${operations.map(op => `
          <tr>
            <td>${escapeHtml(op.op || "")}</td>
            <td class="field-qid">${escapeHtml(op.path || "")}</td>
            <td class="value-cell" title="${escapeHtml(formatValue(op.value, 1000))}">${escapeHtml(formatValue(op.value))}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function renderFieldVerdicts(fieldResults, turnResult, groundTruth, mismatches) {
  const mismatchMap = new Map((mismatches || []).map(row => [row.qid, row]));
  const entries = Object.entries(fieldResults || {});
  if (!entries.length) {
    return `<p class="muted">No field verdicts found.</p>`;
  }
  const order = { incorrect: 0, partially_correct: 1, needs_semantic: 2, correct: 3 };
  entries.sort(([, a], [, b]) => (order[a.correctness] ?? 9) - (order[b.correctness] ?? 9));
  return `
    <table class="field-results-table">
      <thead>
        <tr><th>Field</th><th>Verdict</th><th>Source</th><th>Decision</th><th>Candidate</th><th>Gold</th><th>Reasoning</th></tr>
      </thead>
      <tbody>
        ${entries.map(([qid, verdict]) => {
          const correctness = verdict.correctness || verdict.final_correctness || "unknown";
          const mismatch = mismatchMap.get(qid) || {};
          const candidate = mismatch.actual ?? stateValue(turnResult.answers_after, qid);
          const expected = mismatch.expected ?? goldValue(groundTruth, qid);
          return `
            <tr>
              <td class="field-qid" title="${escapeHtml(qid)}">${escapeHtml(qid)}</td>
              <td>
                <span class="verdict-badge verdict-${escapeHtml(correctness)}">${escapeHtml(correctness.replace(/_/g, " "))}</span>
                ${verdict.partial_reason ? `<span class="partial-reason-badge">${escapeHtml(verdict.partial_reason)}</span>` : ""}
              </td>
              <td>${escapeHtml(verdict.support_source ?? verdict.source ?? "—")}</td>
              <td>${escapeHtml(verdict.decision_source ?? "—")}</td>
              <td class="value-cell" title="${escapeHtml(formatValue(candidate, 1000))}">${escapeHtml(formatValue(candidate))}</td>
              <td class="value-cell" title="${escapeHtml(formatValue(expected, 1000))}">${escapeHtml(formatValue(expected))}</td>
              <td class="reasoning-cell">${escapeHtml(verdict.reasoning || "")}</td>
            </tr>
          `;
        }).join("")}
      </tbody>
    </table>
  `;
}

async function openItemReview(runId, row) {
  const encodedRunId = encodeURIComponent(runId);
  const encodedItemId = encodeURIComponent(row.item_id);
  let detail;
  try {
    detail = await apiGet(`/api/evals/${encodedRunId}/items/${encodedItemId}`);
  } catch (err) {
    showToast(`Failed to load item: ${err.message}`, "err");
    return;
  }
  const turn = detail.turn_result || {};
  const evaluation = detail.evaluation || {};
  const groundTruth = detail.ground_truth || {};
  const utterance = row.utterance || {};
  const scores = utterance.scores || evaluation.summary || {};
  const operations = (turn.agent_response && typeof turn.agent_response === "object")
    ? turn.agent_response.operations
    : [];

  const overlay = document.createElement("div");
  overlay.className = "ev-overlay";
  overlay.innerHTML = `
    <div class="ev-popup drill-down-popup">
      <div class="ev-header">
        <div>
          <h3 class="ev-title">${escapeHtml(row.item_id)}</h3>
          <div class="ev-subtitle">${escapeHtml(row.questionnaire_id)} · ${escapeHtml(row.state_condition || turn.state || "")}</div>
        </div>
        <button class="ev-close-btn" type="button" aria-label="Close">×</button>
      </div>
      <div class="drill-down-body">
        <div class="drill-stats-row">
          <div class="drill-stat-chip" style="border-left:3px solid #10b981;"><strong>${escapeHtml(scores.correct ?? 0)}</strong> correct</div>
          <div class="drill-stat-chip" style="border-left:3px solid #f59e0b;"><strong>${escapeHtml(scores.partially_correct ?? 0)}</strong> partial</div>
          <div class="drill-stat-chip" style="border-left:3px solid #ef4444;"><strong>${escapeHtml(scores.incorrect ?? 0)}</strong> incorrect</div>
          <div class="drill-stat-chip" style="border-left:3px solid #0072B2;"><strong>${formatRate(row.all_accuracy ?? scores.accuracy)}</strong> all-field acc</div>
          <div class="drill-stat-chip" style="border-left:3px solid #6366f1;"><strong>${formatRate(row.changed_f1)}</strong> changed F1</div>
        </div>

        <div class="drill-section-title">Current Utterance</div>
        <blockquote class="drill-utterance">${escapeHtml(turn.current_utterance || "(none)")}</blockquote>

        <div class="drill-section-title">Model Answer</div>
        ${renderOperationList(operations)}

        <div class="drill-section-title">Evaluator Verdicts</div>
        ${renderFieldVerdicts(evaluation.field_results || {}, turn, groundTruth, utterance.mismatches || [])}
      </div>
    </div>
  `;
  overlay.querySelector(".ev-close-btn")?.addEventListener("click", () => overlay.remove());
  overlay.addEventListener("click", event => {
    if (event.target === overlay) overlay.remove();
  });
  document.body.appendChild(overlay);
}

function openFigureViewer(figures, startIndex) {
  const fig = figures[startIndex];
  if (!fig) return;

  const overlay = document.createElement("div");
  overlay.className = "ev-overlay";
  overlay.innerHTML = `
    <div class="ev-popup figure-popup">
      <div class="ev-header">
        <div>
          <div class="figure-viewer-kicker">Figure</div>
          <h3 class="ev-title">${escapeHtml(fig.title)}</h3>
          <div class="ev-subtitle">${escapeHtml(fig.filename)}</div>
        </div>
        <button class="ev-close-btn" type="button" aria-label="Close">×</button>
      </div>
      <div class="drill-down-body">
        <div class="figure-viewer">
          <div class="figure-stage-shell">
            <div class="figure-stage">
              <img class="figure-stage-img" src="${escapeHtml(fig.url)}" alt="${escapeHtml(fig.title)}" />
              <div class="figure-stage-badge">Expanded</div>
            </div>
            <aside class="figure-sidecar">
              <div class="figure-viewer-title">${escapeHtml(fig.title)}</div>
              <div class="figure-viewer-file">${escapeHtml(fig.filename)}</div>
              <div class="figure-viewer-tip">
                <strong>Formula</strong><br>
                ${escapeHtml(fig.note || "No formula note is available for this figure.")}
              </div>
            </aside>
          </div>
        </div>
      </div>
    </div>
  `;
  overlay.querySelector(".ev-close-btn")?.addEventListener("click", () => overlay.remove());
  overlay.addEventListener("click", event => {
    if (event.target === overlay) overlay.remove();
  });
  document.body.appendChild(overlay);
}

// ---------------------------------------------------------------------------
// Hierarchical item tree selector
// ---------------------------------------------------------------------------

function buildItemTree(treeData, container) {
  const allIds = [];
  for (const source of Object.values(treeData)) {
    for (const ids of Object.values(source)) allIds.push(...ids);
  }

  const state = { selected: new Set(allIds) };

  function syncParents(node) {
    if (!node) return;
    const parent = node.parentElement?.closest?.(".tree-group");
    if (!parent) return;
    const cb = parent.querySelector(":scope > label > input[type=checkbox]");
    if (!cb) return;
    const childCbs = parent.querySelectorAll(":scope > .tree-children input[type=checkbox]");
    const checked = [...childCbs].filter(c => c.checked).length;
    cb.checked = checked === childCbs.length;
    cb.indeterminate = checked > 0 && checked < childCbs.length;
    syncParents(parent.parentElement?.closest?.(".tree-group"));
  }

  function setChildren(group, checked) {
    group.querySelectorAll("input[type=checkbox]").forEach(cb => {
      cb.checked = checked;
      cb.indeterminate = false;
      if (cb.dataset.itemId) {
        if (checked) state.selected.add(cb.dataset.itemId);
        else state.selected.delete(cb.dataset.itemId);
      }
    });
  }

  function makeLeaf(itemId) {
    const li = document.createElement("div");
    li.className = "tree-leaf";
    const lbl = document.createElement("label");
    lbl.className = "tree-label";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = true;
    cb.dataset.itemId = itemId;
    cb.addEventListener("change", () => {
      if (cb.checked) state.selected.add(itemId);
      else state.selected.delete(itemId);
      syncParents(li);
      updateSummary();
    });
    lbl.appendChild(cb);
    lbl.appendChild(document.createTextNode(" " + itemId));
    li.appendChild(lbl);
    return li;
  }

  function makeGroup(label, children, startExpanded) {
    const group = document.createElement("div");
    group.className = "tree-group";

    const header = document.createElement("label");
    header.className = "tree-label tree-branch";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = true;
    cb.addEventListener("change", () => {
      setChildren(childrenDiv, cb.checked);
      cb.indeterminate = false;
      updateSummary();
    });
    const toggle = document.createElement("span");
    toggle.className = "tree-toggle";
    toggle.textContent = startExpanded ? "▾" : "▸";
    toggle.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      const open = childrenDiv.style.display !== "none";
      childrenDiv.style.display = open ? "none" : "block";
      toggle.textContent = open ? "▸" : "▾";
    });

    const count = document.createElement("span");
    count.className = "tree-count";
    let total = 0;
    for (const ch of children) {
      total += ch.querySelectorAll("input[data-item-id]").length;
    }
    count.textContent = `(${total})`;

    header.appendChild(cb);
    header.appendChild(toggle);
    header.appendChild(document.createTextNode(" " + label + " "));
    header.appendChild(count);
    group.appendChild(header);

    const childrenDiv = document.createElement("div");
    childrenDiv.className = "tree-children";
    childrenDiv.style.display = startExpanded ? "block" : "none";
    for (const ch of children) childrenDiv.appendChild(ch);
    group.appendChild(childrenDiv);

    return group;
  }

  // Build leaf → questionnaire → source → root
  const sourceGroups = [];
  for (const [sourceName, questionnaires] of Object.entries(treeData)) {
    const qGroups = [];
    for (const [qName, itemIds] of Object.entries(questionnaires)) {
      const leaves = itemIds.map(makeLeaf);
      qGroups.push(makeGroup(qName, leaves, false));
    }
    sourceGroups.push(makeGroup(sourceName, qGroups, false));
  }
  const root = makeGroup("All items", sourceGroups, true);

  container.innerHTML = "";
  container.appendChild(root);

  const summary = document.createElement("div");
  summary.className = "tree-summary";
  container.appendChild(summary);

  function updateSummary() {
    summary.textContent = `${state.selected.size} / ${allIds.length} items selected`;
  }
  updateSummary();

  return state;
}


// ---------------------------------------------------------------------------
// Run detail view
// ---------------------------------------------------------------------------

async function loadRunDetail(runId, detailContainer) {
  const encodedRunId = encodeURIComponent(runId);
  const run = await apiGet(`/api/evals/${encodedRunId}`);
  const summary = await apiGet(`/api/evals/${encodedRunId}/summary`).catch(() => null);
  const report = await apiGet(`/api/evals/${encodedRunId}/report`).catch(() => null);
  const figures = await apiGet(`/api/evals/${encodedRunId}/figures`).catch(() => []);
  const metricViews = summary?.aggregate?.metric_views || {};
  const allFields = metricViews.all_fields || {};
  const changedFields = metricViews.changed_fields || {};
  const exact = metricViews.whole_record_exact_match || {};
  const preservation = metricViews.preservation || {};
  const itemRows = buildRunItemRows(run, summary);
  const metadata = summary?.metadata?.run || summary?.provenance?.run_metadata || {};
  const runTitle = run.display_name || run.run_purpose || run.run_id;
  const runReasoning = run.model_reasoning_effort || metadata.model_reasoning_effort || "n/a";
  const judgeModel = run.evaluator_model_id || metadata.evaluator_model_id || "default judge";
  const judgeReasoning = run.evaluator_reasoning_effort || metadata.evaluator_reasoning_effort || "n/a";
  const figureHtml = figures.length ? `
    <h3 class="drill-section-title" style="margin-top:16px;">Figures</h3>
    <div class="figure-grid" style="margin-bottom:16px;">
      ${figures.map((fig, index) => `
        <button type="button" class="figure-card figure-card-button" data-figure-index="${index}">
          <div class="figure-card-preview">
            <img class="figure-card-img" src="${escapeHtml(fig.url)}" alt="${escapeHtml(fig.title)}" loading="lazy" />
          </div>
          <div class="figure-card-body">
            <div class="figure-card-title">${escapeHtml(fig.title)}</div>
            <div class="figure-card-file">${escapeHtml(fig.filename)}</div>
          </div>
        </button>
      `).join("")}
    </div>
  ` : `<p style="font-size:12px;color:var(--help);margin-top:12px;">No figures generated for this run.</p>`;

  detailContainer.innerHTML = `
    <div style="margin-bottom:12px;">
      <div style="font-size:16px;font-weight:800;color:var(--text);">${escapeHtml(runTitle)}</div>
      <div style="font-size:12px;color:var(--text-3);margin-top:2px;">Artifact: ${escapeHtml(run.run_id)}</div>
    </div>
    <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px;">
      ${renderStatCard("Items", run.item_count)}
      ${renderStatCard("Exact", `${run.exact_matches}/${run.scored_items}`)}
      ${renderStatCard("All-field acc", formatRate(allFields.accuracy ?? run.metrics?.all_field_accuracy))}
      ${renderStatCard("Changed F1", formatRate(changedFields.strict?.f1 ?? run.metrics?.changed_field_f1))}
      ${renderStatCard("Exact rate", formatRate(exact.exact_match_rate ?? run.metrics?.exact_match_rate))}
      ${renderStatCard("Preserve err", formatRate(preservation.preservation_error_rate ?? run.metrics?.preservation_error_rate))}
      ${renderStatCard("Run reasoning", runReasoning)}
      ${renderStatCard("Judge reasoning", `${judgeReasoning}${judgeModel ? ` · ${judgeModel}` : ""}`)}
    </div>
    ${figureHtml}
    ${report?.markdown ? renderMarkdown(overviewMarkdown(report.markdown)) : `<p style="font-size:12px;color:var(--help);">No report.md found for this run.</p>`}
    <h3 class="drill-section-title" style="margin-top:16px;">Items</h3>
    <table class="field-results-table">
      <thead>
        <tr>
          <th>Item</th>
          <th>Questionnaire</th>
          <th>State</th>
          <th>Operations</th>
          <th>All-field Acc</th>
          <th>Changed F1</th>
          <th>Errors</th>
          <th>Exact</th>
        </tr>
      </thead>
      <tbody>
        ${itemRows.map((row, index) => `
          <tr class="utt-row run-item-row" data-row-index="${index}">
            <td>${escapeHtml(row.item_id)}</td>
            <td>${escapeHtml(row.questionnaire_id)}</td>
            <td>${escapeHtml(row.state_condition || "n/a")}</td>
            <td>${escapeHtml(row.operation_count)}</td>
            <td>${formatRate(row.all_accuracy)}</td>
            <td>${formatRate(row.changed_f1)}</td>
            <td>${escapeHtml(row.errors)}</td>
            <td>${row.exact_match === true ? "match" : row.exact_match === false ? "diff" : "unscored"}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;

  detailContainer.querySelectorAll(".figure-card-button").forEach(button => {
    button.addEventListener("click", () => {
      openFigureViewer(figures, Number(button.dataset.figureIndex));
    });
  });

  detailContainer.querySelectorAll(".run-item-row").forEach(rowEl => {
    rowEl.addEventListener("click", () => {
      const index = Number(rowEl.dataset.rowIndex);
      const row = itemRows[index];
      if (row?.item_id) openItemReview(runId, row);
    });
  });
}

async function renderRuns(resultsList, detailContainer) {
  try {
    const runs = await apiGet("/api/evals");
    if (!runs.length) {
      resultsList.innerHTML = `<div style="padding:20px;text-align:center;color:var(--text-3);">No evaluations found yet.</div>`;
      detailContainer.innerHTML = `<p style="font-size:12px;color:var(--help);">Run a solver to see results.</p>`;
      return;
    }

    const splitForRun = run => {
      if (run.split === "test" || run.run_id?.startsWith("TEST144.")) return "test";
      if (run.split === "dev" || run.split === "train" || run.run_id?.startsWith("DEV36.") || run.run_id?.startsWith("TRAIN36.")) return "dev";
      return run.item_count > 36 ? "test" : "dev";
    };
    const grouped = {
      test: runs.filter(run => splitForRun(run) === "test"),
      dev: runs.filter(run => splitForRun(run) === "dev"),
    };
    let activeSplit = grouped.test.length ? "test" : "dev";

    const renderList = (loadFirst = false) => {
      const activeRuns = grouped[activeSplit] || [];
      resultsList.innerHTML = `
        <div class="eval-split-tabs" style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:10px;">
          <button type="button" data-eval-split="test" class="btn ${activeSplit === "test" ? "primary" : ""}" style="justify-content:center;">Test (${grouped.test.length})</button>
          <button type="button" data-eval-split="dev" class="btn ${activeSplit === "dev" ? "primary" : ""}" style="justify-content:center;">Dev (${grouped.dev.length})</button>
        </div>
        <div class="eval-split-list">
          ${activeRuns.length ? activeRuns.map(run => `
            <button type="button" class="eval-card" data-run-id="${escapeHtml(run.run_id)}" style="width:100%;text-align:left;margin-bottom:8px;padding:10px;border:1px solid var(--border);border-radius:8px;background:#fff;">
              <div style="font-weight:700;color:var(--text);">${escapeHtml(run.display_name || run.run_purpose || run.run_id)}</div>
              <div style="font-size:11px;color:var(--text-3);margin-top:2px;overflow-wrap:anywhere;">${escapeHtml(run.run_id)}</div>
              <div style="font-size:12px;color:var(--text-3);margin-top:2px;">${escapeHtml(run.solver || "Run")}${run.model_id ? " · " + escapeHtml(run.model_id) : ""}${run.model_reasoning_effort ? " · " + escapeHtml(run.model_reasoning_effort) + " reasoning" : ""} · ${run.item_count} item(s)</div>
              <div style="font-size:12px;color:#059669;margin-top:2px;">${run.exact_matches}/${run.scored_items} exact matches${run.metrics?.all_field_accuracy !== undefined ? " · " + formatRate(run.metrics.all_field_accuracy) + " all-field acc" : ""}</div>
            </button>
          `).join("") : `<div style="padding:20px;text-align:center;color:var(--text-3);">No ${activeSplit} evaluations found.</div>`}
        </div>
      `;

      resultsList.querySelectorAll("[data-eval-split]").forEach(button => {
        button.addEventListener("click", () => {
          activeSplit = button.dataset.evalSplit;
          detailContainer.innerHTML = `<p style="font-size:12px;color:var(--help);">Select a ${activeSplit} evaluation to inspect item-level results.</p>`;
          renderList(false);
        });
      });

      resultsList.querySelectorAll("[data-run-id]").forEach(button => {
        button.addEventListener("click", async () => {
          await loadRunDetail(button.dataset.runId, detailContainer);
        });
      });
      if (loadFirst) resultsList.querySelector("[data-run-id]")?.click();
    };

    renderList(true);
  } catch (err) {
    resultsList.innerHTML = `<div style="padding:20px;color:#d55e00;">Failed to load evaluations: ${escapeHtml(err.message)}</div>`;
  }
}


// ---------------------------------------------------------------------------
// Main eval view
// ---------------------------------------------------------------------------

export async function renderEvalView() {
  document.getElementById("no-selection").style.display = "none";
  const editor = document.getElementById("editor");
  editor.style.display = "flex";
  document.getElementById("editor-item-id").textContent = "EVAL";
  document.getElementById("editor-item-title").textContent = "Public runner evaluation";
  document.getElementById("btn-save").style.display = "none";
  document.getElementById("btn-prev").style.display = "none";
  document.getElementById("btn-next").style.display = "none";

  const form = document.getElementById("editor-form");
  form.innerHTML = "";
  document.getElementById("checklist-panel").innerHTML = "";

  // Section 1: Run config
  const runSection = makeSection("eval-run", 1, "Run Configuration", "Select system, model, and items.", () => {}, false);
  form.appendChild(runSection);
  const runBody = runSection.querySelector(".section-body");
  runBody.innerHTML = `
    <div style="padding-top:14px;display:flex;gap:10px;align-items:end;flex-wrap:wrap;">
      <label class="field-group" style="margin:0;flex:1;min-width:180px;">
        <span>System</span>
        <select id="eval-system"><option value="">Loading...</option></select>
      </label>
      <label class="field-group" style="margin:0;width:180px;">
        <span>Item set</span>
        <select id="eval-item-source">
          <option value="custom" selected>Custom selection</option>
          <option value="dev">Dev split (36)</option>
          <option value="public">Public demo</option>
          <option value="studio">Benchmark</option>
          <option value="all">Public + benchmark</option>
        </select>
      </label>
      <label class="field-group" style="margin:0;flex:1;min-width:220px;">
        <span>Model</span>
        <select id="eval-model"><option value="">Loading...</option></select>
      </label>
      <label class="field-group" style="margin:0;width:150px;">
        <span>Model Reasoning</span>
        <select id="eval-model-reasoning">${reasoningOptions()}</select>
      </label>
      <label class="field-group" style="margin:0;flex:1;min-width:220px;">
        <span>Evaluator Model</span>
        <select id="eval-evaluator-model"><option value="">Loading...</option></select>
      </label>
      <label class="field-group" style="margin:0;width:170px;">
        <span>Evaluator Reasoning</span>
        <select id="eval-evaluator-reasoning">${reasoningOptions()}</select>
      </label>
      <label class="field-group" style="margin:0;width:180px;">
        <span>Run ID</span>
        <input id="eval-run-id" type="text" placeholder="optional" />
      </label>
      <button id="eval-run-btn" class="btn primary" type="button">Run</button>
    </div>
    <div id="eval-item-tree" style="margin-top:14px;"></div>
    <div id="eval-status" style="font-size:12px;color:var(--help);padding-top:10px;">Select items and click Run.</div>
  `;

  // Populate system dropdown
  const systemSelect = runBody.querySelector("#eval-system");
  try {
    const systems = await apiGet("/api/systems");
    systemSelect.innerHTML = systems.map(s =>
      `<option value="${escapeHtml(s.id)}">${escapeHtml(s.name)}</option>`
    ).join("");
  } catch {
    systemSelect.innerHTML = `<option value="conformbench.systems.YOUR_SYSTEM:solve">YOUR_SYSTEM skeleton</option>`;
  }

  // Populate model dropdown (grouped by provider)
  const modelSelect = runBody.querySelector("#eval-model");
  const evalModelSelect = runBody.querySelector("#eval-evaluator-model");
  try {
    const models = await apiGet("/api/models");
    const byProvider = {};
    for (const m of models) {
      (byProvider[m.provider] ??= []).push(m);
    }
    let html = "";
    for (const [provider, group] of Object.entries(byProvider)) {
      html += `<optgroup label="${escapeHtml(provider)}">`;
      for (const m of group) {
        html += `<option value="${escapeHtml(m.id)}">${escapeHtml(m.name)}</option>`;
      }
      html += `</optgroup>`;
    }
    modelSelect.innerHTML = html;
    evalModelSelect.innerHTML = `<option value="">(default judge)</option>` + html;
  } catch {
    modelSelect.innerHTML = `<option value="openai:gpt-5.4">GPT-5.4</option>`;
    evalModelSelect.innerHTML = `<option value="">(default judge)</option><option value="openai:gpt-5.4">GPT-5.4</option>`;
  }

  // Load tree data
  const itemSourceSelect = runBody.querySelector("#eval-item-source");
  const treeContainer = runBody.querySelector("#eval-item-tree");
  let treeState = { selected: new Set() };
  try {
    const treeData = await apiGet("/api/eval/item-tree");
    treeState = buildItemTree(treeData, treeContainer);
  } catch (err) {
    treeContainer.innerHTML = `<div style="color:#d55e00;">Failed to load items: ${escapeHtml(err.message)}</div>`;
  }

  const syncItemSourceMode = () => {
    const isCustom = itemSourceSelect.value === "custom";
    treeContainer.style.display = isCustom ? "" : "none";
    runBody.querySelector("#eval-status").textContent = isCustom
      ? "Select items and click Run."
      : `Click Run to evaluate the ${itemSourceSelect.options[itemSourceSelect.selectedIndex].text}.`;
  };
  itemSourceSelect.addEventListener("change", syncItemSourceMode);
  syncItemSourceMode();

  // Section 2: Past runs
  const resultsSection = makeSection("eval-results", 2, "Evaluations", "Select an evaluation to inspect item-level results.", () => {}, false);
  form.appendChild(resultsSection);
  const resultsList = el("div", { id: "eval-list", style: "padding:12px;max-height:300px;overflow-y:auto;" });
  resultsSection.querySelector(".section-body").appendChild(resultsList);

  // Section 3: Details
  const detailSection = makeSection("eval-detail", 3, "Details", "Per-item exact match and operation counts.", () => {}, false);
  form.appendChild(detailSection);
  const detailContainer = el("div", { id: "eval-detail-section", style: "padding:12px;" });
  detailSection.querySelector(".section-body").appendChild(detailContainer);

  // Run button handler
  runBody.querySelector("#eval-run-btn").onclick = async () => {
    const status = runBody.querySelector("#eval-status");
    const itemSource = itemSourceSelect.value.trim();
    const useCustomSelection = itemSource === "custom";
    const selectedIds = useCustomSelection ? [...treeState.selected] : [];
    if (useCustomSelection && !selectedIds.length) {
      status.textContent = "No items selected.";
      showToast("No items selected", "err");
      return;
    }
    const runTarget = useCustomSelection ? `${selectedIds.length} item(s)` : `${itemSource} set`;
    status.textContent = `Running evaluation on ${runTarget}...`;
    try {
      const result = await apiPost("/api/evals", {
        solver: runBody.querySelector("#eval-system").value.trim(),
        item_source: useCustomSelection ? "all" : itemSource,
        model_id: runBody.querySelector("#eval-model").value.trim() || null,
        model_reasoning_effort: runBody.querySelector("#eval-model-reasoning").value.trim() || null,
        evaluator_model_id: runBody.querySelector("#eval-evaluator-model").value.trim() || null,
        evaluator_reasoning_effort: runBody.querySelector("#eval-evaluator-reasoning").value.trim() || null,
        item_ids: selectedIds,
        run_id: runBody.querySelector("#eval-run-id").value.trim() || null,
        workers: DEFAULT_EVAL_WORKERS,
      });
      status.textContent = `Run ${result.run_id}: ${result.exact_matches}/${result.scored_items} exact matches (${result.item_count} items).`;
      showToast("Evaluation complete", "ok");
      await renderRuns(resultsList, detailContainer);
    } catch (err) {
      status.textContent = err.message;
      showToast(`Evaluation failed: ${err.message}`, "err");
    }
  };

  await renderRuns(resultsList, detailContainer);
}
