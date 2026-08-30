// -- Item section renderers (4-section redesign) -----------------------------
import { S } from "./state.js";
import { apiGet, apiPost, apiPut, loadContexts, getQFields, getQFieldMeta, getQTree } from "./api.js";
import { el, makeDateTextInput, markDirty, showToast, ensureExpectedOutcome, ensureDifficultyProfile, legacyHistoryConditionFromItem, readyItemRequiresFailureContract, cleanTargetedFailureModes } from "./dom-utils.js";
import {
  FAMILIES, CONTRAST_ROLES, STATE_CONDITIONS, HISTORY_CONDITIONS,
  EVIDENCE_SOURCES, EVAL_STRATEGIES, EXTRACTION_DIFFICULTIES, DIMENSIONS,
  DIMENSION_DEFAULTS, DELTA_DEFAULTS, FAILURE_MODE_SUGGESTIONS, VALUE_SOURCES,
  PRIMARY_DELTA_TYPES, EVIDENCE_DEFAULTS, FIELD_DELTA_TYPES, suggestDeltaType,
  DIFFICULTY_TIERS, formatClaimAnchorLabel, getClaimAnchorMeta,
} from "./constants.js";
import { updateChecklist } from "./checklist.js";
import { openEvidencePopup } from "./evidence-popup.js";
// Lazy import context-sections to avoid circular dependency (context-sections imports from this file)
// Used only in openInlineContextModal, called on click

/**
 * Collect the set of field IDs that live inside repeat_group containers
 * in a questionnaire tree. These should never appear as flat top-level
 * keys in a state/gold object.
 */
function collectRepeatGroupChildIds(qTree) {
  const childIds = new Set();
  function walk(questions, insideRg) {
    for (const q of questions) {
      const st = q.structure_type || "regular";
      if (st === "regular") {
        if (insideRg) childIds.add(q.id);
      } else if (st === "repeat_group") {
        walk(q.fields || q.questions || [], true);
      } else if (st === "group" || st === "gate") {
        walk(q.fields || q.questions || [], insideRg);
      } else if (st === "branch") {
        const branch = q.branch || {};
        for (const route of branch.routes || []) {
          walk(route.children || [], insideRg);
        }
        if (branch.default_children?.length) walk(branch.default_children, insideRg);
      }
    }
  }
  walk(qTree, false);
  return childIds;
}

/**
 * Collect the set of field IDs that are repeat_group containers (e.g. "vehicles").
 * These hold arrays of instance objects in the gold state.
 */
function collectRepeatGroupIds(qTree) {
  const rgIds = new Set();
  function walk(questions) {
    for (const q of questions) {
      const st = q.structure_type || "regular";
      if (st === "repeat_group") {
        rgIds.add(q.id);
        walk(q.fields || q.questions || []);
      } else if (st === "group" || st === "gate") {
        walk(q.fields || q.questions || []);
      } else if (st === "branch") {
        const branch = q.branch || {};
        for (const route of branch.routes || []) {
          walk(route.children || []);
        }
        if (branch.default_children?.length) walk(branch.default_children);
      }
    }
  }
  walk(qTree);
  return rgIds;
}

/** Remove flat repeat-group child keys from a state object. */
function stripFlatRepeatGroupFields(state, rgChildIds) {
  if (!rgChildIds || rgChildIds.size === 0) return state;
  const cleaned = {};
  for (const [k, v] of Object.entries(state)) {
    if (!rgChildIds.has(k)) cleaned[k] = v;
  }
  return cleaned;
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 1: Scenario Setup (Identity + Questionnaire + Conditions)
// ═══════════════════════════════════════════════════════════════════════════════

export async function renderScenarioSetup(container) {
  const d = S.currentData;

  // ── Item ID & Title ──
  const idGroup = el("div", { class: "field-group" });
  const idInput = el("input", { type: "text", value: d.item_id || "", placeholder: "e.g. F1-001" });
  idInput.addEventListener("input", () => {
    S.currentData._pendingItemId = idInput.value.trim();
    markDirty();
  });
  idGroup.append(el("label", {}, "Item ID"), idInput,
    el("p", { class: "help-text" }, "Unique identifier (e.g. F1-001). Letters, numbers, hyphens, dots, underscores only."));

  const titleGroup = el("div", { class: "field-group" });
  const titleInput = el("input", { type: "text", value: d.title || "", placeholder: "e.g. Initialization \u2014 empty form, single date field" });
  titleInput.addEventListener("input", () => {
    S.currentData.title = titleInput.value;
    markDirty();
    updateChecklist();
    document.getElementById("editor-item-title").textContent = titleInput.value || "(untitled)";
  });
  titleGroup.append(el("label", {}, "Item title"), titleInput,
    el("p", { class: "help-text" }, "Short, descriptive. Used in tables and filenames."));

  // ── Primary Delta Type (derived from per-field annotations) ──
  if (!d.primary_delta_type && d.family) d.primary_delta_type = d.family;
  const deltaGroup = el("div", { class: "field-group", style: "margin-top:10px;" });
  deltaGroup.append(
    el("label", {}, "Primary delta type"),
    el("p", { class: "help-text", style: "margin-bottom:8px;" },
      "Derived from per-field delta labels in Section 3. The dominant changed-slot semantic becomes the item\u2019s primary classification. History dependence and inconsistent prior state are tracked separately via evidence properties and state condition."),
    el("div", {
      id: "derived-delta-badge",
      style: "font-size:13px;padding:6px 10px;border:1px dashed var(--border);border-radius:8px;color:var(--text-2);",
    }, d.primary_delta_type || "no field changes yet"),
  );

  // ── Contrast Role ──
  const roleGroup = el("div", { class: "field-group" });
  const roleOpts = el("div", { style: "display:flex; gap:10px; margin-top:4px;" });
  for (const r of CONTRAST_ROLES) {
    const opt = el("label", { style: "flex:1; display:flex; gap:8px; align-items:flex-start; padding:10px; border:1.5px solid var(--border); border-radius:8px; cursor:pointer; transition: all .15s;" });
    const radio = el("input", { type: "radio", name: `cr_${d.item_id}`, value: r.value });
    if (d.contrast_role === r.value) radio.checked = true;
    radio.addEventListener("change", () => {
      S.currentData.contrast_role = r.value;
      markDirty();
      roleOpts.querySelectorAll("label").forEach(l => { l.style.background = ""; l.style.borderColor = "var(--border)"; });
      opt.style.background = "#eef2ff";
      opt.style.borderColor = "#818cf8";
    });
    if (d.contrast_role === r.value) { opt.style.background = "#eef2ff"; opt.style.borderColor = "#818cf8"; }
    opt.append(radio, el("div", {}, el("div", { style: "font-size:13px;font-weight:600;" }, r.label), el("div", { style: "font-size:11px;color:var(--help);font-style:italic;margin-top:2px;" }, r.desc)));
    roleOpts.appendChild(opt);
  }
  roleGroup.append(el("label", {}, "Contrast role"), roleOpts);

  // ── Difficulty tier ──
  if (!d.difficulty_tier) {
    const legacyDifficulty = (d.materialization?.difficulty || "").toLowerCase();
    d.difficulty_tier = legacyDifficulty === "easy"
      ? "anchor"
      : legacyDifficulty === "medium"
        ? "challenge"
        : legacyDifficulty === "hard"
          ? "hard"
          : "challenge";
  }
  if (!d.materialization) d.materialization = {};
  d.materialization.difficulty = d.difficulty_tier;

  const tierGroup = el("div", { class: "field-group" });
  const tierOpts = el("div", { style: "display:flex; gap:10px; margin-top:4px;" });
  for (const tier of DIFFICULTY_TIERS) {
    const opt = el("label", { style: "flex:1; display:flex; gap:8px; align-items:flex-start; padding:10px; border:1.5px solid var(--border); border-radius:8px; cursor:pointer; transition: all .15s;" });
    const radio = el("input", { type: "radio", name: `dt_${d.item_id}`, value: tier.value });
    if (d.difficulty_tier === tier.value) radio.checked = true;
    radio.addEventListener("change", () => {
      S.currentData.difficulty_tier = tier.value;
      if (!S.currentData.materialization) S.currentData.materialization = {};
      S.currentData.materialization.difficulty = tier.value;
      markDirty();
      updateChecklist();
      tierOpts.querySelectorAll("label").forEach(l => { l.style.background = ""; l.style.borderColor = "var(--border)"; });
      opt.style.background = "#ecfeff";
      opt.style.borderColor = "#0891b2";
    });
    if (d.difficulty_tier === tier.value) { opt.style.background = "#ecfeff"; opt.style.borderColor = "#0891b2"; }
    opt.append(radio, el("div", {}, el("div", { style: "font-size:13px;font-weight:600;" }, tier.label), el("div", { style: "font-size:11px;color:var(--help);font-style:italic;margin-top:2px;" }, tier.desc)));
    tierOpts.appendChild(opt);
  }
  tierGroup.append(
    el("label", {}, "Difficulty tier"),
    tierOpts,
    el("p", { class: "help-text" }, "Explicit paper-facing tier used for balancing and reporting. This also drives the materialized runtime difficulty label."),
  );

  // ── Questionnaire ──
  if (!d.questionnaire) d.questionnaire = {};
  const sources = S.questionnaireNames.length ? S.questionnaireNames : await apiGet("/api/questionnaires").catch(() => []);

  const srcGroup = el("div", { class: "field-group" });
  const srcSel = el("select", { class: "code-select" });
  for (const s of sources) srcSel.append(el("option", { value: s, ...(s === d.questionnaire.source ? { selected: "" } : {}) }, s));
  srcSel.addEventListener("change", async () => {
    d.questionnaire.source = srcSel.value;
    S.qFieldsCache[srcSel.value] = undefined;
    S.qFieldMetaCache[srcSel.value] = undefined;
    markDirty();
    refreshContextPreview();
  });
  srcGroup.append(el("label", {}, "Questionnaire"), srcSel, el("p", { class: "help-text" }, "Which form is this item based on?"));

  const notesGroup = el("div", { class: "field-group" });
  const notesInput = el("input", { type: "text", value: d.questionnaire.notes || "", placeholder: "optional free-text note" });
  notesInput.addEventListener("input", () => { d.questionnaire.notes = notesInput.value; markDirty(); });
  notesGroup.append(el("label", {}, "Questionnaire notes"), notesInput);

  // ── Scenario picker ──
  const scenarioGroup = el("div", { class: "field-group" });
  const scenarioSel = el("select", { class: "code-select" });
  scenarioSel.append(el("option", { value: "" }, "— no scenario —"));
  for (const sc of S.allScenarios) {
    scenarioSel.append(el("option", { value: sc.scenario_id, ...(sc.scenario_id === (d.scenario || "") ? { selected: "" } : {}) }, sc.scenario_id.replace(/_/g, " ")));
  }
  scenarioSel.addEventListener("change", () => { d.scenario = scenarioSel.value || undefined; markDirty(); });
  scenarioGroup.append(el("label", {}, "Scenario"), scenarioSel, el("p", { class: "help-text" }, "Which big-picture scenario does this item belong to?"));

  // ── Conditions ──
  function conditionPicker(id, items, current, onChange) {
    const wrap = el("div", { class: "condition-options" });
    for (const item of items) {
      const opt = el("div", { class: `condition-option${current === item.value ? " selected" : ""}`, "data-value": item.value });
      const radio = el("input", { type: "radio", name: id, value: item.value });
      if (current === item.value) radio.checked = true;
      radio.addEventListener("change", () => {
        wrap.querySelectorAll(".condition-option").forEach(o => o.classList.remove("selected"));
        opt.classList.add("selected");
        onChange(item.value);
        markDirty();
      });
      opt.append(radio, el("div", {}, el("div", { class: "co-label" }, item.label), el("div", { class: "co-desc" }, item.desc)));
      wrap.appendChild(opt);
    }
    return wrap;
  }

  const grid = el("div", { class: "condition-grid", style: "margin-top:14px;" });
  const stateBlock = el("div", { class: "condition-block" });
  stateBlock.append(
    el("label", {}, "Prior state condition"),
    conditionPicker(`sc_${d.item_id}`, STATE_CONDITIONS, d.state_condition, v => {
      d.state_condition = v;
      refreshContextPreview();
    }),
    el("p", { class: "help-text", style: "margin-top:6px;" }, "What has already been recorded when the agent receives this utterance?"),
  );

  // ── Evidence properties (replace old history condition) ──
  if (!d.evidence) d.evidence = { ...EVIDENCE_DEFAULTS };
  if (d.evidence.history_required == null) d.evidence.history_required = EVIDENCE_DEFAULTS.history_required;
  if (d.evidence.support_distance == null) d.evidence.support_distance = EVIDENCE_DEFAULTS.support_distance;
  if (d.evidence.conflict_present == null) d.evidence.conflict_present = EVIDENCE_DEFAULTS.conflict_present;

  const evidenceBlock = el("div", { class: "condition-block" });
  const evidenceContent = el("div", { style: "display:flex; flex-direction:column; gap:12px; margin-top:6px;" });

  // history_required toggle
  const hrWrap = el("div", { style: "display:flex; align-items:center; gap:10px;" });
  const hrCb = el("input", { type: "checkbox" });
  hrCb.checked = !!d.evidence.history_required;
  hrCb.addEventListener("change", () => { d.evidence.history_required = hrCb.checked; markDirty(); });
  hrWrap.append(hrCb, el("span", { style: "font-size:13px;font-weight:600;" }, "History required"));
  const hrDesc = el("p", { class: "help-text", style: "margin:0;" }, "Does the system need prior dialogue turns (not just the current utterance) to produce the gold state?");

  // support_distance number input
  const sdWrap = el("div", { style: "display:flex; align-items:center; gap:10px;" });
  const sdInput = el("input", { type: "number", min: "0", value: String(d.evidence.support_distance || 0), style: "width:70px;" });
  sdInput.addEventListener("input", () => { d.evidence.support_distance = parseInt(sdInput.value, 10) || 0; markDirty(); });
  sdWrap.append(el("span", { style: "font-size:13px;font-weight:600;" }, "Support distance"), sdInput, el("span", { style: "font-size:11px;color:var(--help);" }, "turns back"));
  const sdDesc = el("p", { class: "help-text", style: "margin:0;" }, "0 = current utterance is enough. 1 = evidence is one turn back. 3+ = distant history.");

  // conflict_present toggle
  const cpWrap = el("div", { style: "display:flex; align-items:center; gap:10px;" });
  const cpCb = el("input", { type: "checkbox" });
  cpCb.checked = !!d.evidence.conflict_present;
  cpCb.addEventListener("change", () => { d.evidence.conflict_present = cpCb.checked; markDirty(); });
  cpWrap.append(cpCb, el("span", { style: "font-size:13px;font-weight:600;" }, "Conflict present"));
  const cpDesc = el("p", { class: "help-text", style: "margin:0;" }, "Does the history contain claims that contradict the current utterance or each other?");

  evidenceContent.append(
    el("div", {}, hrWrap, hrDesc),
    el("div", {}, sdWrap, sdDesc),
    el("div", {}, cpWrap, cpDesc),
  );

  evidenceBlock.append(
    el("label", {}, "Evidence properties"),
    el("p", { class: "help-text", style: "margin-bottom:6px;" }, "History dependence is an orthogonal evidence axis, not a primary delta type. Use these fields to record where the justification for the gold state lives."),
    evidenceContent,
  );
  grid.append(stateBlock, evidenceBlock);

  container.append(
    el("div", { class: "field-row" }, idGroup, titleGroup),
    deltaGroup,
    el("div", { class: "field-row" }, roleGroup, tierGroup),
    el("div", { class: "field-row" }, srcGroup, notesGroup),
    scenarioGroup,
    grid,
  );
}

function applyDeltaDefaults(deltaType) {
  const defs = DELTA_DEFAULTS[deltaType];
  if (!defs) return;
  const d = S.currentData;
  if (d.status !== "template") return;
  if (defs.state_condition) d.state_condition = defs.state_condition;
  if (!d.evidence) d.evidence = { ...EVIDENCE_DEFAULTS };
  if (defs.history_required != null) d.evidence.history_required = defs.history_required;
  if (defs.support_distance != null) d.evidence.support_distance = defs.support_distance;
  if (defs.conflict_present != null) d.evidence.conflict_present = defs.conflict_present;
}

// Keep old name as alias for any remaining callers
function applyFamilyDefaults(family) { applyDeltaDefaults(family); }

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 2: Context & Utterance (Context Preview + Utterance)
// ═══════════════════════════════════════════════════════════════════════════════

export function renderContextAndUtterance(container) {
  // ── Context panel ──
  const ctxWrap = el("div", { id: "context-preview", style: "margin-top:12px;" });
  S._ctxContainer = ctxWrap;
  container.appendChild(ctxWrap);
  refreshContextPreview();

  // ── Utterance ──
  const d = S.currentData;
  if (!d.current_utterance) d.current_utterance = { speaker: "user", text: "" };
  const uttWrap = el("div", { class: "field-group", style: "margin-top:20px;" });
  const ta = el("textarea", { id: "utt-textarea", placeholder: "Write the exact user utterance this item is testing\u2026" });
  ta.value = d.current_utterance.text || "";
  ta.addEventListener("input", () => { d.current_utterance.text = ta.value; markDirty(); updateChecklist(); });

  const uttLabelRow = el("div", {});
  uttLabelRow.append(
    el("label", {}, "User utterance"),
    el("p", { class: "help-text", style: "margin-bottom:6px;" }, "Write the exact user message. Include specific values, names, dates. This is the core creative work."),
  );
  uttWrap.append(uttLabelRow, ta);
  container.appendChild(uttWrap);
}

// ── Context preview helpers ──────────────────────────────────────────────────

export function renderContextPane(target, { title, ref, status, buttonText, buttonClass, onButtonClick, onChangeRef }) {
  const pane = el("div", { class: "context-pane" });
  const statusEl = makeCtxStatus(status || "loading");
  const btnGroup = el("div", { class: "context-pane-actions" });
  btnGroup.append(
    statusEl,
    el("span", { class: "ref-path" }, ref || "\u2014 no ref \u2014"),
    el("button", { class: buttonClass || "btn sm", type: "button", onclick: onButtonClick }, buttonText),
  );
  if (onChangeRef) {
    btnGroup.append(el("button", { class: "btn sm", type: "button", onclick: onChangeRef }, "\u21c4 Change ref"));
  }
  const header = el("div", { class: "context-pane-header" }, title, btnGroup);
  const body = el("div", { class: "context-pane-body" });
  pane.append(header, body);
  target.appendChild(pane);
  return body;
}

export function renderStateBody(body, ctx) {
  const answers = ctx?.questionnaire_answers;
  if (!answers) {
    body.innerHTML = `<span class="context-empty">Empty form \u2014 no fields pre-filled.</span>`;
  } else {
    const table = el("table", { class: "kv-table" });
    for (const [k, v] of Object.entries(answers)) {
      table.append(el("tr", {}, el("td", {}, k), el("td", {}, JSON.stringify(v))));
    }
    body.appendChild(table);
  }
  if (ctx?.description) body.appendChild(el("p", { style: "font-size:11px;color:var(--text-3);margin-top:6px;" }, ctx.description));
}

export function renderHistoryBody(body, ctx) {
  const turns = ctx?.turns || [];
  if (!turns.length) {
    body.innerHTML = `<span class="context-empty">No prior turns \u2014 first message in session.</span>`;
    return;
  }
  for (const t of turns) {
    body.append(el("div", { class: `history-turn ${t.speaker}` }, el("div", { class: "speaker" }, t.speaker), el("div", { class: "text" }, t.text)));
  }
}

export function refreshContextPreview() {
  if (!S._ctxContainer || !S.currentData) return;
  S._ctxContainer.innerHTML = "";

  const guide = el("div", { class: "context-guide-banner" },
    el("p", { style: "font-size:12px;color:var(--text-2);line-height:1.6;margin-bottom:8px;" },
      "Each item needs a ", el("strong", {}, "Form State"), " and ",
      el("strong", {}, "Conversation History"), " context. ",
      "Click ", el("strong", {}, "Edit"), " to modify, or ",
      el("strong", {}, "Create"), " to start a new one."),
  );
  S._ctxContainer.appendChild(guide);

  // State pane
  const stateExists = !!S.currentData.state_ref;
  const stateBody = renderContextPane(S._ctxContainer, {
    title: "Form state (what\u2019s already recorded)",
    ref: S.currentData.state_ref,
    status: "loading",
    buttonText: stateExists ? "\u270f\ufe0f Edit state" : "\uff0b Create state",
    buttonClass: stateExists ? "btn sm" : "btn sm primary",
    onButtonClick: () => openInlineContextModal("state"),
    onChangeRef: () => showChangeRefPicker("state"),
  });

  // History pane
  const histExists = !!S.currentData.history_ref;
  const historyBody = renderContextPane(S._ctxContainer, {
    title: "Conversation history (what the agent has already seen)",
    ref: S.currentData.history_ref,
    status: "loading",
    buttonText: histExists ? "\u270f\ufe0f Edit history" : "\uff0b Create history",
    buttonClass: histExists ? "btn sm" : "btn sm primary",
    onButtonClick: () => openInlineContextModal("history"),
    onChangeRef: () => showChangeRefPicker("history"),
  });

  if (stateExists) {
    apiGet(`/api/contexts/state/${S.currentData.state_ref}`).then(ctx => {
      renderStateBody(stateBody, ctx);
      stateBody.closest(".context-pane").querySelector(".ctx-status")?.replaceWith(makeCtxStatus(ctx.questionnaire_answers ? "ok" : "empty"));
    }).catch(() => {
      stateBody.innerHTML = `<span class="context-empty">\u26a0 Could not load state context.</span>`;
      stateBody.closest(".context-pane").querySelector(".ctx-status")?.replaceWith(makeCtxStatus("missing"));
    });
    // Check if state is shared by other items
    apiGet(`/api/contexts/state/${S.currentData.state_ref}/usage`).then(usage => {
      const others = (usage.items || []).filter(u => u.item_id !== S.currentData.item_id);
      if (others.length > 0) {
        const badge = el("span", {
          class: "shared-badge",
          title: `Also used by: ${others.map(u => u.item_id).join(", ")}`,
          style: "display:inline-flex;align-items:center;gap:3px;font-size:10px;font-weight:600;color:#92400e;background:#fef3c7;border:1px solid #f59e0b;border-radius:4px;padding:1px 6px;margin-left:6px;cursor:help;",
        }, `\ud83d\udd17 shared (${others.length + 1} items)`);
        stateBody.closest(".context-pane").querySelector(".context-pane-actions")?.appendChild(badge);
      }
    }).catch(() => {});
  } else {
    stateBody.innerHTML = `<span class="context-empty">No state context linked yet. Click <strong>Create state</strong> above.</span>`;
    stateBody.closest(".context-pane").querySelector(".ctx-status")?.replaceWith(makeCtxStatus("missing"));
  }

  if (histExists) {
    apiGet(`/api/contexts/history/${S.currentData.history_ref}`).then(ctx => {
      renderHistoryBody(historyBody, ctx);
      historyBody.closest(".context-pane").querySelector(".ctx-status")?.replaceWith(makeCtxStatus((ctx.turns?.length || 0) > 0 ? "ok" : "empty"));
    }).catch(() => {
      historyBody.innerHTML = `<span class="context-empty">\u26a0 Could not load history context.</span>`;
      historyBody.closest(".context-pane").querySelector(".ctx-status")?.replaceWith(makeCtxStatus("missing"));
    });
    // Check if history is shared by other items
    apiGet(`/api/contexts/history/${S.currentData.history_ref}/usage`).then(usage => {
      const others = (usage.items || []).filter(u => u.item_id !== S.currentData.item_id);
      if (others.length > 0) {
        const badge = el("span", {
          class: "shared-badge",
          title: `Also used by: ${others.map(u => u.item_id).join(", ")}`,
          style: "display:inline-flex;align-items:center;gap:3px;font-size:10px;font-weight:600;color:#92400e;background:#fef3c7;border:1px solid #f59e0b;border-radius:4px;padding:1px 6px;margin-left:6px;cursor:help;",
        }, `\ud83d\udd17 shared (${others.length + 1} items)`);
        historyBody.closest(".context-pane").querySelector(".context-pane-actions")?.appendChild(badge);
      }
    }).catch(() => {});
  } else {
    historyBody.innerHTML = `<span class="context-empty">No history context linked yet. Click <strong>Create history</strong> above.</span>`;
    historyBody.closest(".context-pane").querySelector(".ctx-status")?.replaceWith(makeCtxStatus("missing"));
  }
}

export function makeCtxStatus(state) {
  const colors = { ok: "#10b981", empty: "#f59e0b", missing: "#94a3b8", loading: "#94a3b8" };
  const labels = { ok: "\u2713 Loaded", empty: "\u25cb Empty", missing: "\u2014 Not set", loading: "\u2026" };
  const dot = el("span", { class: "ctx-status", style: `display:inline-flex;align-items:center;gap:4px;font-size:10px;font-weight:600;color:${colors[state] || colors.missing};` },
    el("span", { style: `width:6px;height:6px;border-radius:50%;background:${colors[state] || colors.missing};` }),
    labels[state] || "\u2014",
  );
  return dot;
}

// =============================================================================
// SECTION 3: Expected Outcome & Difficulty
// =============================================================================

export async function renderExpectedOutcomeAndDifficulty(container) {
  const d = S.currentData;
  ensureExpectedOutcome(d);
  ensureDifficultyProfile(d);
  const fieldsMeta = await getQFieldMeta(d.questionnaire?.source);
  const fieldMetaMap = Object.fromEntries(fieldsMeta.map(f => [f.id, f]));
  const fields = fieldsMeta.map(f => f.id);
  const qTree = await getQTree(d.questionnaire?.source);

  // ── Resolve prior state from context ref ──
  let priorAnswers = {};
  try {
    const resolved = await apiGet(`/api/items/${d.item_id}/resolve-prior-state`);
    priorAnswers = resolved.questionnaire_answers || {};
  } catch { /* empty prior */ }

  // Initialize gold_resulting_state from prior + preserve existing edits
  if (!d.gold_resulting_state) d.gold_resulting_state = {};

  // -- 3a: Gold Resulting State (inline form + JSON mode) --
  const goldHeadingRow = el("div", { style: "display:flex;align-items:center;gap:10px;margin:12px 0 4px;" });
  goldHeadingRow.appendChild(el("h3", { style: "font-size:13px;font-weight:600;color:var(--text);margin:0;" }, "Gold Resulting State"));

  // Mode toggle: Form | JSON
  const modeToggle = el("div", { class: "gold-mode-toggle" });
  const btnForm = el("button", { class: "gold-mode-btn active", "data-mode": "form" }, "Form");
  const btnJson = el("button", { class: "gold-mode-btn", "data-mode": "json" }, "JSON");
  modeToggle.append(btnForm, btnJson);
  goldHeadingRow.appendChild(modeToggle);
  container.appendChild(goldHeadingRow);

  container.appendChild(el("p", { style: "font-size:12px;color:var(--help);margin-bottom:6px;line-height:1.5;" },
    "This editor is whole-record and starts from the prior state. Edit it into the correct full record AFTER the utterance. ",
    "Preserve unrelated fields unless the utterance justifies a change; collateral edits are benchmark errors. ",
    "Gated fields appear when their gate question is answered, and repeat groups expand based on count."));

  const formWrap = el("div", { class: "gold-form" });
  container.appendChild(formWrap);
  let dimensionGridHost = null;

  // JSON editor (hidden by default)
  const jsonWrap = el("div", { class: "gold-json-wrap", style: "display:none;" });
  const jsonError = el("div", { class: "gold-json-error" });
  const jsonArea = el("textarea", { class: "gold-json-textarea", spellcheck: "false", placeholder: "Paste or edit gold_resulting_state JSON here…" });
  jsonWrap.append(jsonError, jsonArea);
  container.appendChild(jsonWrap);

  let goldMode = "form"; // "form" | "json"

  /**
   * Build an ordered gold-state object following the questionnaire tree.
   * Gated sections become nested objects keyed by their label/id,
   * and repeat groups appear at their natural position — not appended last.
   */
  function buildOrderedGoldJson() {
    const gold = d.gold_resulting_state || {};
    const result = {};

    function walk(questions) {
      for (const q of questions) {
        const st = q.structure_type || "regular";
        if (st === "regular") {
          if (q.id in gold) result[q.id] = gold[q.id];
        } else if (st === "group") {
          walk(q.fields || q.questions || []);
        } else if (st === "gate") {
          const gate = q.gate;
          if (!gate) continue;
          const children = q.fields || q.questions || [];
          // Always include gated children that exist in gold
          walk(children);
        } else if (st === "repeat_group") {
          if (q.id in gold && Array.isArray(gold[q.id])) {
            // Filter out instances where every value is null/empty (form scaffolding)
            const filtered = gold[q.id].filter(inst =>
              Object.values(inst).some(v => v != null && v !== "")
            );
            if (filtered.length > 0) result[q.id] = filtered;
          }
        }
      }
    }
    walk(qTree);

    // Append any extra keys not in the tree (safety net),
    // but skip flat repeat-group child fields that leaked in.
    const rgChildIds = collectRepeatGroupChildIds(qTree);
    for (const k of Object.keys(gold)) {
      if (!(k in result) && !rgChildIds.has(k)) {
        const v = gold[k];
        if (Array.isArray(v)) {
          const filtered = v.filter(inst =>
            typeof inst === "object" && inst !== null
              ? Object.values(inst).some(val => val != null && val !== "")
              : inst != null
          );
          if (filtered.length > 0) result[k] = filtered;
        } else {
          result[k] = v;
        }
      }
    }
    return result;
  }

  function setGoldMode(mode) {
    goldMode = mode;
    btnForm.classList.toggle("active", mode === "form");
    btnJson.classList.toggle("active", mode === "json");
    formWrap.style.display = mode === "form" ? "" : "none";
    jsonWrap.style.display = mode === "json" ? "" : "none";
    if (mode === "json") {
      jsonArea.value = JSON.stringify(buildOrderedGoldJson(), null, 2);
      jsonError.textContent = "";
    } else {
      // Switching back to form — try to apply JSON edits first
      applyJsonToGold();
      rebuildForm();
      updateChangeSummary();
    }
  }

  function applyJsonToGold() {
    try {
      const parsed = JSON.parse(jsonArea.value);
      if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
        jsonError.textContent = "JSON must be an object (not array or primitive).";
        return false;
      }
      d.gold_resulting_state = parsed;
      syncExpectedOutcome();
      markDirty();
      jsonError.textContent = "";
      return true;
    } catch (e) {
      jsonError.textContent = "Invalid JSON: " + e.message;
      return false;
    }
  }

  jsonArea.addEventListener("input", () => {
    // Live-validate and apply on each edit
    applyJsonToGold();
  });

  btnForm.addEventListener("click", () => setGoldMode("form"));
  btnJson.addEventListener("click", () => setGoldMode("json"));

  // Change summary (collapsible)
  const changeSummary = el("div", { id: "change-summary-panel", class: "change-summary-panel" });
  const changeSummaryHeader = el("div", { class: "change-summary-header" });
  const changeSummaryToggle = el("span", { class: "change-summary-toggle" }, "\u25b8");
  const changeSummaryTitle = el("h4", { style: "font-size:12px;font-weight:600;color:var(--text);margin:0;" }, "Change Summary");
  const changeSummaryCount = el("span", { id: "change-summary-count", style: "font-size:11px;color:var(--help);font-style:italic;" }, "");
  changeSummaryHeader.append(changeSummaryToggle, changeSummaryTitle, changeSummaryCount);
  const changeSummaryBody = el("div", { id: "change-summary-body", style: "display:none;" });
  changeSummaryHeader.addEventListener("click", () => {
    const collapsed = changeSummaryBody.style.display === "none";
    changeSummaryBody.style.display = collapsed ? "" : "none";
    changeSummaryToggle.textContent = collapsed ? "\u25be" : "\u25b8";
  });
  changeSummary.append(changeSummaryHeader, changeSummaryBody);
  const derivedVariablesPanel = el("div", {
    id: "derived-variables-panel",
    style: "margin-top:8px;padding:8px;border:1px solid var(--border);border-radius:6px;background:var(--bg);",
  });

  // Collect actual repeat_group IDs from the questionnaire tree so we can
  // distinguish them from regular fields whose values happen to be arrays
  // (e.g. multiple_choice fields like ["Speeding", "Distraction"]).
  const repeatGroupIds = collectRepeatGroupIds(qTree);

  function updateChangeSummary() {
    const body = document.getElementById("change-summary-body");
    const countEl = document.getElementById("change-summary-count");
    if (!body) return;
    body.innerHTML = "";
    const gold = d.gold_resulting_state || {};

    // Collect all keys currently in gold
    const allKeys = Object.keys(gold);
    let setCount = 0, changedCount = 0, clearedCount = 0, preservedCount = 0;
    let repeatChangedCount = 0;
    const changedRepeatGroups = new Set();

    // --- Flat field changes (includes multiple_choice arrays) ---
    for (const key of allKeys) {
      const { value: pv } = effectivePriorForField(key, priorAnswers[key]);
      const rv = gold[key];
      // Only skip actual repeat groups — compare everything else as flat
      if (repeatGroupIds.has(key)) continue;
      const chKind = classifyChange(pv, rv);
      // Skip fields with delta_type "keep" — they are intentionally unchanged
      const dt = deltaTypeGet(key);
      if (!chKind || dt === "keep") { preservedCount++; continue; }
      let kind, color;
      if (chKind === "SET")     { kind = "SET";     color = "#10b981"; setCount++; }
      else if (chKind === "CLEAR") { kind = "CLEAR";   color = "#ef4444"; clearedCount++; }
      else                      { kind = "CHANGED"; color = "#f59e0b"; changedCount++; }

      const meta = fieldMetaMap[key];
      const label = meta?.label && meta.label !== key ? `${key} \u2014 ${meta.label.slice(0, 40)}` : key;
      const src = annotationGet(key);
      const srcInfo = src ? VALUE_SOURCES.find(s => s.value === src) : null;
      const dtInfo = dt ? FIELD_DELTA_TYPES.find(d => d.value === dt) : null;
      body.appendChild(el("div", { style: `display:flex;gap:8px;align-items:center;padding:2px 0;font-size:11px;` },
        el("span", { style: `font-weight:700;color:${color};min-width:55px;` }, kind),
        dtInfo ? el("span", { style: `font-size:9px;font-weight:700;color:${dtInfo.color};border:1px solid ${dtInfo.color};border-radius:8px;padding:0 5px;` }, dtInfo.label) : null,
        el("span", { style: "color:var(--text-2);" }, label),
        pv != null && pv !== "" ? el("span", { style: "color:var(--text-3);text-decoration:line-through;" }, JSON.stringify(pv)) : null,
        rv != null && rv !== "" ? el("span", { style: "color:var(--text);" }, "\u2192 " + JSON.stringify(rv)) : null,
        srcInfo ? el("span", { style: `font-size:9px;font-weight:700;color:${srcInfo.color};border:1px solid ${srcInfo.color};border-radius:8px;padding:0 5px;` }, srcInfo.label) : null,
      ));
    }

    // --- Repeat group (nested) changes — only actual repeat_groups ---
    for (const key of allKeys) {
      if (!repeatGroupIds.has(key)) continue;
      const rv = Array.isArray(gold[key]) ? gold[key] : [];
      const priorArr = Array.isArray(priorAnswers[key]) ? priorAnswers[key] : [];
      const maxLen = Math.max(rv.length, priorArr.length);
      if (maxLen === 0) continue;

      // Section header for the repeat group
      body.appendChild(el("div", { style: "margin-top:6px;font-size:11px;font-weight:700;color:var(--text-2);border-top:1px solid var(--border);padding-top:4px;" },
        `${key} (${rv.length} instance${rv.length !== 1 ? "s" : ""})`));

      if (rv.length !== priorArr.length) {
        const countChange = rv.length > priorArr.length ? "added" : "removed";
        const countDiff = Math.abs(rv.length - priorArr.length);
        body.appendChild(el("div", { style: "font-size:10px;color:var(--text-3);padding-left:12px;" },
          `Instance count ${countChange}: ${priorArr.length} \u2192 ${rv.length} (${countDiff} ${countChange})`));
      }

      for (let i = 0; i < maxLen; i++) {
        const goldInst = rv[i] || {};
        const priorInst = priorArr[i] || {};
        const fieldKeys = new Set([...Object.keys(goldInst), ...Object.keys(priorInst)]);
        for (const fid of fieldKeys) {
          const { value: pv } = effectivePriorForField(fid, priorInst[fid], { repeatGroupId: key, repeatIndex: i });
          const cv = goldInst[fid];
          const chKind = classifyChange(pv, cv);
          // Skip fields with delta_type "keep" — they are intentionally unchanged
          const dt = deltaTypeRepeatGet(key, i, fid);
          if (!chKind || dt === "keep") { preservedCount++; continue; }
          let kind, color;
          if (chKind === "SET")     { kind = "SET";     color = "#10b981"; setCount++; }
          else if (chKind === "CLEAR") { kind = "CLEAR";   color = "#ef4444"; clearedCount++; }
          else                      { kind = "CHANGED"; color = "#f59e0b"; changedCount++; }
          repeatChangedCount++;
          changedRepeatGroups.add(key);

          const label = `${key}[${i}].${fid}`;
          const src = annotationRepeatGet(key, i, fid);
          const srcInfo = src ? VALUE_SOURCES.find(s => s.value === src) : null;
          const dtInfo = dt ? FIELD_DELTA_TYPES.find(d => d.value === dt) : null;
          body.appendChild(el("div", { style: `display:flex;gap:8px;align-items:center;padding:2px 0 2px 12px;font-size:11px;` },
            el("span", { style: `font-weight:700;color:${color};min-width:55px;` }, kind),
            dtInfo ? el("span", { style: `font-size:9px;font-weight:700;color:${dtInfo.color};border:1px solid ${dtInfo.color};border-radius:8px;padding:0 5px;` }, dtInfo.label) : null,
            el("span", { style: "color:var(--text-2);" }, label),
            pv != null && pv !== "" ? el("span", { style: "color:var(--text-3);text-decoration:line-through;" }, JSON.stringify(pv)) : null,
            cv != null && cv !== "" ? el("span", { style: "color:var(--text);" }, "\u2192 " + JSON.stringify(cv)) : null,
            srcInfo ? el("span", { style: `font-size:9px;font-weight:700;color:${srcInfo.color};border:1px solid ${srcInfo.color};border-radius:8px;padding:0 5px;` }, srcInfo.label) : null,
          ));
        }
      }
    }

    const total = setCount + changedCount + clearedCount;
    if (countEl) countEl.textContent = total > 0
      ? `(${total} change${total !== 1 ? "s" : ""}: ${setCount} set, ${changedCount} changed, ${clearedCount} cleared, ${preservedCount} preserved)`
      : `(no changes)`;

    if (total === 0) {
      body.appendChild(el("p", { style: "font-size:11px;color:var(--text-3);font-style:italic;margin:2px 0;" }, "All fields unchanged from prior state."));
    }

    // Derive primary_delta_type from per-field delta types (flat + repeat groups)
    const dtCounts = {};
    for (const key of allKeys) {
      if (repeatGroupIds.has(key)) {
        // Collect delta types from repeat group instance fields
        const annArr = d.gold_annotations?.[key];
        if (Array.isArray(annArr)) {
          for (const inst of annArr) {
            if (!inst || typeof inst !== "object") continue;
            for (const fid of Object.keys(inst)) {
              const dt = inst[fid]?.delta_type;
              if (dt && dt !== "keep") dtCounts[dt] = (dtCounts[dt] || 0) + 1;
            }
          }
        }
        continue;
      }
      const dt = deltaTypeGet(key);
      if (dt && dt !== "keep") dtCounts[dt] = (dtCounts[dt] || 0) + 1;
    }
    const sorted = Object.entries(dtCounts).sort((a, b) => b[1] - a[1]);
    const derived = sorted.length > 0 ? sorted[0][0] : null;
    const revisionOperation = (() => {
      const ops = new Set();
      for (const dt of Object.keys(dtCounts)) {
        if (dt === "add") ops.add("append");
        else if (dt === "refine") ops.add("refine");
        else if (dt === "correct") ops.add("overwrite");
        else if (dt === "retract") ops.add("retract");
      }
      if (ops.size === 0) {
        if (setCount) ops.add("append");
        if (changedCount) ops.add("overwrite");
        if (clearedCount) ops.add("retract");
      }
      if (ops.size === 0) return "no_change";
      if (ops.size === 1) return Array.from(ops)[0];
      return "mixed";
    })();
    const repeatGroupNames = Array.from(changedRepeatGroups).sort();
    const repeatGroupInvolvement = repeatGroupNames.length === 0
      ? "none"
      : repeatGroupNames.length === 1
        ? "single_group"
        : "multi_group";
    d.derived_variables = {
      changed_leaf_field_count: total,
      primary_delta_type: derived || d.primary_delta_type || "add",
      prior_state_condition: d.state_condition || "unknown",
      history_required: !!d.evidence?.history_required,
      support_distance: Number(d.evidence?.support_distance || 0),
      conflict_present: !!d.evidence?.conflict_present,
      repeat_group_involvement: repeatGroupInvolvement,
      repeat_group_involved: repeatGroupNames.length > 0,
      repeat_group_names: repeatGroupNames,
      repeat_group_changed_leaf_count: repeatChangedCount,
      revision_operation: revisionOperation,
    };
    if (derived) {
      d.primary_delta_type = derived;
      d.family = derived;
    }
    // Update the derived delta badge in Section 1
    const derivedBadge = document.getElementById("derived-delta-badge");
    if (derivedBadge) {
      if (sorted.length > 0) {
        const parts = sorted.map(([dt, n]) => {
          const info = FIELD_DELTA_TYPES.find(f => f.value === dt);
          return `<span style="color:${info?.color || "var(--text)"};font-weight:600;">${n} ${dt}</span>`;
        });
        derivedBadge.innerHTML = parts.join(", ");
      } else {
        derivedBadge.textContent = "no field changes yet";
      }
    }
    if (derivedVariablesPanel) {
      const dv = d.derived_variables;
      derivedVariablesPanel.innerHTML = `
        <div style="font-size:11px;font-weight:700;color:var(--text-2);text-transform:uppercase;margin-bottom:6px;">Derived Variables</div>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:6px;font-size:11px;color:var(--text-2);">
          <div><strong>${dv.changed_leaf_field_count}</strong><br><span style="color:var(--text-3);">changed leaves</span></div>
          <div><strong>${dv.primary_delta_type}</strong><br><span style="color:var(--text-3);">primary delta</span></div>
          <div><strong>${dv.revision_operation}</strong><br><span style="color:var(--text-3);">revision operation</span></div>
          <div><strong>${dv.repeat_group_involvement}</strong><br><span style="color:var(--text-3);">repeat-group involvement</span></div>
          <div><strong>${dv.prior_state_condition}</strong><br><span style="color:var(--text-3);">prior state</span></div>
          <div><strong>${dv.history_required ? "yes" : "no"} / ${dv.support_distance}</strong><br><span style="color:var(--text-3);">history / support distance</span></div>
          <div><strong>${dv.conflict_present ? "yes" : "no"}</strong><br><span style="color:var(--text-3);">conflict present</span></div>
        </div>
      `;
    }
    if (dimensionGridHost) {
      renderDimensionGrid(dimensionGridHost, d);
    }
  }

  /** Read a gold value for flat fields. */
  function goldGet(fieldId) {
    return d.gold_resulting_state?.[fieldId];
  }
  /** Write a gold value for flat fields. */
  function goldSet(fieldId, val) {
    if (!d.gold_resulting_state) d.gold_resulting_state = {};
    d.gold_resulting_state[fieldId] = val;
  }
  /** Read a gold value for repeat-group instance fields. */
  function goldRepeatGet(groupId, idx, fieldId) {
    const arr = d.gold_resulting_state?.[groupId];
    if (!Array.isArray(arr) || idx >= arr.length) return undefined;
    return arr[idx]?.[fieldId];
  }
  /** Write a gold value for repeat-group instance fields. */
  function goldRepeatSet(groupId, idx, fieldId, val) {
    if (!d.gold_resulting_state) d.gold_resulting_state = {};
    if (!Array.isArray(d.gold_resulting_state[groupId])) d.gold_resulting_state[groupId] = [];
    while (d.gold_resulting_state[groupId].length <= idx) d.gold_resulting_state[groupId].push({});
    d.gold_resulting_state[groupId][idx][fieldId] = val;
  }
  /** Read a prior value for repeat-group instance fields. */
  function priorRepeatGet(groupId, idx, fieldId) {
    const arr = priorAnswers[groupId];
    if (!Array.isArray(arr) || idx >= arr.length) return undefined;
    return arr[idx]?.[fieldId];
  }

  // -- Gold annotations (value source per field) --
  if (!d.gold_annotations) d.gold_annotations = {};

  /** Read annotation for a flat field. */
  function annotationGet(fieldId) {
    return d.gold_annotations?.[fieldId]?.value_source || null;
  }
  /** Write annotation for a flat field. */
  function annotationSet(fieldId, source) {
    if (!d.gold_annotations) d.gold_annotations = {};
    if (!d.gold_annotations[fieldId]) d.gold_annotations[fieldId] = {};
    d.gold_annotations[fieldId].value_source = source;
  }
  /** Read full annotation object for a flat field. */
  function annotationObjGet(fieldId) {
    if (!d.gold_annotations) d.gold_annotations = {};
    return d.gold_annotations[fieldId] || null;
  }
  /** Write full annotation (source + evidence) for a flat field. */
  function annotationObjSet(fieldId, source, evidence) {
    if (!d.gold_annotations) d.gold_annotations = {};
    if (!d.gold_annotations[fieldId]) d.gold_annotations[fieldId] = {};
    d.gold_annotations[fieldId].value_source = source;
    d.gold_annotations[fieldId].evidence = evidence;
  }
  /** Read annotation for a repeat-group instance field. */
  function annotationRepeatGet(groupId, idx, fieldId) {
    const arr = d.gold_annotations?.[groupId];
    if (!Array.isArray(arr) || idx >= arr.length) return null;
    return arr[idx]?.[fieldId]?.value_source || null;
  }
  /** Write annotation for a repeat-group instance field. */
  function annotationRepeatSet(groupId, idx, fieldId, source) {
    if (!d.gold_annotations) d.gold_annotations = {};
    if (!Array.isArray(d.gold_annotations[groupId])) d.gold_annotations[groupId] = [];
    while (d.gold_annotations[groupId].length <= idx) d.gold_annotations[groupId].push({});
    if (!d.gold_annotations[groupId][idx][fieldId]) d.gold_annotations[groupId][idx][fieldId] = {};
    d.gold_annotations[groupId][idx][fieldId].value_source = source;
  }
  /** Read full annotation object for a repeat-group instance field. */
  function annotationRepeatObjGet(groupId, idx, fieldId) {
    if (!d.gold_annotations) d.gold_annotations = {};
    const arr = d.gold_annotations[groupId];
    if (!Array.isArray(arr) || idx >= arr.length) return null;
    return arr[idx]?.[fieldId] || null;
  }
  /** Write full annotation (source + evidence) for a repeat-group instance field. */
  function annotationRepeatObjSet(groupId, idx, fieldId, source, evidence) {
    if (!d.gold_annotations) d.gold_annotations = {};
    if (!Array.isArray(d.gold_annotations[groupId])) d.gold_annotations[groupId] = [];
    while (d.gold_annotations[groupId].length <= idx) d.gold_annotations[groupId].push({});
    if (!d.gold_annotations[groupId][idx][fieldId]) d.gold_annotations[groupId][idx][fieldId] = {};
    d.gold_annotations[groupId][idx][fieldId].value_source = source;
    d.gold_annotations[groupId][idx][fieldId].evidence = evidence;
  }

  function fieldPathFromLocator(locator, fallbackFieldId) {
    if (locator?.repeatGroupId != null && locator?.repeatIndex != null) {
      return `${locator.repeatGroupId}[${locator.repeatIndex}].${locator.fieldId || fallbackFieldId}`;
    }
    return locator?.fieldId || fallbackFieldId;
  }

  function semanticIusGet(fieldPath) {
    if (!Array.isArray(d.semantic_ius)) d.semantic_ius = [];
    return d.semantic_ius.filter(iu => iu && iu.field_path === fieldPath);
  }

  function semanticIusSet(fieldPath, rows) {
    if (!Array.isArray(d.semantic_ius)) d.semantic_ius = [];
    const nextRows = (rows || [])
      .filter(iu => iu && typeof iu === "object")
      .map(iu => ({ ...iu, field_path: fieldPath }));
    d.semantic_ius = [
      ...d.semantic_ius.filter(iu => !iu || iu.field_path !== fieldPath),
      ...nextRows,
    ];
  }

  function expectedEntryForPath(fieldPath) {
    ensureExpectedOutcome(d);
    const repeatMatch = fieldPath.match(/^([^\[]+)\[(\d+)\]\.([^.\]]+)$/);
    if (!repeatMatch) {
      if (!d.expected_outcome.fields[fieldPath]) {
        d.expected_outcome.fields[fieldPath] = { expected: "", strategy: "exact", alternatives: [], evidence: "", evidence_source: "current_utterance", extraction_difficulty: "direct" };
      }
      return d.expected_outcome.fields[fieldPath];
    }

    const [, groupId, idxStr, childId] = repeatMatch;
    const idx = Number(idxStr);
    if (!d.expected_outcome.repeat_groups) d.expected_outcome.repeat_groups = {};
    let group = d.expected_outcome.repeat_groups[groupId];
    if (Array.isArray(group)) {
      while (group.length <= idx) group.push({});
      if (!group[idx][childId]) {
        group[idx][childId] = { expected: "", strategy: "exact", alternatives: [], evidence: "", evidence_source: "current_utterance", extraction_difficulty: "direct" };
      }
      return group[idx][childId];
    }
    if (!group || typeof group !== "object") {
      group = { alignment_keys: [], instances: [] };
      d.expected_outcome.repeat_groups[groupId] = group;
    }
    if (!Array.isArray(group.instances)) group.instances = [];
    let inst = group.instances.find((row, fallbackIdx) => {
      if (!row || typeof row !== "object") return false;
      return Number(row.ground_truth_index ?? fallbackIdx) === idx;
    });
    if (!inst) {
      inst = { ground_truth_index: idx, fields: {} };
      group.instances.push(inst);
    }
    if (!inst.fields || typeof inst.fields !== "object") inst.fields = {};
    if (!inst.fields[childId]) {
      inst.fields[childId] = { expected: "", strategy: "exact", alternatives: [], evidence: "", evidence_source: "current_utterance", extraction_difficulty: "direct" };
    }
    return inst.fields[childId];
  }

  function setSemanticIuScoring(fieldPath, enabled) {
    const entry = expectedEntryForPath(fieldPath);
    if (!entry) return;
    if (enabled) {
      entry.strategy = "semantic_iu";
      entry.semantic_ius_status = "ready";
    } else if (entry.strategy === "semantic_iu") {
      entry.strategy = "semantic";
      delete entry.semantic_ius_status;
    }
  }

  function canAttachSemanticIus(meta, q, schemaIus, existingIus) {
    if ((schemaIus || []).length || (existingIus || []).length) return true;
    const type = String(meta.type || q.type || q.answer_type || "").toLowerCase();
    return ["text", "string", "multiline_text"].includes(type);
  }

  // -- Per-field delta type accessors --
  /** Read delta_type for a flat field. */
  function deltaTypeGet(fieldId) {
    return d.gold_annotations?.[fieldId]?.delta_type || null;
  }
  /** Write delta_type for a flat field. */
  function deltaTypeSet(fieldId, dt) {
    if (!d.gold_annotations) d.gold_annotations = {};
    if (!d.gold_annotations[fieldId]) d.gold_annotations[fieldId] = {};
    d.gold_annotations[fieldId].delta_type = dt;
  }
  /** Read delta_type for a repeat-group instance field. */
  function deltaTypeRepeatGet(groupId, idx, fieldId) {
    const arr = d.gold_annotations?.[groupId];
    if (!Array.isArray(arr) || idx >= arr.length) return null;
    return arr[idx]?.[fieldId]?.delta_type || null;
  }
  /** Write delta_type for a repeat-group instance field. */
  function deltaTypeRepeatSet(groupId, idx, fieldId, dt) {
    if (!d.gold_annotations) d.gold_annotations = {};
    if (!Array.isArray(d.gold_annotations[groupId])) d.gold_annotations[groupId] = [];
    while (d.gold_annotations[groupId].length <= idx) d.gold_annotations[groupId].push({});
    if (!d.gold_annotations[groupId][idx][fieldId]) d.gold_annotations[groupId][idx][fieldId] = {};
    d.gold_annotations[groupId][idx][fieldId].delta_type = dt;
  }

  // Collect which fields control gates or repeat groups
  const structuralFields = new Set();
  function collectStructuralFields(questions) {
    for (const q of questions) {
      const st = q.structure_type || "regular";
      if (st === "gate" && q.gate?.gate_on) structuralFields.add(q.gate.gate_on);
      if (st === "branch" && q.branch?.branch_on) structuralFields.add(q.branch.branch_on);
      if (st === "repeat_group" && q.repeat?.from_slot) structuralFields.add(q.repeat.from_slot);
      if (st === "group" || st === "gate" || st === "repeat_group") {
        collectStructuralFields(q.fields || q.questions || []);
      } else if (st === "branch") {
        const branch = q.branch || {};
        for (const route of branch.routes || []) {
          collectStructuralFields(route.children || []);
        }
        if (branch.default_children?.length) collectStructuralFields(branch.default_children);
      }
    }
  }
  collectStructuralFields(qTree);

  function captureGoldFieldFocus() {
    const active = document.activeElement;
    if (!(active instanceof HTMLElement)) return null;
    const row = active.closest(".gold-form-field");
    if (!(row instanceof HTMLElement)) return null;
    const focusables = Array.from(row.querySelectorAll("input, select, textarea, button"));
    const focusIndex = focusables.indexOf(active);
    return {
      fieldId: row.dataset.fieldId || "",
      repeatGroupId: row.dataset.repeatGroupId || "",
      repeatIndex: row.dataset.repeatIndex || "",
      focusIndex: focusIndex >= 0 ? focusIndex : 0,
      selectionStart: typeof active.selectionStart === "number" ? active.selectionStart : null,
      selectionEnd: typeof active.selectionEnd === "number" ? active.selectionEnd : null,
    };
  }

  function restoreGoldFieldFocus(snapshot) {
    if (!snapshot) return;
    const selector = snapshot.repeatGroupId
      ? `.gold-form-field[data-field-id="${snapshot.fieldId}"][data-repeat-group-id="${snapshot.repeatGroupId}"][data-repeat-index="${snapshot.repeatIndex}"]`
      : `.gold-form-field[data-field-id="${snapshot.fieldId}"]:not([data-repeat-group-id])`;
    const row = formWrap.querySelector(selector);
    if (!(row instanceof HTMLElement)) return;
    const focusables = Array.from(row.querySelectorAll("input, select, textarea, button"));
    const target = focusables[Math.min(snapshot.focusIndex, Math.max(focusables.length - 1, 0))];
    if (!(target instanceof HTMLElement)) return;
    target.focus({ preventScroll: true });
    if (typeof target.setSelectionRange === "function" && snapshot.selectionStart != null && snapshot.selectionEnd != null) {
      target.setSelectionRange(snapshot.selectionStart, snapshot.selectionEnd);
    }
  }

  function preserveEditorScroll(fn) {
    const scroller = document.getElementById("editor-body");
    const focusSnapshot = captureGoldFieldFocus();
    if (!scroller) {
      const result = fn();
      restoreGoldFieldFocus(focusSnapshot);
      requestAnimationFrame(() => restoreGoldFieldFocus(focusSnapshot));
      return result;
    }
    const { scrollTop, scrollLeft } = scroller;
    const result = fn();
    scroller.scrollTop = scrollTop;
    scroller.scrollLeft = scrollLeft;
    requestAnimationFrame(() => {
      restoreGoldFieldFocus(focusSnapshot);
      scroller.scrollTop = scrollTop;
      scroller.scrollLeft = scrollLeft;
    });
    restoreGoldFieldFocus(focusSnapshot);
    return result;
  }

  // Helper: onFieldChange for flat fields
  function onFlatFieldChange(fieldId, newVal) {
    goldSet(fieldId, newVal);
    ensureExpectedOutcome(d);
    if (!d.expected_outcome.fields[fieldId]) {
      d.expected_outcome.fields[fieldId] = { expected: "", strategy: "exact", alternatives: [], evidence: "", evidence_source: "current_utterance", extraction_difficulty: "direct" };
    }
    d.expected_outcome.fields[fieldId].expected = newVal;
    markDirty();
    try { updateChecklist(); } catch (_) {}
    try { updateChangeSummary(); } catch (_) {}
    // Only rebuild the form if this field controls a gate or repeat group
    if (structuralFields.has(fieldId)) preserveEditorScroll(() => rebuildForm());
  }

  // Helper: onFieldChange for repeat-group instance fields
  function onRepeatFieldChange(groupId, idx, fieldId, newVal) {
    goldRepeatSet(groupId, idx, fieldId, newVal);
    markDirty();
    try { updateChecklist(); } catch (_) {}
    try { updateChangeSummary(); } catch (_) {}
  }

  function valueMatchesCondition(currentVal, expectedVal) {
    if (expectedVal === true && (currentVal === true || currentVal === "true")) return true;
    if (expectedVal === false && (currentVal === false || currentVal === "false")) return true;
    return String(expectedVal) === String(currentVal);
  }

  /** Check if a gate is open based on a field-value reader. */
  function isGateOpenForReader(gate, readValue) {
    const currentVal = readValue(gate.gate_on);
    if (currentVal === undefined || currentVal === null) return false;
    for (const wv of gate.when_values) {
      if (valueMatchesCondition(currentVal, wv)) return true;
    }
    return false;
  }

  /** Check if a gate is open based on current gold_resulting_state. */
  function isGateOpen(gate) {
    return isGateOpenForReader(gate, goldGet);
  }

  function collectDescendantRegularIds(questions) {
    const ids = [];
    for (const child of questions || []) {
      const st = child.structure_type || "regular";
      if (st === "regular") {
        ids.push(child.id);
      } else if (st === "group" || st === "gate" || st === "repeat_group") {
        ids.push(...collectDescendantRegularIds(child.fields || child.questions || []));
      } else if (st === "branch") {
        const branch = child.branch || {};
        for (const route of branch.routes || []) ids.push(...collectDescendantRegularIds(route.children || []));
        if (branch.default_children?.length) ids.push(...collectDescendantRegularIds(branch.default_children));
      }
    }
    return ids;
  }

  function collectGateChildRelationships(questions, repeatGroupId = null) {
    const relationships = [];
    for (const q of questions || []) {
      const st = q.structure_type || "regular";
      if (st === "gate" && q.gate?.gate_on) {
        for (const childId of collectDescendantRegularIds(q.fields || q.questions || [])) {
          relationships.push({ childId, repeatGroupId, gate: q.gate });
        }
        relationships.push(...collectGateChildRelationships(q.fields || q.questions || [], repeatGroupId));
      } else if (st === "repeat_group") {
        relationships.push(...collectGateChildRelationships(q.fields || q.questions || [], q.id));
      } else if (st === "group") {
        relationships.push(...collectGateChildRelationships(q.fields || q.questions || [], repeatGroupId));
      } else if (st === "branch") {
        const branch = q.branch || {};
        for (const route of branch.routes || []) relationships.push(...collectGateChildRelationships(route.children || [], repeatGroupId));
        if (branch.default_children?.length) relationships.push(...collectGateChildRelationships(branch.default_children, repeatGroupId));
      }
    }
    return relationships;
  }

  const gateChildRelationships = collectGateChildRelationships(qTree);

  function effectivePriorForField(fieldId, priorVal, { repeatGroupId = null, repeatIndex = null } = {}) {
    const relationships = gateChildRelationships.filter(rel => rel.childId === fieldId && rel.repeatGroupId === repeatGroupId);
    for (const rel of relationships) {
      const priorOpen = isGateOpenForReader(rel.gate, controllerId =>
        repeatGroupId != null ? priorRepeatGet(repeatGroupId, repeatIndex, controllerId) : priorAnswers[controllerId]
      );
      const currentOpen = isGateOpenForReader(rel.gate, controllerId =>
        repeatGroupId != null ? goldRepeatGet(repeatGroupId, repeatIndex, controllerId) : goldGet(controllerId)
      );
      if (!priorOpen && currentOpen) return { value: null, adjustedByGateActivation: true };
    }
    return { value: priorVal, adjustedByGateActivation: false };
  }

  function resolveBranchRoute(branch, readValue) {
    if (!branch?.branch_on) return { route: null, usedDefault: false, branchValue: undefined };
    const branchValue = readValue(branch.branch_on);
    for (const route of branch.routes || []) {
      if (valueMatchesCondition(branchValue, route.when_value)) {
        return { route, usedDefault: false, branchValue };
      }
    }
    const defaultChildren = branch.default_children || [];
    if (defaultChildren.length) {
      return { route: { when_value: null, children: defaultChildren }, usedDefault: true, branchValue };
    }
    return { route: null, usedDefault: false, branchValue };
  }

  /** Get repeat count from a from_slot field. Pure reader — no side effects. */
  function getRepeatCount(repeatConfig, groupId) {
    if (repeatConfig.mode === "fixed") return repeatConfig.count || 0;
    const val = goldGet(repeatConfig.from_slot);
    const n = parseInt(val, 10);
    return isNaN(n) || n < 0 ? 0 : Math.min(n, 20); // cap at 20
  }

  /** Render a single regular field row. */
  function renderField(q, priorVal, currentVal, onChangeCallback, { getSource, setSource, getAnnotation, setAnnotation, getDeltaType, setDeltaType, locator, diffContext } = {}) {
    const meta = fieldMetaMap[q.id] || {
      id: q.id,
      label: q.question_text || q.label || q.id,
      type: q.type || "text",
      options: q.options || [],
      other_specify: !!q.other_specify,
      columns: q.columns || [],
      information_units: q.information_units || [],
    };
    const fieldPath = fieldPathFromLocator(locator, q.id);
    const schemaIus = Array.isArray(meta.information_units)
      ? meta.information_units
      : (Array.isArray(q.information_units) ? q.information_units : []);
    const itemSemanticIus = semanticIusGet(fieldPath);
    const supportsSemanticIus = canAttachSemanticIus(meta, q, schemaIus, itemSemanticIus);
    const kind = classifyChange(priorVal, currentVal);
    let deltaType = getDeltaType ? getDeltaType() : null;

    // Auto-suggest delta type. If an older render marked an inactive gated
    // child as keep, refresh it once the effective prior becomes empty.
    if (getDeltaType && setDeltaType) {
      const suggested = suggestDeltaType(kind);
      if (!deltaType || (diffContext?.adjustedByGateActivation && deltaType === "keep" && suggested !== "keep")) {
        setDeltaType(suggested);
        deltaType = suggested;
      }
    }

    // Respect delta_type: if "keep", don't show diff highlighting (value is intentionally preserved)
    const isChanged = kind !== null && deltaType !== "keep";
    let liveKind = kind;
    let liveCurrentVal = currentVal;

    const rowDataset = locator ? {
      fieldId: locator.fieldId || "",
      ...(locator.repeatGroupId ? { repeatGroupId: locator.repeatGroupId } : {}),
      ...(locator.repeatIndex != null ? { repeatIndex: String(locator.repeatIndex) } : {}),
    } : undefined;
    const row = el("div", {
      class: `gold-form-field${isChanged ? " gold-changed gold-changed-" + kind.toLowerCase() : ""}`,
      ...(rowDataset ? { dataset: rowDataset } : {}),
    });

    const labelCol = el("div", { class: "gold-form-label-col" });
    const fieldLabel = meta.label && meta.label !== q.id
      ? el("label", { class: "gold-form-label", title: q.id }, meta.label)
      : el("label", { class: "gold-form-label" }, q.id);
    labelCol.appendChild(fieldLabel);

    const badges = el("div", { class: "gold-form-badges" });
    badges.appendChild(el("span", { class: "gold-type-badge" }, meta.type || "text"));
    if (meta.claim_anchor) {
      const anchorMeta = getClaimAnchorMeta(meta.claim_anchor);
      const anchorTitleBits = [];
      if (anchorMeta?.desc) anchorTitleBits.push(anchorMeta.desc);
      if (meta.entity_role) anchorTitleBits.push(`Resolved role: ${meta.entity_role}`);
      badges.appendChild(el("span", {
        class: "gold-anchor-badge",
        title: anchorTitleBits.join(" "),
      }, formatClaimAnchorLabel(meta.claim_anchor, meta.entity_role, { short: true })));
    }
    labelCol.appendChild(badges);

    // Question guidance: gold_standard + information_units from questionnaire
    if (meta.gold_standard || meta.information_units?.length) {
      const guidanceDetails = el("details", { class: "field-question-guidance" });
      guidanceDetails.appendChild(el("summary", {}, "question guidance"));
      const guidanceBody = el("div", { class: "field-question-guidance-body" });
      if (meta.gold_standard) {
        guidanceBody.appendChild(el("div", { class: "field-gold-standard" }, meta.gold_standard));
      }
      if (meta.information_units?.length) {
        const iuList = el("div", { class: "field-iu-list" });
        iuList.appendChild(el("div", { class: "field-iu-header" }, "Information units:"));
        for (const iu of meta.information_units) {
          iuList.appendChild(el("div", { class: "field-iu-item" },
            el("span", { class: "field-iu-name" }, iu.name || iu.id),
            el("span", { class: "field-iu-desc" }, iu.description || ""),
          ));
        }
        guidanceBody.appendChild(iuList);
      }
      guidanceDetails.appendChild(guidanceBody);
      labelCol.appendChild(guidanceDetails);
    }

    const inputCol = el("div", { class: "gold-form-input-col" });

    // Auto-filled fields: render as read-only indicator, no editable input
    if (q.auto_fill) {
      const autoVal = locator?.repeatIndex != null
        ? (locator.repeatIndex + (q.auto_fill.offset || 1))
        : currentVal;
      const badge = el("span", { class: "gold-auto-fill-badge" }, `${autoVal}`);
      badge.title = "Auto-filled from row position. Not scored.";
      inputCol.appendChild(badge);
      row.appendChild(labelCol);
      row.appendChild(inputCol);
      row.classList.add("gold-auto-fill-row");
      return row;
    }

    function setDeltaPillsSelection(wrap, value) {
      wrap.querySelectorAll(".delta-pill").forEach(p => {
        const pDt = FIELD_DELTA_TYPES.find(d => d.value === p.dataset.delta);
        const isActive = p.dataset.delta === value;
        p.classList.toggle("active", isActive);
        p.style.borderColor = isActive && pDt ? pDt.color : "var(--border)";
        p.style.background = isActive && pDt ? pDt.color + "18" : "transparent";
        p.style.color = isActive && pDt ? pDt.color : "var(--text-3)";
      });
    }

    function updateRowDiffUi(activeDeltaType) {
      const shouldShow = liveKind !== null && activeDeltaType !== "keep";
      row.className = "gold-form-field" + (shouldShow ? " gold-changed gold-changed-" + liveKind.toLowerCase() : "");
      const existingDiff = inputCol.querySelector(".gold-inline-diff");
      if (existingDiff) existingDiff.remove();
      if (shouldShow) inputCol.appendChild(buildInlineDiff(liveKind, priorVal, liveCurrentVal));
    }

    // Delta type pill selector (per-field) — when changed, update diff highlighting
    let dtWrap = null;
    if (getDeltaType && setDeltaType) {
      dtWrap = buildDeltaTypePills(getDeltaType(), (dt) => {
        setDeltaType(dt);
        markDirty();
        updateChangeSummary();
        preserveEditorScroll(() => updateRowDiffUi(dt));
      });
      labelCol.appendChild(dtWrap);
    }
    const input = makeGoldFormInput(meta, currentVal, (newVal) => {
      onChangeCallback(newVal);
      liveCurrentVal = newVal;
      liveKind = classifyChange(priorVal, newVal);

      // Auto-update delta type BEFORE applying diff highlighting so the
      // effective delta type is used (fixes stale "keep" suppressing diffs)
      let effectiveDt = getDeltaType ? getDeltaType() : null;
      if (getDeltaType && setDeltaType) {
        const suggested = suggestDeltaType(liveKind);
        // Only auto-update if current is the old auto-suggestion (don't override manual picks)
        const autoTypes = ["keep", "add", "retract", "correct"];
        if (!effectiveDt || autoTypes.includes(effectiveDt)) {
          setDeltaType(suggested);
          effectiveDt = suggested;
          if (dtWrap) setDeltaPillsSelection(dtWrap, suggested);
        }
      }

      preserveEditorScroll(() => {
        updateRowDiffUi(effectiveDt);
        // Show/hide source selector based on whether field has a value
        updateSourceVisibility(newVal);
      });
    });
    inputCol.appendChild(input);
    if (isChanged) inputCol.appendChild(buildInlineDiff(kind, priorVal, currentVal));

    // Value source selector (only shown when field has a non-empty value)
    let sourceWrap = null;
    function updateSourceVisibility(val) {
      const hasValue = val != null && val !== "";
      if (sourceWrap) sourceWrap.style.display = hasValue ? "" : "none";
    }
    if (getSource && setSource) {
      sourceWrap = buildValueSourceSelector(getSource(), (src) => {
        setSource(src);
        markDirty();
      }, {
        fieldId: q.id,
        fieldPath,
        fieldLabel: (meta.label && meta.label !== q.id) ? meta.label : q.id,
        claimAnchor: meta.claim_anchor || null,
        entityRole: meta.entity_role || null,
        getAnnotation,
        setAnnotation,
        semanticIu: supportsSemanticIus ? {
          itemId: d.item_id,
          fieldId: q.id,
          fieldPath,
          schemaIus,
          existingIus: itemSemanticIus,
          allowCustom: true,
        } : null,
        setSemanticIus: (rows) => {
          const nextRows = Array.isArray(rows) ? rows : [];
          semanticIusSet(fieldPath, nextRows);
          setSemanticIuScoring(fieldPath, nextRows.length > 0);
        },
        getSemanticIusCount: () => semanticIusGet(fieldPath).length,
        getSemanticIus: () => semanticIusGet(fieldPath),
      });
      updateSourceVisibility(currentVal);
      inputCol.appendChild(sourceWrap);
    }

    row.append(labelCol, inputCol);
    return row;
  }

  /** Build a value-source pill selector for a field. */
  function buildValueSourceSelector(currentSource, onSelect, evidenceOpts) {
    const wrap = el("div", { class: "gold-source-selector" });

    // Evidence indicator (shows when evidence exists)
    const evIndicator = el("button", {
      type: "button",
      class: "ev-indicator",
      title: "View/edit evidence",
    });
    function openFieldEvidencePopup(sourceForPopup) {
      if (!evidenceOpts) return;
      const ann = evidenceOpts.getAnnotation() || {};
      openEvidencePopup({
        fieldId: evidenceOpts.fieldPath || evidenceOpts.fieldId,
        fieldLabel: evidenceOpts.fieldLabel,
        claimAnchor: evidenceOpts.claimAnchor,
        entityRole: evidenceOpts.entityRole,
        currentSource: sourceForPopup,
        annotation: ann,
        semanticIu: evidenceOpts.semanticIu ? {
          ...evidenceOpts.semanticIu,
          existingIus: evidenceOpts.getSemanticIus ? evidenceOpts.getSemanticIus() : evidenceOpts.semanticIu.existingIus,
        } : null,
        onDone: (finalSource, evidence, semanticIus) => {
          currentSource = finalSource;
          wrap.querySelectorAll(".gold-source-pill").forEach(p => {
            const pv = p.dataset.source;
            const active = pv === finalSource;
            p.classList.toggle("active", active);
            const src = VALUE_SOURCES.find(s => s.value === pv);
            p.style.setProperty("--pill-color", active && src ? src.color : "");
          });
          onSelect(finalSource);
          evidenceOpts.setAnnotation(finalSource, evidence);
          if (Array.isArray(semanticIus) && evidenceOpts.setSemanticIus) {
            evidenceOpts.setSemanticIus(semanticIus);
          }
          markDirty();
          updateEvIndicator();
        },
      });
    }
    function updateEvIndicator() {
      if (!evidenceOpts) { evIndicator.style.display = "none"; return; }
      const ann = evidenceOpts.getAnnotation();
      const hasEvidence = ann?.evidence?.spans?.length > 0 || (ann?.evidence?.note && ann.evidence.note.trim());
      const iuCount = evidenceOpts.getSemanticIusCount ? evidenceOpts.getSemanticIusCount() : (evidenceOpts.semanticIu?.existingIus?.length || 0);
      const hasSemanticIus = Boolean(evidenceOpts.semanticIu) || iuCount > 0;
      evIndicator.style.display = (currentSource || hasSemanticIus) ? "" : "none";
      const count = ann?.evidence?.spans?.length || 0;
      if (hasEvidence || iuCount > 0) {
        evIndicator.classList.add("has-evidence");
        evIndicator.innerHTML = `<span class="ev-indicator-icon">\ud83d\udccc</span><span class="ev-indicator-count">${count}</span>${iuCount ? `<span class="ev-indicator-iu">${iuCount} IU</span>` : ""}`;
        evIndicator.title = `${count} evidence span${count !== 1 ? "s" : ""}${iuCount ? `, ${iuCount} semantic IU${iuCount !== 1 ? "s" : ""}` : ""} — click to edit`;
      } else {
        evIndicator.classList.remove("has-evidence");
        evIndicator.innerHTML = `<span class="ev-indicator-icon">\ud83d\udccc</span>`;
        evIndicator.title = hasSemanticIus ? "Add field evidence or semantic IU gold" : "Add evidence for this annotation";
      }
    }

    for (const vs of VALUE_SOURCES) {
      const pill = el("button", {
        type: "button",
        class: `gold-source-pill${currentSource === vs.value ? " active" : ""}`,
        title: vs.desc,
        "data-source": vs.value,
        style: currentSource === vs.value ? `--pill-color: ${vs.color};` : "",
      }, vs.label);
      pill.addEventListener("click", () => {
        const isNewSelection = currentSource !== vs.value;
        // Toggle: clicking active pill deselects
        const newVal = currentSource === vs.value ? null : vs.value;
        // Update all pills in this selector
        wrap.querySelectorAll(".gold-source-pill").forEach(p => {
          const pv = p.dataset.source;
          const active = pv === newVal;
          p.classList.toggle("active", active);
          const src = VALUE_SOURCES.find(s => s.value === pv);
          p.style.setProperty("--pill-color", active && src ? src.color : "");
        });
        currentSource = newVal;
        onSelect(newVal);
        updateEvIndicator();

        // Open evidence popup when selecting a source (new selection)
        if (isNewSelection && newVal && evidenceOpts) {
          openFieldEvidencePopup(newVal);
        }
      });
      wrap.appendChild(pill);
    }

    // Evidence indicator click opens popup too
    evIndicator.addEventListener("click", () => {
      if (!evidenceOpts) return;
      openFieldEvidencePopup(currentSource || "stated");
    });

    wrap.appendChild(evIndicator);
    updateEvIndicator();
    return wrap;
  }

  /** Build a delta type pill selector for a field (compact, below label). */
  function buildDeltaTypePills(currentDt, onSelect) {
    const wrap = el("div", { class: "delta-type-pills", style: "display:flex;gap:3px;margin-top:4px;flex-wrap:wrap;" });
    for (const dt of FIELD_DELTA_TYPES) {
      const active = currentDt === dt.value;
      const pill = el("button", {
        type: "button",
        class: `delta-pill${active ? " active" : ""}`,
        title: dt.desc,
        "data-delta": dt.value,
        style: `font-size:10px;font-weight:600;padding:1px 6px;border-radius:8px;border:1.5px solid ${active ? dt.color : "var(--border)"};background:${active ? dt.color + "18" : "transparent"};color:${active ? dt.color : "var(--text-3)"};cursor:pointer;transition:all .12s;`,
      }, dt.label);
      pill.addEventListener("click", () => {
        wrap.querySelectorAll(".delta-pill").forEach(p => {
          const pDt = FIELD_DELTA_TYPES.find(d => d.value === p.dataset.delta);
          const isActive = p.dataset.delta === dt.value;
          p.classList.toggle("active", isActive);
          p.style.borderColor = isActive && pDt ? pDt.color : "var(--border)";
          p.style.background = isActive && pDt ? pDt.color + "18" : "transparent";
          p.style.color = isActive && pDt ? pDt.color : "var(--text-3)";
        });
        onSelect(dt.value);
      });
      wrap.appendChild(pill);
    }
    return wrap;
  }

  /** Walk the questionnaire tree and render into formWrap. */
  function rebuildForm() {
    formWrap.innerHTML = "";

    // One-time initial sync: if a from_slot count field was never set but the
    // repeat group array already has data (e.g. loaded from disk), infer the
    // count. This only fires when the field is truly absent from gold state.
    function syncFromSlotCounts(questions) {
      for (const q of questions) {
        const st = q.structure_type || "regular";
        if (st === "repeat_group" && q.repeat?.mode === "from_slot") {
          const slot = q.repeat.from_slot;
          const slotVal = d.gold_resulting_state?.[slot];
          if (slotVal === undefined && !(slot in (d.gold_resulting_state || {}))) {
            const arr = d.gold_resulting_state?.[q.id];
            if (Array.isArray(arr) && arr.length > 0) {
              goldSet(slot, arr.length);
            }
          }
        }
        if (st === "group" || st === "gate" || st === "repeat_group") {
          syncFromSlotCounts(q.fields || q.questions || []);
        } else if (st === "branch") {
          const branch = q.branch || {};
          for (const route of branch.routes || []) {
            syncFromSlotCounts(route.children || []);
          }
          if (branch.default_children?.length) syncFromSlotCounts(branch.default_children);
        }
      }
    }
    syncFromSlotCounts(qTree);

    function walkRepeatQuestions(questions, instanceEl, groupId, idx, parentLabel) {
      const groupArr = Array.isArray(d.gold_resulting_state[groupId]) ? d.gold_resulting_state[groupId] : [];
      for (const q of questions) {
        const st = q.structure_type || "regular";

        if (st === "regular") {
          if (groupArr[idx] && !(q.id in groupArr[idx])) {
            const priorInst = Array.isArray(priorAnswers[groupId]) ? priorAnswers[groupId][idx] : null;
            groupArr[idx][q.id] = priorInst?.[q.id] ?? null;
          }
          const rawPriorVal = priorRepeatGet(groupId, idx, q.id);
          const priorInfo = effectivePriorForField(q.id, rawPriorVal, { repeatGroupId: groupId, repeatIndex: idx });
          const currentVal = goldRepeatGet(groupId, idx, q.id);
          const capturedIdx = idx;
          instanceEl.appendChild(renderField(q, priorInfo.value, currentVal, (newVal) => onRepeatFieldChange(groupId, capturedIdx, q.id, newVal), {
            getSource: () => annotationRepeatGet(groupId, capturedIdx, q.id),
            setSource: (src) => annotationRepeatSet(groupId, capturedIdx, q.id, src),
            getAnnotation: () => annotationRepeatObjGet(groupId, capturedIdx, q.id),
            setAnnotation: (src, ev) => annotationRepeatObjSet(groupId, capturedIdx, q.id, src, ev),
            getDeltaType: () => deltaTypeRepeatGet(groupId, capturedIdx, q.id),
            setDeltaType: (dt) => deltaTypeRepeatSet(groupId, capturedIdx, q.id, dt),
            locator: { fieldId: q.id, repeatGroupId: groupId, repeatIndex: capturedIdx },
            diffContext: priorInfo,
          }));
          continue;
        }

        if (st === "group") {
          const label = q.label || q.id;
          instanceEl.appendChild(el("div", { class: "gold-form-group-header" }, parentLabel ? `${parentLabel} / ${label}` : label));
          walkRepeatQuestions(q.fields || q.questions || [], instanceEl, groupId, idx, label);
          continue;
        }

        if (st === "gate") {
          const gate = q.gate;
          if (!gate) continue;
          const open = isGateOpenForReader(gate, (fieldId) => goldRepeatGet(groupId, idx, fieldId));
          const gateEl = el("div", { class: `gold-gate-section${open ? "" : " gold-gate-closed"}` });
          const gateHeader = el("div", { class: "gold-gate-header" },
            el("span", { class: `gold-gate-indicator ${open ? "gold-gate-open" : ""}` }, open ? "\u25bc" : "\u25b6"),
            el("span", {}, q.label || q.id),
            el("span", { class: "gold-gate-status" }, open ? `(gate open \u2014 ${gate.gate_on})` : `(gate closed \u2014 ${gate.gate_on} \u2260 ${gate.when_values.join("|")})`),
          );
          gateEl.appendChild(gateHeader);
          instanceEl.appendChild(gateEl);
          if (open) {
            walkRepeatQuestions(q.fields || q.questions || [], instanceEl, groupId, idx, q.label || q.id);
          } else {
            instanceEl.appendChild(el("div", { class: "gold-gate-hint" },
              `Set "${gate.gate_on}" to ${gate.when_values.map(v => JSON.stringify(v)).join(" or ")} to reveal these fields.`));
          }
          continue;
        }

        if (st === "branch") {
          const branch = q.branch;
          if (!branch) continue;
          const { route, usedDefault, branchValue } = resolveBranchRoute(
            branch,
            (fieldId) => goldRepeatGet(groupId, idx, fieldId),
          );
          const branchEl = el("div", { class: `gold-gate-section${route ? "" : " gold-gate-closed"}` });
          const branchHeader = el("div", { class: "gold-gate-header" },
            el("span", { class: `gold-gate-indicator ${route ? "gold-gate-open" : ""}` }, route ? "\u25bc" : "\u25b6"),
            el("span", {}, q.label || q.id),
            el("span", { class: "gold-gate-status" }, route
              ? usedDefault
                ? `(default route \u2014 ${branch.branch_on} has no matching value)`
                : `(${branch.branch_on} = ${JSON.stringify(route.when_value)})`
              : `(no route selected \u2014 set ${branch.branch_on})`),
          );
          branchEl.appendChild(branchHeader);
          instanceEl.appendChild(branchEl);
          if (route) {
            walkRepeatQuestions(route.children || [], instanceEl, groupId, idx, q.label || q.id);
          } else {
            const observed = branchValue === undefined || branchValue === null ? "\u2014" : JSON.stringify(branchValue);
            instanceEl.appendChild(el("div", { class: "gold-gate-hint" },
              `Set "${branch.branch_on}" to one of ${(branch.routes || []).map(r => JSON.stringify(r.when_value)).join(", ")} to reveal these fields. Current value: ${observed}.`));
          }
        }
      }
    }

    function walkQuestions(questions, parentLabel) {
      for (const q of questions) {
        const st = q.structure_type || "regular";

        if (st === "regular") {
          // Ensure field in gold state
          if (!(q.id in (d.gold_resulting_state || {}))) {
            goldSet(q.id, priorAnswers[q.id] ?? null);
          }
          const priorInfo = effectivePriorForField(q.id, priorAnswers[q.id]);
          const currentVal = goldGet(q.id);
          formWrap.appendChild(renderField(q, priorInfo.value, currentVal, (newVal) => onFlatFieldChange(q.id, newVal), {
            getSource: () => annotationGet(q.id),
            setSource: (src) => annotationSet(q.id, src),
            getAnnotation: () => annotationObjGet(q.id),
            setAnnotation: (src, ev) => annotationObjSet(q.id, src, ev),
            getDeltaType: () => deltaTypeGet(q.id),
            setDeltaType: (dt) => deltaTypeSet(q.id, dt),
            locator: { fieldId: q.id },
            diffContext: priorInfo,
          }));

        } else if (st === "group") {
          const label = q.label || q.id;
          formWrap.appendChild(el("div", { class: "gold-form-group-header" }, parentLabel ? `${parentLabel} / ${label}` : label));
          walkQuestions(q.fields || q.questions || [], label);

        } else if (st === "gate") {
          const gate = q.gate;
          if (!gate) continue;
          const open = isGateOpen(gate);
          const gateEl = el("div", { class: `gold-gate-section${open ? "" : " gold-gate-closed"}` });
          const gateHeader = el("div", { class: "gold-gate-header" },
            el("span", { class: `gold-gate-indicator ${open ? "gold-gate-open" : ""}` }, open ? "\u25bc" : "\u25b6"),
            el("span", {}, q.label || q.id),
            el("span", { class: "gold-gate-status" }, open ? `(gate open \u2014 ${gate.gate_on})` : `(gate closed \u2014 ${gate.gate_on} \u2260 ${gate.when_values.join("|")})`),
          );
          gateEl.appendChild(gateHeader);
          formWrap.appendChild(gateEl);
          if (open) {
            walkQuestions(q.fields || q.questions || [], q.label || q.id);
          } else {
            const hint = el("div", { class: "gold-gate-hint" },
              `Set "${gate.gate_on}" to ${gate.when_values.map(v => JSON.stringify(v)).join(" or ")} to reveal these fields.`);
            formWrap.appendChild(hint);
            continue;
          }

        } else if (st === "branch") {
          const branch = q.branch;
          if (!branch) continue;
          const { route, usedDefault, branchValue } = resolveBranchRoute(branch, goldGet);
          const branchEl = el("div", { class: `gold-gate-section${route ? "" : " gold-gate-closed"}` });
          const branchHeader = el("div", { class: "gold-gate-header" },
            el("span", { class: `gold-gate-indicator ${route ? "gold-gate-open" : ""}` }, route ? "\u25bc" : "\u25b6"),
            el("span", {}, q.label || q.id),
            el("span", { class: "gold-gate-status" }, route
              ? usedDefault
                ? `(default route \u2014 ${branch.branch_on} has no matching value)`
                : `(${branch.branch_on} = ${JSON.stringify(route.when_value)})`
              : `(no route selected \u2014 set ${branch.branch_on})`),
          );
          formWrap.appendChild(branchEl);
          if (route) {
            walkQuestions(route.children || [], q.label || q.id);
          } else {
            const observed = branchValue === undefined || branchValue === null ? "\u2014" : JSON.stringify(branchValue);
            formWrap.appendChild(el("div", { class: "gold-gate-hint" },
              `Set "${branch.branch_on}" to one of ${(branch.routes || []).map(r => JSON.stringify(r.when_value)).join(", ")} to reveal these fields. Current value: ${observed}.`));
          }

        } else if (st === "repeat_group") {
          const repeat = q.repeat;
          if (!repeat) continue;
          const groupId = q.id;
          const count = getRepeatCount(repeat, groupId);
          const childFields = q.fields || q.questions || [];

          // Ensure gold state has the right number of instances
          if (!Array.isArray(d.gold_resulting_state[groupId])) {
            const priorArr = Array.isArray(priorAnswers[groupId]) ? priorAnswers[groupId] : [];
            d.gold_resulting_state[groupId] = priorArr.map(inst => ({...inst}));
          }
          const arr = d.gold_resulting_state[groupId];
          // Expand to match count
          while (arr.length < count) {
            const priorInst = Array.isArray(priorAnswers[groupId]) ? priorAnswers[groupId][arr.length] : null;
            arr.push(priorInst ? {...priorInst} : {});
          }
          // Shrink to match count — remove trailing instances
          if (arr.length > count) {
            arr.length = count;
          }

          const repeatHeader = el("div", { class: "gold-repeat-header" },
            el("span", { class: "gold-repeat-icon" }, "\u21bb"),
            el("span", {}, q.label || groupId),
            el("span", { class: "gold-repeat-count" }, count > 0 ? `${count} instance${count !== 1 ? "s" : ""}` : "0 instances \u2014 set the count field above"),
          );
          formWrap.appendChild(repeatHeader);

          if (count === 0) {
            formWrap.appendChild(el("div", { class: "gold-repeat-empty" },
              `Set "${repeat.from_slot}" to a number to add instances.`));
          }

          for (let i = 0; i < count; i++) {
            const instanceLabel = (repeat.item_label || `${groupId} {{index}}`).replace("{{index}}", String(i + 1));
            const instanceEl = el("div", { class: "gold-repeat-instance" });
            instanceEl.appendChild(el("div", { class: "gold-repeat-instance-header" }, instanceLabel));
            walkRepeatQuestions(childFields, instanceEl, groupId, i, q.label || groupId);
            formWrap.appendChild(instanceEl);
          }
        }
      }
    }

    walkQuestions(qTree, "");
  }

  rebuildForm();

  // Append change summary
  container.appendChild(changeSummary);
  container.appendChild(derivedVariablesPanel);
  updateChangeSummary();

  // Sync gold_resulting_state -> expected_outcome.fields for backward compat
  function syncExpectedOutcome() {
    ensureExpectedOutcome(d);
    const eo = d.expected_outcome.fields;
    const gold = d.gold_resulting_state || {};
    for (const [fid, val] of Object.entries(gold)) {
      // Only treat actual repeat_group IDs as nested instance arrays;
      // other arrays (e.g. multiple_choice values like ["Speeding"]) are flat fields.
      if (repeatGroupIds.has(fid) && Array.isArray(val)) {
        // Sync repeat group arrays to expected_outcome.repeat_groups
        if (!d.expected_outcome.repeat_groups[fid]) d.expected_outcome.repeat_groups[fid] = [];
        d.expected_outcome.repeat_groups[fid] = val.map((inst, idx) => {
          const existing = d.expected_outcome.repeat_groups[fid]?.[idx] || {};
          const synced = {};
          for (const [k, v] of Object.entries(inst)) {
            synced[k] = existing[k] || { expected: "", strategy: "exact", alternatives: [], evidence: "", evidence_source: "current_utterance", extraction_difficulty: "direct" };
            synced[k].expected = v;
          }
          return synced;
        });
        continue;
      }
      if (val == null && !eo[fid]) continue;
      if (!eo[fid]) eo[fid] = { expected: "", strategy: "exact", alternatives: [], evidence: "", evidence_source: "current_utterance", extraction_difficulty: "direct" };
      eo[fid].expected = val;
    }
  }

  // -- 3b: Forbidden Commits --
  normalizeForbiddenMutations(d);
  const forbiddenWrap = el("div", { class: "forbidden-subsection", style: "margin-top:20px;" });
  const forbiddenHeader = el("div", { class: "forbidden-header", style: "display:flex;align-items:center;gap:8px;cursor:pointer;padding:8px 0;" });
  const forbiddenToggle = el("span", { style: "font-size:14px;color:var(--text-3);transition:transform .2s;" }, "\u25be");
  const forbiddenBody = el("div", { class: "forbidden-body" });
  const forbiddenCount = el("span", { style: "font-size:11px;color:var(--help);font-style:italic;" });

  forbiddenHeader.append(
    forbiddenToggle,
    el("h3", { style: "font-size:13px;font-weight:600;color:var(--text-2);margin:0;" }, "Forbidden Commits"),
    forbiddenCount,
  );
  forbiddenHeader.addEventListener("click", () => {
    const collapsed = forbiddenBody.style.display === "none";
    forbiddenBody.style.display = collapsed ? "" : "none";
    forbiddenToggle.style.transform = collapsed ? "" : "rotate(-90deg)";
  });

  forbiddenWrap.append(forbiddenHeader, forbiddenBody);
  container.appendChild(forbiddenWrap);
  renderForbiddenMutations(forbiddenBody, d, fields, qTree, fieldMetaMap, forbiddenCount);

  // -- 3c: Item stressor profile --
  container.appendChild(el("div", { style: "margin-top:24px;border-top:1px solid var(--border);padding-top:16px;" }));
  container.appendChild(el("h3", { style: "font-size:13px;font-weight:600;color:var(--text);margin-bottom:4px;" }, "Human-Coded Item Stressors"));
  container.appendChild(el("p", { style: "font-size:12px;color:var(--help);margin-bottom:12px;line-height:1.5;" },
    "Pre-run item stressors only. Field count, primary delta, evidence properties, repeat-group involvement, and revision operation are derived above and again on save."));

  dimensionGridHost = el("div");
  container.appendChild(dimensionGridHost);
  renderDimensionGrid(dimensionGridHost, d);

  // -- 3d: Failure Modes --
  renderFailureModes(container, d);

  // Sync on initial render
  syncExpectedOutcome();
}

/** Normalize a value to the canonical string form used for comparison. */
function norm(v) {
  if (v === undefined || v === null || v === "") return "";
  if (Array.isArray(v) && v.length === 0) return "";
  if (v && typeof v === "object" && !Array.isArray(v) && Object.keys(v).length === 0) return "";
  if (typeof v === "object") return stableStringify(v);
  return String(v).trim();
}

function stableStringify(value) {
  if (Array.isArray(value)) return `[${value.map(stableStringify).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${stableStringify(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

function normalizeFieldType(type) {
  return String(type || "text").trim().toLowerCase().replace(/[-\s]+/g, "_");
}

/** Format a value for display (safe for arrays and objects). */
function displayVal(v) {
  if (v === undefined || v === null) return "";
  if (typeof v === "object") return stableStringify(v);
  return String(v);
}

function forbiddenSlot(mut) {
  return String(mut?.field ?? mut?.slot ?? mut?.field_path ?? "").trim();
}

function forbiddenWrongValue(mut) {
  if (!mut || typeof mut !== "object") return null;
  if ("wrong_value" in mut) return mut.wrong_value;
  if ("resulting_value" in mut) return mut.resulting_value;
  if ("forbidden_value" in mut) return mut.forbidden_value;
  return null;
}

function setForbiddenSlot(mut, slot) {
  mut.field = slot;
  mut.slot = slot;
  mut.field_path = slot;
}

function setForbiddenWrongValue(mut, value) {
  mut.wrong_value = value;
  mut.resulting_value = value;
}

function syncForbiddenCommits(d) {
  d.forbidden_commits = (d.forbidden_mutations || []).map(mut => ({ ...mut }));
}

function normalizeForbiddenMutations(d) {
  const mutationRows = Array.isArray(d.forbidden_mutations) ? d.forbidden_mutations : [];
  const commitRows = Array.isArray(d.forbidden_commits) ? d.forbidden_commits : [];
  const mutationsHaveContent = mutationRows.some(mut =>
    forbiddenSlot(mut) || forbiddenWrongValue(mut) != null || String(mut?.reason || "").trim()
  );
  const source = mutationsHaveContent ? mutationRows : commitRows;
  d.forbidden_mutations = source.map(raw => {
    const mut = raw && typeof raw === "object" ? { ...raw } : {};
    setForbiddenSlot(mut, forbiddenSlot(mut));
    setForbiddenWrongValue(mut, forbiddenWrongValue(mut));
    mut.reason = mut.reason || "";
    return mut;
  });
  syncForbiddenCommits(d);
}

export function classifyChange(priorVal, currentVal) {
  const p = norm(priorVal);
  const c = norm(currentVal);
  if (p === "" && c !== "") return "SET";
  if (p !== "" && c === "") return "CLEAR";
  if (p !== c) return "CHANGED";
  return null;
}

export function buildInlineDiff(kind, priorVal, currentVal) {
  const strip = el("div", { class: `gold-inline-diff gold-diff-${kind.toLowerCase()}` });
  const tag = el("span", { class: `gold-diff-tag gold-diff-tag-${kind.toLowerCase()}` }, kind);
  strip.appendChild(tag);

  if (kind === "SET") {
    strip.appendChild(el("span", { class: "gold-diff-empty" }, "(empty)"));
    strip.appendChild(el("span", { class: "gold-diff-arrow" }, "\u2192"));
    strip.appendChild(el("span", { class: "gold-diff-new" }, displayVal(currentVal)));
  } else if (kind === "CLEAR") {
    strip.appendChild(el("span", { class: "gold-diff-old" }, displayVal(priorVal)));
    strip.appendChild(el("span", { class: "gold-diff-arrow" }, "\u2192"));
    strip.appendChild(el("span", { class: "gold-diff-empty" }, "(empty)"));
  } else {
    strip.appendChild(el("span", { class: "gold-diff-old" }, displayVal(priorVal)));
    strip.appendChild(el("span", { class: "gold-diff-arrow" }, "\u2192"));
    strip.appendChild(el("span", { class: "gold-diff-new" }, displayVal(currentVal)));
  }
  return strip;
}

export function makeGoldFormInput(meta, currentVal, onChange) {
  const type = normalizeFieldType(meta.type);
  const options = Array.isArray(meta.options) ? meta.options : [];
  // ── multiple_choice: checkbox grid ──
  if (type === "multiple_choice" && options.length) {
    const wrap = el("div", { class: "gold-multi-choice-wrap" });
    const curArr = Array.isArray(currentVal) ? currentVal : (currentVal ? [currentVal] : []);
    const opts = meta.other_specify ? options.filter(o => String(o).toLowerCase() !== "other") : [...options];
    const knownSet = new Set(opts.map(o => String(o)));
    const otherVal = curArr.find(v => !knownSet.has(String(v))) ?? "";

    function collect() {
      const selected = [];
      wrap.querySelectorAll("input[type=checkbox][data-opt]").forEach(cb => {
        if (cb.checked) selected.push(cb.dataset.opt);
      });
      if (meta.other_specify) {
        const otherCb = wrap.querySelector("input[type=checkbox][data-other]");
        const otherInp = wrap.querySelector("input.gold-other-text");
        if (otherCb?.checked && otherInp?.value) selected.push(otherInp.value);
      }
      onChange(selected.length ? selected : null);
    }

    for (const opt of opts) {
      const lbl = el("label", { class: "gold-multi-choice-label" });
      const cb = el("input", { type: "checkbox", "data-opt": opt, ...(curArr.some(v => String(v) === String(opt)) ? { checked: "" } : {}) });
      cb.addEventListener("change", collect);
      lbl.append(cb, " " + opt);
      wrap.appendChild(lbl);
    }

    if (meta.other_specify) {
      const otherRow = el("div", { class: "gold-other-row" });
      const lbl = el("label", { class: "gold-multi-choice-label" });
      const cb = el("input", { type: "checkbox", "data-other": "1", ...(otherVal ? { checked: "" } : {}) });
      lbl.append(cb, " Other:");
      const inp = el("input", { type: "text", class: "gold-form-input gold-other-text", value: otherVal, placeholder: "Specify\u2026" });
      inp.style.display = otherVal || cb.checked ? "" : "none";
      cb.addEventListener("change", () => { inp.style.display = cb.checked ? "" : "none"; if (!cb.checked) inp.value = ""; collect(); });
      inp.addEventListener("input", collect);
      otherRow.append(lbl, inp);
      wrap.appendChild(otherRow);
    }
    return wrap;
  }

  // ── single_choice ──
  if (type === "single_choice" && options.length) {
    const opts = meta.other_specify ? options.filter(o => String(o).toLowerCase() !== "other") : [...options];
    const sel = el("select", { class: "gold-form-input gold-choice-input" });
    sel.append(el("option", { value: "" }, "\u2014"));
    const strVal = currentVal != null ? String(currentVal) : null;
    for (const opt of opts) {
      sel.append(el("option", { value: String(opt), ...(strVal !== null && String(opt) === strVal ? { selected: "" } : {}) }, String(opt)));
    }
    const matchesOption = opts.some(o => String(o) === strVal);
    if (meta.other_specify) {
      const isOther = strVal && !matchesOption;
      sel.append(el("option", { value: "__other__", ...(isOther ? { selected: "" } : {}) }, "Other\u2026"));
      const wrap = el("div", { class: "gold-other-wrap" });
      const otherInp = el("input", { type: "text", class: "gold-form-input gold-other-text", value: isOther ? strVal : "", placeholder: "Specify\u2026" });
      otherInp.style.display = isOther ? "" : "none";
      sel.addEventListener("change", () => {
        if (sel.value === "__other__") { otherInp.style.display = ""; otherInp.focus(); onChange(otherInp.value || null); }
        else { otherInp.style.display = "none"; otherInp.value = ""; onChange(sel.value || null); }
      });
      otherInp.addEventListener("input", () => onChange(otherInp.value || null));
      wrap.append(sel, otherInp);
      return wrap;
    }
    if (strVal !== null && !matchesOption) {
      sel.append(el("option", { value: strVal, selected: "" }, `${strVal} (not in options)`));
    } else if (strVal !== null) {
      sel.value = strVal;
    }
    sel.addEventListener("change", () => onChange(sel.value || null));
    return sel;
  }
  if (type === "bool") {
    const sel = el("select", { class: "gold-form-input" });
    sel.append(el("option", { value: "" }, "\u2014"));
    sel.append(el("option", { value: "true", ...(String(currentVal) === "true" ? { selected: "" } : {}) }, "true"));
    sel.append(el("option", { value: "false", ...(String(currentVal) === "false" ? { selected: "" } : {}) }, "false"));
    if (currentVal != null) sel.value = String(currentVal);
    sel.addEventListener("change", () => {
      const v = sel.value;
      onChange(v === "true" ? true : v === "false" ? false : null);
    });
    return sel;
  }

  // ── table: editable rows with column inputs ──
  if (type === "table" && meta.columns?.length) {
    const wrap = el("div", { class: "gold-table-wrap" });
    const rows = Array.isArray(currentVal) ? currentVal.map(r => ({...r})) : [];

    function fireChange() {
      const cleaned = rows.filter(r => Object.values(r).some(v => v != null && v !== ""));
      onChange(cleaned.length ? cleaned : null);
    }

    function renderTableRows() {
      wrap.innerHTML = "";
      // Header row
      const headerRow = el("div", { class: "gold-table-header" });
      for (const col of meta.columns) {
        headerRow.appendChild(el("span", { class: "gold-table-col-header", title: col.gold_standard || "" }, col.question_text || col.id));
      }
      headerRow.appendChild(el("span", { class: "gold-table-col-action" }, ""));
      wrap.appendChild(headerRow);

      // Data rows
      rows.forEach((row, rowIdx) => {
        const rowEl = el("div", { class: "gold-table-row" });
        for (const col of meta.columns) {
          const cell = el("div", { class: "gold-table-cell" });
          const colVal = row[col.id] ?? "";
          const colType = normalizeFieldType(col.type);

          if (colType === "single_choice" && col.options?.length) {
            const sel = el("select", { class: "gold-form-input gold-table-input" });
            sel.append(el("option", { value: "" }, "\u2014"));
            for (const opt of col.options) {
              sel.append(el("option", { value: String(opt), ...(String(colVal) === String(opt) ? { selected: "" } : {}) }, String(opt)));
            }
            if (colVal !== "" && !col.options.some(opt => String(opt) === String(colVal))) {
              sel.append(el("option", { value: String(colVal), selected: "" }, `${String(colVal)} (not in options)`));
            }
            sel.addEventListener("change", () => { row[col.id] = sel.value || null; fireChange(); });
            cell.appendChild(sel);
          } else {
            const inputType = colType === "number" ? "number" : "text";
            const inp = el("input", { type: inputType, class: "gold-form-input gold-table-input", value: colVal, placeholder: "\u2014" });
            inp.addEventListener("input", () => { row[col.id] = inp.value || null; fireChange(); });
            cell.appendChild(inp);
          }
          rowEl.appendChild(cell);
        }
        // Remove button
        const rmBtn = el("button", { class: "gold-table-rm", title: "Remove row", type: "button" }, "\u00d7");
        rmBtn.addEventListener("click", () => { rows.splice(rowIdx, 1); renderTableRows(); fireChange(); });
        rowEl.appendChild(rmBtn);
        wrap.appendChild(rowEl);
      });

      // Add row button
      const addBtn = el("button", { class: "btn sm", type: "button" }, "+ Add row");
      addBtn.addEventListener("click", () => {
        const newRow = {};
        for (const col of meta.columns) newRow[col.id] = null;
        rows.push(newRow);
        renderTableRows();
        fireChange();
      });
      wrap.appendChild(addBtn);
    }

    renderTableRows();
    return wrap;
  }

  if (type === "date") {
    return makeDateTextInput(currentVal, onChange, { class: "gold-form-input" });
  }

  const inputType = type === "time" ? "time" : type === "number" ? "number" : "text";
  const inp = el("input", { type: inputType, class: "gold-form-input", value: currentVal ?? "", placeholder: "\u2014" });
  const fire = () => onChange(inp.value || null);
  inp.addEventListener("input", fire);
  // time pickers on some browsers only fire "change", not "input"
  if (inputType === "time") inp.addEventListener("change", fire);
  return inp;
}

function buildForbiddenFieldOptions(qTree, goldState, fieldMetaMap) {
  const options = []; // { value, label, group }

  function matchesCondition(currentVal, expectedVal) {
    if (expectedVal === true && (currentVal === true || currentVal === "true")) return true;
    if (expectedVal === false && (currentVal === false || currentVal === "false")) return true;
    return String(expectedVal) === String(currentVal);
  }

  function isGateOpenForReader(gate, readValue) {
    if (!gate) return false;
    const val = readValue(gate.gate_on);
    if (val === undefined || val === null) return false;
    for (const wv of gate.when_values) {
      if (matchesCondition(val, wv)) return true;
    }
    return false;
  }

  function resolveBranchRouteForReader(branch, readValue) {
    if (!branch?.branch_on) return { route: null };
    const branchValue = readValue(branch.branch_on);
    for (const route of branch.routes || []) {
      if (matchesCondition(branchValue, route.when_value)) return { route };
    }
    const defaultChildren = branch.default_children || [];
    if (defaultChildren.length) return { route: { when_value: null, children: defaultChildren } };
    return { route: null };
  }

  function walkRepeatQuestions(questions, groupId, idx, groupLabel, readValue, sink = options) {
    for (const q of questions) {
      const st = q.structure_type || "regular";
      if (st === "regular") {
        const meta = fieldMetaMap[q.id];
        const childLabel = meta?.label || q.question_text || q.id;
        sink.push({ value: `${groupId}[${idx}].${q.id}`, label: childLabel, group: groupLabel });
      } else if (st === "group") {
        walkRepeatQuestions(q.fields || q.questions || [], groupId, idx, groupLabel, readValue, sink);
      } else if (st === "gate") {
        if (isGateOpenForReader(q.gate, readValue)) {
          walkRepeatQuestions(q.fields || q.questions || [], groupId, idx, groupLabel, readValue, sink);
        }
      } else if (st === "branch") {
        const { route } = resolveBranchRouteForReader(q.branch, readValue);
        if (route) {
          walkRepeatQuestions(route.children || [], groupId, idx, groupLabel, readValue, sink);
        }
      }
    }
  }

  function walk(questions, groupLabel) {
    for (const q of questions) {
      const st = q.structure_type || "regular";
      if (st === "regular") {
        const meta = fieldMetaMap[q.id];
        const label = meta?.label || q.question_text || q.id;
        options.push({ value: q.id, label, group: groupLabel || "Fields" });
      } else if (st === "group") {
        walk(q.fields || q.questions || [], q.label || q.id);
      } else if (st === "gate") {
        if (isGateOpenForReader(q.gate, (fieldId) => goldState[fieldId])) {
          walk(q.fields || q.questions || [], q.label || q.id);
        }
      } else if (st === "branch") {
        const { route } = resolveBranchRouteForReader(q.branch, (fieldId) => goldState[fieldId]);
        if (route) {
          walk(route.children || [], q.label || q.id);
        }
      } else if (st === "repeat_group") {
        const groupId = q.id;
        const arr = goldState[groupId];
        const childFields = q.fields || q.questions || [];
        const count = Array.isArray(arr) ? arr.length : 0;
        const itemLabel = q.repeat?.item_label || `${q.label || groupId} {{index}}`;
        for (let i = 0; i < count; i++) {
          const instLabel = itemLabel.replace("{{index}}", String(i + 1));
          walkRepeatQuestions(
            childFields,
            groupId,
            i,
            instLabel,
            (fieldId) => (arr[i] || {})[fieldId],
          );
        }
        // Also add a generic (any instance) entry for each child field
        if (count > 0) {
          const seen = new Set();
          for (let i = 0; i < count; i++) {
            const instanceOptions = [];
            walkRepeatQuestions(
              childFields,
              groupId,
              i,
              q.label || groupId,
              (fieldId) => (arr[i] || {})[fieldId],
              instanceOptions,
            );
            for (const opt of instanceOptions) {
              const genericValue = opt.value.replace(`[${i}]`, "[*]");
              if (seen.has(genericValue)) continue;
              seen.add(genericValue);
              options.push({
                value: genericValue,
                label: `${opt.label} (any instance)`,
                group: q.label || groupId,
              });
            }
          }
        }
      }
    }
  }

  walk(qTree, "");
  return options;
}

function renderForbiddenMutations(container, d, fields, qTree, fieldMetaMap, countEl = null) {
  container.innerHTML = "";
  container.appendChild(el("p", { style: "font-size:12px;color:var(--help);margin-top:8px;" },
    "Record outcomes the agent must NOT produce. What wrong resulting value might it commit?"));
  const list = el("div", { class: "action-list" });
  container.appendChild(list);

  const fieldOptions = buildForbiddenFieldOptions(qTree || [], d.gold_resulting_state || {}, fieldMetaMap || {});

  function buildFieldSelect(currentValue) {
    const sel = el("select", { class: "gold-form-input" });
    sel.append(el("option", { value: "" }, "\u2014 select field \u2014"));

    // Group options by optgroup
    const groups = new Map();
    for (const opt of fieldOptions) {
      if (!groups.has(opt.group)) groups.set(opt.group, []);
      groups.get(opt.group).push(opt);
    }
    for (const [groupLabel, opts] of groups) {
      const og = el("optgroup", { label: groupLabel });
      for (const opt of opts) {
        og.append(el("option", { value: opt.value, ...(opt.value === currentValue ? { selected: "" } : {}) }, `${opt.label}  (${opt.value})`));
      }
      sel.appendChild(og);
    }

    // If current value is not in options, add it so it's still visible
    if (currentValue && !fieldOptions.some(o => o.value === currentValue)) {
      const extra = el("optgroup", { label: "Custom" });
      extra.append(el("option", { value: currentValue, selected: "" }, currentValue));
      sel.insertBefore(extra, sel.firstChild.nextSibling);
    }

    return sel;
  }

  function renderAllRows() {
    list.innerHTML = "";
    if (countEl) countEl.textContent = `(${d.forbidden_mutations.length} defined)`;
    d.forbidden_mutations.forEach((mut, idx) => {
      const row = el("div", { class: "action-row" });
      const rmBtn = el("button", { class: "action-row-remove", title: "Remove" }, "\u00d7");
      rmBtn.addEventListener("click", () => {
        d.forbidden_mutations.splice(idx, 1);
        syncForbiddenCommits(d);
        renderAllRows();
        markDirty();
      });

      const fieldSel = buildFieldSelect(forbiddenSlot(mut));
      fieldSel.addEventListener("change", () => {
        setForbiddenSlot(mut, fieldSel.value);
        syncForbiddenCommits(d);
        markDirty();
      });
      const valInp = el("input", { type: "text", value: displayVal(forbiddenWrongValue(mut)), placeholder: "wrong resulting value the agent might produce" });
      valInp.addEventListener("input", () => {
        setForbiddenWrongValue(mut, valInp.value || null);
        syncForbiddenCommits(d);
        markDirty();
      });
      const reasonTa = el("textarea", { placeholder: "Why is this wrong?", style: "min-height:54px;" });
      reasonTa.value = mut.reason || "";
      reasonTa.addEventListener("input", () => {
        mut.reason = reasonTa.value;
        syncForbiddenCommits(d);
        markDirty();
      });

      function f(label, child) { return el("div", {}, el("label", {}, label), child); }
      row.append(rmBtn, el("div", { class: "action-grid" }, f("Slot", fieldSel), f("Wrong resulting value", valInp)), el("div", { style: "margin-top:6px;" }, f("Reason", reasonTa)));
      list.appendChild(row);
    });
  }
  renderAllRows();
  const addBtn = el("button", { class: "btn sm add-row-btn", type: "button" }, "+ Add forbidden commit");
  addBtn.addEventListener("click", () => {
    const mut = { field: "", slot: "", field_path: "", wrong_value: null, resulting_value: null, reason: "" };
    d.forbidden_mutations.push(mut);
    syncForbiddenCommits(d);
    renderAllRows();
    markDirty();
  });
  container.appendChild(addBtn);
}

function renderDimensionGrid(container, d) {
  container.innerHTML = "";
  const dp = d.difficulty_profile;
  const grid = el("div", { class: "dimension-grid", style: "display:grid;grid-template-columns:1fr 1fr;gap:8px 16px;" });
  const repeatGroupInvolvement = d.derived_variables?.repeat_group_involvement || "none";
  const repeatGroupApplicable = repeatGroupInvolvement !== "none";

  for (const dim of DIMENSIONS) {
    let val = dp.dimensions[dim.key] || DIMENSION_DEFAULTS[dim.key];
    const isRepeatDimension = dim.key === "repeat_instance_routing_pressure";
    const isNonDefault = val !== DIMENSION_DEFAULTS[dim.key];

    const cell = el("div", {
      class: `dimension-cell${isNonDefault ? " active" : ""}`,
      style: `padding:8px 10px;border-radius:6px;border:1px solid ${isNonDefault ? "var(--primary)" : "var(--border)"};background:${isNonDefault ? "var(--primary-bg, #eef2ff)" : "var(--bg)"};`,
    });

    const labelRow = el("div", { style: "display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;" });
    const nameEl = el("span", { style: "font-size:12px;font-weight:600;color:var(--text);" }, dim.label);
    const sel = el("select", { style: "font-size:11px;padding:2px 4px;border:1px solid var(--border);border-radius:4px;" });
    for (const v of dim.values) {
      sel.append(el("option", { value: v, ...(v === val ? { selected: "" } : {}) }, v));
    }
    labelRow.append(nameEl, sel);

    const tipEl = el("div", { style: "font-size:10px;color:var(--text-3);line-height:1.3;" }, dim.tip);
    const warningEl = dim.warning
      ? el("div", {
          style: "font-size:10px;line-height:1.35;margin-top:4px;padding:4px 6px;border-radius:4px;background:rgba(245,158,11,.12);color:#92400e;",
        }, dim.warning)
      : null;
    const applicabilityEl = isRepeatDimension
      ? el("div", {
          style: `font-size:10px;line-height:1.35;margin-top:4px;padding:4px 6px;border-radius:4px;background:${repeatGroupApplicable ? "rgba(2,132,199,.10)" : "rgba(148,163,184,.12)"};color:${repeatGroupApplicable ? "#0c4a6e" : "#475569"};`,
        }, repeatGroupApplicable
          ? `Derived repeat-group involvement is ${repeatGroupInvolvement}; non-none repeat pressure may be appropriate, but it is still a human-coded judgment rather than a forced consequence of the gold diff.`
          : "Derived repeat-group involvement is none; non-none repeat pressure is still allowed when the item tests preserving, avoiding, creating, or attaching repeated instances without a gold repeat-group change.")
      : null;
    const selectedRule = el("div", {
      style: "font-size:10px;color:var(--text-2);line-height:1.35;margin-top:4px;padding:4px 6px;border-radius:4px;background:rgba(148,163,184,.10);",
    }, dim.rules?.[val] || "");

    const guidanceEl = el("details", { style: "margin-top:4px;" },
      el("summary", { style: "font-size:10px;color:var(--text-3);cursor:pointer;" }, "coding guidance"),
      el("div", { style: "display:flex;flex-direction:column;gap:6px;margin-top:6px;" },
        el("div", { style: "font-size:10px;color:var(--text-3);line-height:1.35;" },
          el("strong", {}, "What counts: "),
          dim.countsAs || "—",
        ),
        el("div", { style: "font-size:10px;color:var(--text-3);line-height:1.35;" },
          el("strong", {}, "What does not count: "),
          dim.notCountsAs || "—",
        ),
        el("div", { style: "font-size:10px;color:var(--text-3);line-height:1.35;" },
          el("strong", {}, "Contrastive example: "),
          dim.example || "—",
        ),
        el("div", { style: "display:flex;flex-direction:column;gap:3px;" },
          ...dim.values.map(v => el("div", { style: "font-size:10px;color:var(--text-3);line-height:1.3;" },
            el("strong", {}, `${v}: `),
            dim.rules?.[v] || "",
          )),
        ),
      ),
    );

    // Rationale note. Non-default levels require an observable cue before ready.
    const noteInp = el("input", {
      type: "text",
      value: dp.dimension_notes[dim.key] || "",
      placeholder: isNonDefault ? `required: ${dim.notePrompt || "observable cue or count"}` : "optional baseline note",
      style: "font-size:10px;padding:2px 4px;border:1px solid transparent;border-radius:3px;width:100%;margin-top:2px;background:transparent;color:var(--text-2);",
    });

    sel.addEventListener("change", () => {
      dp.dimensions[dim.key] = sel.value;
      const noneOption = sel.querySelector("option[value='none']");
      if (isRepeatDimension && repeatGroupApplicable && noneOption) {
        noneOption.disabled = sel.value !== "none";
      }
      const newIsNonDefault = sel.value !== DIMENSION_DEFAULTS[dim.key];
      const invalidRepeat = isRepeatDimension && repeatGroupApplicable && sel.value === "none";
      cell.style.border = `1px solid ${invalidRepeat ? "#dc2626" : newIsNonDefault ? "var(--primary)" : "var(--border)"}`;
      cell.style.background = newIsNonDefault ? "var(--primary-bg, #eef2ff)" : "var(--bg)";
      cell.className = `dimension-cell${newIsNonDefault ? " active" : ""}`;
      selectedRule.textContent = dim.rules?.[sel.value] || "";
      noteInp.placeholder = newIsNonDefault ? `required: ${dim.notePrompt || "observable cue or count"}` : "optional baseline note";
      markDirty();
      updateChecklist();
    });
    noteInp.addEventListener("focus", () => { noteInp.style.border = "1px solid var(--border)"; noteInp.style.background = "var(--bg)"; });
    noteInp.addEventListener("blur", () => {
      if (!noteInp.value) { noteInp.style.border = "1px solid transparent"; noteInp.style.background = "transparent"; }
    });
    noteInp.addEventListener("input", () => {
      dp.dimension_notes[dim.key] = noteInp.value;
      markDirty();
    });

    cell.append(labelRow, tipEl);
    if (warningEl) cell.appendChild(warningEl);
    if (applicabilityEl) cell.appendChild(applicabilityEl);
    cell.append(selectedRule, guidanceEl, noteInp);
    grid.appendChild(cell);
  }

  container.appendChild(grid);
}

function renderFailureModes(container, d) {
  const dp = d.difficulty_profile;

  const failWrap = el("div", { style: "margin-top:16px;" });
  failWrap.append(
    el("h3", { style: "font-size:13px;font-weight:600;color:var(--text);margin-bottom:4px;" }, "Targeted Failure Modes"),
  );

  const tagContainer = el("div", { class: "tags-container", style: "margin-top:8px;" });
  const tagInput = el("input", { class: "tags-input", placeholder: "e.g. premature_commitment, press Enter..." });
  const currentTags = () => Array.from(tagContainer.querySelectorAll(".tag")).map(t => t.dataset.value);

  function addTag(val) {
    val = String(val || "").trim().replace(/[^a-z0-9_]/gi, "_").toLowerCase();
    if (!val || currentTags().includes(val)) return;
    const tag = el("span", { class: "tag", dataset: { value: val } }, val);
    const rm = el("button", { type: "button", "aria-label": "remove" }, "\u00d7");
    rm.addEventListener("click", () => { tag.remove(); dp.targeted_failure_modes = currentTags(); markDirty(); updateChecklist(); });
    tag.appendChild(rm);
    tagContainer.insertBefore(tag, tagInput);
    dp.targeted_failure_modes = currentTags();
    markDirty();
    updateChecklist();
  }

  tagInput.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === ",") { e.preventDefault(); addTag(tagInput.value); tagInput.value = ""; } });
  tagContainer.addEventListener("click", () => tagInput.focus());
  tagContainer.appendChild(tagInput);
  (dp.targeted_failure_modes || []).filter(v => v && v !== "TODO").forEach(v => addTag(v));

  const chips = el("div", { style: "display:flex;flex-wrap:wrap;gap:6px;margin-top:8px;" });
  for (const s of FAILURE_MODE_SUGGESTIONS) {
    const chip = el("button", { type: "button", style: "font-size:11px;padding:3px 8px;border-radius:4px;border:1px solid var(--border);background:var(--bg);cursor:pointer;color:var(--text-2);" }, s);
    chip.addEventListener("click", () => addTag(s));
    chips.appendChild(chip);
  }

  // Failure explanation
  const explGroup = el("div", { class: "field-group", style: "margin-top:12px;" });
  const explTa = el("textarea", { placeholder: "In one sentence, what specific error should this item expose?", style: "min-height:60px;" });
  explTa.value = dp.failure_explanation || "";
  explTa.addEventListener("input", () => { dp.failure_explanation = explTa.value; markDirty(); });
  explGroup.append(el("label", {}, "Failure explanation"), explTa);

  failWrap.append(tagContainer, el("p", { class: "help-text" }, "Suggestions (click to add):"), chips, explGroup);
  container.appendChild(failWrap);
}

// =============================================================================
// SECTION 4: Status & Advanced (Save & Status + materialization)
// =============================================================================

export function renderStatusAndAdvanced(container) {
  const d = S.currentData;

  // -- Continue as Next Turn (prominent action) --
  const continueWrap = el("div", { class: "continue-turn-banner" });
  const continueInfo = el("div", { style: "flex:1;" });
  continueInfo.append(
    el("div", { style: "font-size:13px;font-weight:600;color:var(--text);" }, "Continue as next turn"),
    el("div", { style: "font-size:11px;color:var(--text-3);margin-top:2px;line-height:1.4;" },
      "Create a new item using this item\u2019s gold state as its prior state. Auto-creates state & history contexts."),
  );
  const continueBtn = el("button", { class: "btn sm continue-turn-btn", type: "button" }, "\u2192 Next turn");
  const goldReady = d.gold_resulting_state && typeof d.gold_resulting_state === "object" && Object.values(d.gold_resulting_state).some(v => v != null && v !== "");
  if (!goldReady) {
    continueBtn.disabled = true;
    continueBtn.title = "Fill in the gold resulting state first";
  }
  continueBtn.addEventListener("click", async () => {
    const { continueAsNextTurn } = await import("./dialogs.js");
    await continueAsNextTurn();
  });
  continueWrap.append(continueInfo, continueBtn);
  container.appendChild(continueWrap);

  // -- Status toggle --
  const bar = el("div", { class: "save-bar" });
  const statusGroup = el("div", { class: "status-group" });
  statusGroup.append(el("label", {}, "Status:"));
  for (const [val, label, cls] of [["template", "Template", ""], ["draft", "Draft", ""], ["ready", "Ready", "success"]]) {
    const btn = el("button", { class: `btn sm${d.status === val ? ` ${cls || "primary"}` : ""}`, type: "button" }, label);
    btn.addEventListener("click", () => {
      if (val === "ready" && !validateForReady()) return;
      d.status = val;
      container.innerHTML = "";
      renderStatusAndAdvanced(container);
      markDirty();
    });
    statusGroup.appendChild(btn);
  }
  const hint = el("div", { class: "help-text" }, "Once saved, run python main.py materialize-pilot.");
  bar.append(statusGroup, hint);
  container.appendChild(bar);

  // -- Author notes --
  const notesGroup = el("div", { class: "field-group", style: "margin-top:16px;" });
  const notesTa = el("textarea", { placeholder: "Open questions, implementation notes, or benchmark-maintainer comments...", style: "min-height:60px;" });
  notesTa.value = d.author_notes || "";
  notesTa.addEventListener("input", () => { d.author_notes = notesTa.value; markDirty(); });
  notesGroup.append(el("label", {}, "Author notes"), notesTa, el("p", { class: "help-text" }, "Free text for open questions or benchmark-maintainer comments."));
  container.appendChild(notesGroup);

  // -- Advanced (collapsed by default) --
  const advWrap = el("div", { class: "advanced-section", style: "margin-top:16px;" });
  const advHeader = el("div", { class: "advanced-header", style: "display:flex;align-items:center;gap:8px;cursor:pointer;padding:8px 0;border-top:1px solid var(--border);" });
  const advToggle = el("span", { style: "font-size:14px;color:var(--text-3);transition:transform .2s;transform:rotate(-90deg);" }, "\u25be");
  const advBody = el("div", { style: "display:none;" });

  advHeader.append(
    advToggle,
    el("h3", { style: "font-size:13px;font-weight:600;color:var(--text-2);margin:0;" }, "Advanced (materialization config)"),
  );
  advHeader.addEventListener("click", () => {
    const collapsed = advBody.style.display === "none";
    advBody.style.display = collapsed ? "" : "none";
    advToggle.style.transform = collapsed ? "" : "rotate(-90deg)";
  });

  const mat = d.materialization || {};

  function advField(label, value, onChange) {
    const g = el("div", { class: "field-group" });
    const inp = el("input", { type: "text", value: value || "" });
    inp.addEventListener("input", () => { onChange(inp.value); markDirty(); });
    g.append(el("label", {}, label), inp);
    return g;
  }

  advBody.append(
    el("p", { style: "font-size:12px;color:var(--help);margin:12px 0 8px;" }, "These are auto-populated. Only edit if you know what you're doing."),
    el("div", { class: "field-row" },
      advField("scenario_name", mat.scenario_name, v => { mat.scenario_name = v; }),
      advField("difficulty", mat.difficulty, v => { mat.difficulty = v; d.difficulty_tier = v; }),
    ),
    el("div", { class: "field-row" },
      advField("state_id", mat.state_id, v => { mat.state_id = v; }),
      advField("utterance_id", mat.utterance_id, v => { mat.utterance_id = v; }),
    ),
    advField("iu_description", mat.iu_description, v => { mat.iu_description = v; }),
  );

  advWrap.append(advHeader, advBody);
  container.appendChild(advWrap);
}

// -- Validate for Ready -------------------------------------------------------

export function validateForReady() {
  const d = S.currentData;
  const issues = [];
  if (!d.current_utterance?.text || d.current_utterance.text.includes("TODO")) issues.push("utterance is empty or still a TODO");
  if (!d.difficulty_tier) issues.push("difficulty tier is missing");
  ensureDifficultyProfile(d);
  const dp = d.difficulty_profile;
  for (const dim of DIMENSIONS) {
    const val = dp.dimensions?.[dim.key];
    if (!val) {
      issues.push(`${dim.label} is missing`);
      continue;
    }
    if (!dim.values.includes(val)) {
      issues.push(`${dim.label} has invalid level '${val}'`);
      continue;
    }
    if (val !== DIMENSION_DEFAULTS[dim.key] && !(dp.dimension_notes?.[dim.key] || "").trim()) {
      issues.push(`${dim.label} needs an observable rationale`);
    }
  }
  const requiresFailureContract = readyItemRequiresFailureContract(d);
  const targetedModes = cleanTargetedFailureModes(d);
  if (requiresFailureContract && !targetedModes.length) issues.push("at least one targeted failure mode is required");
  if (requiresFailureContract && !(dp.failure_explanation || "").trim()) issues.push("failure explanation is required");
  const forbidden = [
    ...(Array.isArray(d.forbidden_commits) ? d.forbidden_commits : []),
    ...(Array.isArray(d.forbidden_mutations) ? d.forbidden_mutations : []),
  ];
  const completeForbidden = forbidden.filter(fm => {
    const slot = String(fm?.slot || fm?.field || "").trim();
    const hasValue = fm?.resulting_value != null || fm?.wrong_value != null;
    const reason = String(fm?.reason || "").trim();
    return slot && hasValue && reason;
  });
  const partialForbidden = forbidden.filter(fm => {
    const slot = String(fm?.slot || fm?.field || "").trim();
    const hasValue = fm?.resulting_value != null || fm?.wrong_value != null;
    const reason = String(fm?.reason || "").trim();
    return slot || hasValue || reason;
  }).length - completeForbidden.length;
  if (partialForbidden > 0) issues.push("every forbidden commit needs a slot, wrong resulting value, and reason");
  if ((dp.dimensions?.unsupported_alternative_affordance || "none") !== "none" && !completeForbidden.length) {
    issues.push("unsupported alternative affordance requires a documented forbidden commit");
  }
  if (!d.evidence) d.evidence = { ...EVIDENCE_DEFAULTS };
  const supportDistance = Number(d.evidence.support_distance || 0);
  if (supportDistance > 0 && !d.evidence.history_required) issues.push("support_distance > 0 requires history_required");
  if (d.evidence.history_required && supportDistance < 1) issues.push("history_required items need support_distance >= 1");
  const grs = d.gold_resulting_state;
  if (grs && typeof grs === "object") {
    const vals = Object.values(grs);
    if (!vals.length) issues.push("no fields in gold resulting state");
    else if (!vals.some(v => v != null && v !== "")) issues.push("gold resulting state has no non-empty values");
  } else {
    issues.push("gold resulting state is missing");
  }
  if (issues.length) {
    showToast("Fix before marking ready: " + issues.join("; "), "err");
    return false;
  }
  return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// CHANGE REF PICKER (dropdown to select an existing context)
// ═══════════════════════════════════════════════════════════════════════════════

function showChangeRefPicker(kind) {
  const d = S.currentData;
  if (!d) return;
  const available = (S.allContexts[kind] || []).filter(c => {
    // Optionally filter by questionnaire match
    const q = d.questionnaire?.source;
    return !q || c.questionnaire === q;
  });

  if (!available.length) {
    showToast(`No existing ${kind} contexts found for this questionnaire. Create one first.`, "err");
    return;
  }

  // Build a small modal with a dropdown
  const overlay = el("div", { class: "context-modal-overlay" });
  const panel = el("div", { class: "context-modal-panel", style: "width:420px;height:auto;max-height:60vh;" });

  const header = el("div", { class: "context-modal-header" });
  header.append(
    el("h3", {}, `Select ${kind} context`),
    el("button", { class: "btn sm", type: "button", onclick: () => overlay.remove() }, "\u2715"),
  );

  const body = el("div", { class: "context-modal-body", style: "padding:16px 20px;" });
  body.append(el("p", { style: "font-size:12px;color:var(--help);margin-bottom:12px;" },
    `Choose from existing ${kind} contexts for this questionnaire.`));

  const sel = el("select", { style: "width:100%;padding:8px;border:1px solid var(--border);border-radius:6px;font-size:13px;" });
  sel.append(el("option", { value: "" }, `\u2014 pick a ${kind} context \u2014`));
  const currentRef = kind === "state" ? d.state_ref : d.history_ref;
  for (const ctx of available) {
    const label = `${ctx.ref} (${ctx.condition_code}${ctx.description ? " \u2014 " + ctx.description.slice(0, 40) : ""})`;
    sel.append(el("option", { value: ctx.ref, ...(ctx.ref === currentRef ? { selected: "" } : {}) }, label));
  }
  body.appendChild(sel);

  // Preview area
  const previewArea = el("div", { style: "margin-top:12px;min-height:40px;" });
  body.appendChild(previewArea);

  sel.addEventListener("change", async () => {
    previewArea.innerHTML = "";
    if (!sel.value) return;
    try {
      const ctx = await apiGet(`/api/contexts/${kind}/${sel.value}`);
      const previewBody = el("div", { class: "context-pane-body", style: "max-height:200px;overflow-y:auto;border:1px solid var(--border);border-radius:6px;padding:8px;" });
      if (kind === "state") renderStateBody(previewBody, ctx);
      else renderHistoryBody(previewBody, ctx);
      previewArea.appendChild(previewBody);
    } catch {
      previewArea.appendChild(el("p", { style: "font-size:12px;color:var(--text-3);font-style:italic;" }, "Could not load preview."));
    }
  });

  const footer = el("div", { class: "context-modal-footer" });
  const cancelBtn = el("button", { class: "btn sm", type: "button" }, "Cancel");
  const applyBtn = el("button", { class: "btn sm primary", type: "button" }, "Apply");

  cancelBtn.addEventListener("click", () => overlay.remove());
  applyBtn.addEventListener("click", async () => {
    if (!sel.value) {
      showToast("Pick a context first", "err");
      return;
    }
    try {
      const ctx = await apiGet(`/api/contexts/${kind}/${sel.value}`);
      if (kind === "state") {
        d.state_ref = sel.value;
        d.prior_state = ctx.questionnaire_answers || {};
        d.current_state = { questionnaire_answers: ctx.questionnaire_answers || null };
      } else {
        d.history_ref = sel.value;
        d.visible_history = ctx.turns || [];
      }
    } catch {
      if (kind === "state") d.state_ref = sel.value;
      else d.history_ref = sel.value;
    }
    markDirty();
    overlay.remove();
    refreshContextPreview();
  });

  overlay.addEventListener("click", (e) => { if (e.target === overlay) overlay.remove(); });

  footer.append(cancelBtn, applyBtn);
  panel.append(header, body, footer);
  overlay.appendChild(panel);
  document.body.appendChild(overlay);
}

// ═══════════════════════════════════════════════════════════════════════════════
// INLINE CONTEXT MODAL (slide-over panel)
// ═══════════════════════════════════════════════════════════════════════════════

async function openInlineContextModal(kind) {
  const d = S.currentData;
  if (!d) return;

  const questionnaire = d.questionnaire?.source || S.questionnaireNames[0] || "";
  const ref = kind === "state" ? d.state_ref : d.history_ref;
  const conditionCode = kind === "state" ? d.state_condition : legacyHistoryConditionFromItem(d);

  // Load or create context data
  let contextData;
  let contextRef = ref;
  let isNew = false;

  if (ref) {
    try {
      contextData = await apiGet(`/api/contexts/${kind}/${ref}`);
    } catch {
      // Context doesn't exist yet, create a draft
      isNew = true;
    }
  } else {
    isNew = true;
  }

  if (isNew) {
    contextRef = ref || `${questionnaire}/${d.item_id || "new_item"}_${kind === "state" ? "prior_state" : "prior_history"}`;
    if (kind === "state") {
      contextData = {
        condition_code: conditionCode || "S1",
        questionnaire,
        description: "",
        questionnaire_answers: (conditionCode || "S1") === "S1" ? null : {},
      };
    } else {
      contextData = {
        condition_code: conditionCode || "H1",
        questionnaire,
        description: "",
        turns: [],
      };
    }
  }

  // ── Check if context is shared by multiple items ──
  let otherUsers = [];
  if (!isNew && contextRef) {
    try {
      const usage = await apiGet(`/api/contexts/${kind}/${contextRef}/usage`);
      otherUsers = (usage.items || []).filter(u => u.item_id !== d.item_id);
    } catch { /* ignore — non-critical */ }
  }

  // ── Build the slide-over modal ──
  const overlay = el("div", { class: "context-modal-overlay" });
  const panel = el("div", { class: "context-modal-panel" });

  const header = el("div", { class: "context-modal-header" });
  const titleLabel = kind === "state" ? "Edit Form State" : "Edit Conversation History";
  header.append(
    el("h3", {}, isNew ? `Create ${kind} context` : titleLabel),
    el("button", { class: "btn sm", type: "button", onclick: () => overlay.remove() }, "\u2715 Close"),
  );

  const body = el("div", { class: "context-modal-body" });

  // ── Shared-state warning banner ──
  if (otherUsers.length > 0) {
    const names = otherUsers.map(u => u.item_id).join(", ");
    const bannerText = otherUsers.length === 1
      ? `This ${kind} is also used by ${names}. Edits here will affect that item too.`
      : `This ${kind} is shared by ${otherUsers.length} other items (${names}). Edits here will affect all of them.`;

    const detachBtn = el("button", {
      class: "btn sm",
      type: "button",
      style: "margin-left:8px;white-space:nowrap;",
    }, "\u2702 Detach — make a private copy");

    detachBtn.addEventListener("click", async () => {
      try {
        detachBtn.disabled = true;
        detachBtn.textContent = "Detaching\u2026";
        const newRef = `${questionnaire}/${d.item_id}_${kind}`;
        await apiPost(`/api/contexts/${kind}/${contextRef}/copy`, { new_ref: newRef });

        // Re-point this item to the private copy
        if (kind === "state") {
          d.state_ref = newRef;
        } else {
          d.history_ref = newRef;
        }
        contextRef = newRef;
        S.currentContextRef = newRef;
        markDirty();
        await loadContexts();

        // Remove the warning banner and show confirmation
        banner.remove();
        showToast(`\u2713 Detached — now editing private copy: ${newRef}`, "ok");
      } catch (e) {
        detachBtn.disabled = false;
        detachBtn.textContent = "\u2702 Detach — make a private copy";
        showToast(`\u2717 Detach failed: ${e.message}`, "err");
      }
    });

    const banner = el("div", {
      class: "shared-context-warning",
      style: "display:flex;align-items:center;gap:8px;padding:10px 14px;margin-bottom:12px;border-radius:6px;background:#fef3c7;border:1px solid #f59e0b;color:#92400e;font-size:12px;line-height:1.5;",
    },
      el("span", { style: "font-size:16px;" }, "\u26a0\ufe0f"),
      el("span", { style: "flex:1;" }, bannerText),
      detachBtn,
    );
    body.appendChild(banner);
  }

  // Save the current S context state so we can restore it later
  const savedContextKind = S.currentContextKind;
  const savedContextRef = S.currentContextRef;
  const savedContextData = S.currentContextData;

  // Temporarily set S context state for the context builder renderers
  S.currentContextKind = kind;
  S.currentContextRef = contextRef;
  S.currentContextData = contextData;

  // Render the context content builder (lazy import to avoid circular deps)
  const ctxMod = await import("./context-sections.js");
  const contentWrap = el("div", { style: "margin-top:12px;" });
  if (kind === "state") {
    await ctxMod.renderStateContextSection(contentWrap);
  } else {
    ctxMod.renderHistoryContextSection(contentWrap);
  }
  body.appendChild(contentWrap);

  const footer = el("div", { class: "context-modal-footer" });
  const cancelBtn = el("button", { class: "btn sm", type: "button" }, "Cancel");
  const saveBtn = el("button", { class: "btn sm primary", type: "button" }, "\ud83d\udcbe Save context");

  cancelBtn.addEventListener("click", () => {
    // Restore state
    S.currentContextKind = savedContextKind;
    S.currentContextRef = savedContextRef;
    S.currentContextData = savedContextData;
    overlay.remove();
  });

  saveBtn.addEventListener("click", async () => {
    try {
      saveBtn.disabled = true;
      saveBtn.textContent = "Saving\u2026";
      await apiPut(`/api/contexts/${kind}/${contextRef}`, contextData);

      // Update item refs if needed
      if (kind === "state") {
        d.state_ref = contextRef;
        d.prior_state = contextData.questionnaire_answers || {};
        d.current_state = { questionnaire_answers: contextData.questionnaire_answers || null };
      } else {
        d.history_ref = contextRef;
        d.visible_history = contextData.turns || [];
      }
      markDirty();

      // Restore state
      S.currentContextKind = savedContextKind;
      S.currentContextRef = savedContextRef;
      S.currentContextData = savedContextData;

      // Refresh context list so sidebar and other pickers see the saved context
      await loadContexts();

      overlay.remove();
      showToast(`\u2713 ${kind} context saved`, "ok");
      refreshContextPreview();
    } catch (e) {
      saveBtn.disabled = false;
      saveBtn.textContent = "\ud83d\udcbe Save context";
      showToast(`\u2717 Save failed: ${e.message}`, "err");
    }
  });

  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) {
      S.currentContextKind = savedContextKind;
      S.currentContextRef = savedContextRef;
      S.currentContextData = savedContextData;
      overlay.remove();
    }
  });

  footer.append(cancelBtn, saveBtn);
  panel.append(header, body, footer);
  overlay.appendChild(panel);
  document.body.appendChild(overlay);
}

// =============================================================================
// SECTION REGISTRY (4 sections)
// =============================================================================

export function getItemSections() {
  return [
    { id: "section-scenario", n: 1, title: "Scenario Setup", sub: "Identity, questionnaire, prior state, delta type, and evidence.", render: body => renderScenarioSetup(body) },
    { id: "section-context-utt", n: 2, title: "Context & Utterance", sub: "Review context, then write the user message -- the core creative work.", render: body => renderContextAndUtterance(body) },
    { id: "section-gold", n: 3, title: "Gold Resulting State & Difficulty", sub: "Edit the correct record state after the utterance. Diff is then materialized from your authored prior and gold states.", render: body => renderExpectedOutcomeAndDifficulty(body) },
    { id: "section-status", n: 4, title: "Status & Notes", sub: "Save status, author notes, and materialization config.", render: body => renderStatusAndAdvanced(body) },
  ];
}
