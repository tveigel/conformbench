// ── DOM utilities & shared helpers ────────────────────────────────────────────
import { S } from "./state.js";

export function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "dataset") Object.assign(e.dataset, v);
    else if (k === "style" && typeof v === "object") Object.assign(e.style, v);
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (k === "checked") e.checked = (v != null && v !== false);
    else if (k === "selected") e.selected = (v != null && v !== false);
    else if (k === "disabled") e.disabled = (v != null && v !== false);
    else if (k === "hidden") e.hidden = (v != null && v !== false);
    else if (k === "value") e.value = v;
    else e.setAttribute(k, v);
  }
  for (const c of children) {
    if (c == null) continue;
    e.append(typeof c === "string" ? document.createTextNode(c) : c);
  }
  return e;
}

function validDateParts(year, month, day) {
  if (year < 1 || year > 9999 || month < 1 || month > 12 || day < 1 || day > 31) {
    return false;
  }
  const d = new Date(Date.UTC(year, month - 1, day));
  return d.getUTCFullYear() === year
    && d.getUTCMonth() === month - 1
    && d.getUTCDate() === day;
}

function toIsoDate(year, month, day) {
  return `${String(year).padStart(4, "0")}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
}

export function normalizeDateTextValue(raw) {
  const value = String(raw || "").trim();
  if (!value) return null;

  let match = value.match(/^(\d{4})-(\d{1,2})-(\d{1,2})$/);
  if (match) {
    const [, y, m, d] = match.map(Number);
    return validDateParts(y, m, d) ? toIsoDate(y, m, d) : null;
  }

  match = value.match(/^(\d{1,2})\/(\d{1,2})\/(\d{4})$/);
  if (match) {
    const month = Number(match[1]);
    const day = Number(match[2]);
    const year = Number(match[3]);
    return validDateParts(year, month, day) ? toIsoDate(year, month, day) : null;
  }

  return null;
}

export function makeDateTextInput(currentVal, onChange, attrs = {}) {
  const className = attrs.class ? `${attrs.class} gold-date-input` : "gold-date-input";
  const inp = el("input", {
    ...attrs,
    type: "text",
    class: className,
    value: currentVal ?? "",
    placeholder: attrs.placeholder ?? "YYYY-MM-DD",
    inputmode: "numeric",
    autocomplete: "off",
    spellcheck: "false",
    title: attrs.title ?? "Enter YYYY-MM-DD. MM/DD/YYYY also works.",
  });
  let lastCommitted = currentVal ?? null;

  function commitIfReady({ force = false } = {}) {
    const raw = inp.value.trim();
    if (!raw) {
      inp.classList.remove("gold-date-invalid");
      if (lastCommitted !== null) {
        lastCommitted = null;
        onChange(null);
      }
      return;
    }

    const normalized = normalizeDateTextValue(raw);
    if (normalized) {
      inp.classList.remove("gold-date-invalid");
      if (force && inp.value !== normalized) inp.value = normalized;
      if (normalized !== lastCommitted) {
        lastCommitted = normalized;
        onChange(normalized);
      }
      return;
    }

    if (force) inp.classList.add("gold-date-invalid");
  }

  inp.addEventListener("input", () => commitIfReady());
  inp.addEventListener("change", () => commitIfReady({ force: true }));
  inp.addEventListener("blur", () => commitIfReady({ force: true }));
  return inp;
}

export function showToast(msg, type = "ok") {
  const t = document.getElementById("save-toast");
  t.textContent = msg;
  t.className = `show ${type}`;
  setTimeout(() => t.className = "", 3200);
}

export function updateSaveButtonLabel() {
  const label = S.appMode === "items" ? "Save item" : "Save";
  document.getElementById("btn-save").textContent = S.isDirty ? `${label}*` : label;
}

export function markDirty() {
  S.isDirty = true;
  updateSaveButtonLabel();
}

export function markClean() {
  S.isDirty = false;
  updateSaveButtonLabel();
}

export function ensureExpectedOutcome(data) {
  if (!data.expected_outcome) data.expected_outcome = {};
  if (!data.expected_outcome.fields) data.expected_outcome.fields = {};
  if (!data.expected_outcome.repeat_groups) data.expected_outcome.repeat_groups = {};
}

export function ensureDifficultyProfile(data) {
  if (!data.difficulty_profile) data.difficulty_profile = {};
  if (!data.difficulty_profile.dimensions) data.difficulty_profile.dimensions = {};
  if (!data.difficulty_profile.dimension_notes) data.difficulty_profile.dimension_notes = {};
  if (!data.difficulty_profile.targeted_failure_modes) data.difficulty_profile.targeted_failure_modes = [];
  if (data.difficulty_profile.failure_explanation == null) data.difficulty_profile.failure_explanation = "";
}

export function cleanTargetedFailureModes(data = {}) {
  ensureDifficultyProfile(data);
  const profileModes = data.difficulty_profile.targeted_failure_modes || [];
  const topLevelModes = data.targeted_failure_mode || [];
  return Array.from(new Set([...profileModes, ...topLevelModes]
    .map(v => String(v || "").trim())
    .filter(v => v && v !== "TODO")));
}

export function readyItemRequiresFailureContract(data = {}) {
  const modes = cleanTargetedFailureModes(data);
  const roleTokens = [data.contrast_role, data.item_role]
    .map(v => String(v || "").trim().toLowerCase())
    .filter(Boolean);
  const difficultyTokens = [data.difficulty_tier, data.materialization?.difficulty]
    .map(v => String(v || "").trim().toLowerCase())
    .filter(Boolean);
  const probeRoles = new Set(["diagnostic_probe", "probe", "failure_revealing"]);
  const controlRoles = new Set(["anchor", "control", "baseline"]);
  if (roleTokens.some(role => probeRoles.has(role))) return true;
  if (modes.length) return true;
  if (roleTokens.some(role => controlRoles.has(role))) return false;
  if (difficultyTokens.some(level => level === "anchor" || level === "easy")) return false;
  return true;
}

export function legacyHistoryConditionFromItem(data = {}) {
  const evidence = data.evidence || {};
  const conflictPresent = !!evidence.conflict_present;
  const historyRequired = !!evidence.history_required;
  const supportDistance = Number(evidence.support_distance || 0);
  const visibleHistory = Array.isArray(data.visible_history) && data.visible_history.length > 0;
  const historyRef = String(data.history_ref || "");
  const hasReferencedHistory = !!historyRef && !historyRef.endsWith("/empty") && !historyRef.endsWith("/H1_none");

  if (conflictPresent) return "H4";
  if (historyRequired || visibleHistory || hasReferencedHistory) {
    return supportDistance >= 2 ? "H3" : "H2";
  }
  return "H1";
}

export function slugify(text) {
  return String(text || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 48) || "new_context";
}

export function defaultContextRef(kind, questionnaire, conditionCode) {
  const suffix = kind === "state"
    ? `${conditionCode || "S1"}_custom_${kind}`
    : `${conditionCode || "H1"}_custom_${kind}`;
  return `${questionnaire}/${suffix}`;
}

export function baseContextDraft(kind, questionnaire, conditionCode) {
  if (kind === "state") {
    return {
      condition_code: conditionCode || "S1",
      questionnaire,
      description: "",
      questionnaire_answers: (conditionCode || "S1") === "S1" ? null : {},
    };
  }
  return {
    condition_code: conditionCode || "H1",
    questionnaire,
    description: "",
    turns: [],
  };
}

export function getContextSummaries() {
  return [...S.allContexts.state, ...S.allContexts.history].sort((a, b) => a.ref.localeCompare(b.ref));
}

export function currentContextIndex() {
  const contexts = getContextSummaries();
  return contexts.findIndex(c => c.kind === S.currentContextKind && c.ref === S.currentContextRef);
}

export function currentRefFileName() {
  if (!S.currentContextRef) return "";
  const parts = S.currentContextRef.split("/");
  return parts.slice(1).join("/");
}

export function setContextRefFileName(name) {
  const questionnaire = S.currentContextData?.questionnaire || S.questionnaireNames[0] || "";
  S.currentContextRef = `${questionnaire}/${name.replace(/^\/+|\/+$/g, "")}`;
}

export function setModeTabs() {
  document.getElementById("mode-items")?.classList.toggle("active", S.appMode === "items");
  document.getElementById("mode-eval")?.classList.toggle("active", S.appMode === "eval");
  document.getElementById("mode-howto")?.classList.toggle("active", S.appMode === "howto");
}

export function updateEmptyState() {
  const title = document.getElementById("no-selection-title");
  const copy = document.getElementById("no-selection-copy");
  const msgs = {
    items: ["Select an item to begin", "Pick any item from the left panel to open its authoring form."],
    eval: ["Evaluation Results", "View benchmark run summaries and diagnostics."],
    howto: ["How-To", "Add questionnaires and build custom datasets for ConFormBench."],
  };
  const [t, c] = msgs[S.appMode] || msgs.items;
  title.textContent = t;
  copy.textContent = c;
}

export function makeSection(id, number, title, subtitle, bodyFn, startsCollapsed = false) {
  const section = el("div", { class: `section${startsCollapsed ? " collapsed" : ""}`, id });
  const header = el("div", { class: "section-header", onclick: () => section.classList.toggle("collapsed") },
    el("div", { class: "section-number" }, String(number)),
    el("div", { class: "section-title" }, el("h2", {}, title), el("p", {}, subtitle)),
    el("span", { class: "section-toggle" }, "▾"),
  );
  const body = el("div", { class: "section-body" });
  bodyFn(body);
  section.append(header, body);
  return section;
}

export function setEditorHeader({ idText, titleText, saveLabel }) {
  document.getElementById("editor-item-id").textContent = idText;
  document.getElementById("editor-item-title").textContent = titleText;
  document.getElementById("btn-save").textContent = saveLabel;
}

export function setNavButtons() {
  const prev = document.getElementById("btn-prev");
  const next = document.getElementById("btn-next");
  if (S.appMode === "items") {
    prev.style.display = "inline-flex";
    next.style.display = "inline-flex";
    const idx = S.allItems.findIndex(i => i.item_id === S.currentId);
    prev.disabled = idx <= 0;
    next.disabled = idx >= S.allItems.length - 1;
  } else {
    prev.style.display = "none";
    next.style.display = "none";
  }
}
