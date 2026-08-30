// ── Context editor section renderers ──────────────────────────────────────────
import { S } from "./state.js";
import { apiGet, getQFieldMeta, getQTree } from "./api.js";
import {
  el, makeDateTextInput, markDirty, showToast, slugify, currentRefFileName, setContextRefFileName,
  defaultContextRef, baseContextDraft,
} from "./dom-utils.js";
import { STATE_CONDITIONS, HISTORY_CONDITIONS, CONTEXT_KIND_LABELS } from "./constants.js";
import { updateChecklist } from "./checklist.js";
import { renderStateBody, renderHistoryBody } from "./item-sections.js";
// Lazy import – used only inside async change handlers
import { renderActiveEditor } from "./editor.js";

// ── Context identity ─────────────────────────────────────────────────────────

export function renderContextIdentitySection(container) {
  const d = S.currentContextData;
  const kindSel = el("select", { class: "code-select" });
  ["state", "history"].forEach(kind => kindSel.append(el("option", { value: kind, ...(kind === S.currentContextKind ? { selected: "" } : {}) }, CONTEXT_KIND_LABELS[kind])));
  kindSel.addEventListener("change", async () => {
    const questionnaire = d.questionnaire || S.questionnaireNames[0] || "";
    const conditionCode = kindSel.value === "state" ? "S1" : "H1";
    S.currentContextKind = kindSel.value;
    S.currentContextData = baseContextDraft(S.currentContextKind, questionnaire, conditionCode);
    S.currentContextRef = defaultContextRef(S.currentContextKind, questionnaire, conditionCode);
    markDirty();
    await renderActiveEditor();
  });

  const qSel = el("select", { class: "code-select" });
  S.questionnaireNames.forEach(q => qSel.append(el("option", { value: q, ...(q === d.questionnaire ? { selected: "" } : {}) }, q)));
  qSel.addEventListener("change", async () => {
    d.questionnaire = qSel.value;
    S.currentContextRef = `${d.questionnaire}/${currentRefFileName() || slugify(`${d.condition_code}_${S.currentContextKind}`)}`;
    markDirty();
    await renderActiveEditor();
  });

  const condSel = el("select", { class: "code-select" });
  const conditionOptions = S.currentContextKind === "state" ? STATE_CONDITIONS : HISTORY_CONDITIONS;
  conditionOptions.forEach(c => condSel.append(el("option", { value: c.value, ...(c.value === d.condition_code ? { selected: "" } : {}) }, `${c.value} — ${c.label.split("–")[1]?.trim() || c.label}`)));
  condSel.addEventListener("change", async () => {
    d.condition_code = condSel.value;
    if (!currentRefFileName() || currentRefFileName().startsWith("S") || currentRefFileName().startsWith("H")) {
      setContextRefFileName(slugify(`${condSel.value}_${S.currentContextKind}`));
    }
    if (S.currentContextKind === "state" && condSel.value === "S1" && !d.questionnaire_answers) d.questionnaire_answers = null;
    if (S.currentContextKind === "history" && condSel.value === "H1" && !Array.isArray(d.turns)) d.turns = [];
    markDirty();
    await renderActiveEditor();
  });

  const fileNameInput = el("input", { type: "text", value: currentRefFileName(), placeholder: S.currentContextKind === "state" ? "S2_partial_correct_variant" : "H2_recent_support_variant" });
  fileNameInput.addEventListener("input", () => {
    const cleaned = fileNameInput.value.replace(/[^a-zA-Z0-9_./-]/g, "_");
    setContextRefFileName(cleaned);
    markDirty();
    refPreview.textContent = S.currentContextRef;
  });

  const descTa = el("textarea", { placeholder: "Short description for interns" });
  descTa.value = d.description || "";
  descTa.addEventListener("input", () => { d.description = descTa.value; markDirty(); });

  const refPreview = el("div", { class: "context-ref-preview" }, S.currentContextRef || "—");

  // ── Scenario picker ──
  const scenarioSel = el("select", { class: "code-select" });
  scenarioSel.append(el("option", { value: "" }, "— no scenario —"));
  for (const sc of S.allScenarios) {
    scenarioSel.append(el("option", { value: sc.scenario_id, ...(sc.scenario_id === (d.scenario || "") ? { selected: "" } : {}) }, sc.scenario_id.replace(/_/g, " ")));
  }
  scenarioSel.addEventListener("change", () => { d.scenario = scenarioSel.value || undefined; markDirty(); });

  container.append(
    el("div", { class: "field-row" },
      el("div", { class: "field-group" }, el("label", {}, "Context type"), kindSel),
      el("div", { class: "field-group" }, el("label", {}, "Questionnaire"), qSel),
      el("div", { class: "field-group" }, el("label", {}, "Condition code"), condSel),
    ),
    el("div", { class: "field-row" },
      el("div", { class: "field-group" }, el("label", {}, "File name inside the questionnaire folder"), fileNameInput, el("p", { class: "help-text" }, "This becomes the reusable ref interns can pick from items.")),
      el("div", { class: "field-group" }, el("label", {}, "Resulting ref"), refPreview, el("p", { class: "help-text" }, "The item editor will reference this path.")),
    ),
    el("div", { class: "field-row" },
      el("div", { class: "field-group" }, el("label", {}, "Scenario"), scenarioSel, el("p", { class: "help-text" }, "Link to a big-picture scenario.")),
      el("div", { class: "field-group" }, el("label", {}, "Description"), descTa, el("p", { class: "help-text" }, "Explain when this reusable context should be used.")),
    ),
  );
}

// ── Datalist helper ──────────────────────────────────────────────────────────

export function makeOptionDatalist(id, options) {
  const list = el("datalist", { id });
  options.forEach(option => list.append(el("option", { value: option })));
  return list;
}

// ── State context builder ────────────────────────────────────────────────────

export async function renderStateContextSection(container) {
  const d = S.currentContextData;
  if (d.questionnaire_answers === undefined) d.questionnaire_answers = null;
  const fieldsMeta = await getQFieldMeta(d.questionnaire);
  const fieldMetaMap = Object.fromEntries(fieldsMeta.map(f => [f.id, f]));
  const qTree = await getQTree(d.questionnaire);

  // Condition-specific guidance
  const hints = {
    S1: { text: "S1 = Empty form. No fields should be pre-filled. The toggle below should stay checked.", color: "#10b981" },
    S2: { text: "S2 = Partial correct. Add some fields with correct values. Leave other fields empty.", color: "#0ea5e9" },
    S3: { text: "S3 = Partial incorrect. Add some fields, but make at least one value intentionally wrong.", color: "#f97316" },
    S4: { text: "S4 = Inconsistent. Add fields whose values contradict each other (e.g. date says Tuesday but weekday says Monday).", color: "#dc2626" },
  };
  const hint = hints[d.condition_code] || hints.S1;
  const hintBanner = el("div", { class: "ctx-condition-hint", style: `border-left:3px solid ${hint.color};padding:8px 12px;background:#f8fafc;border-radius:0 6px 6px 0;margin-top:12px;font-size:12px;color:var(--text-2);line-height:1.5;` }, hint.text);

  const emptyToggle = el("input", { type: "checkbox", checked: d.questionnaire_answers === null });

  emptyToggle.addEventListener("change", () => {
    d.questionnaire_answers = emptyToggle.checked ? null : {};
    markDirty(); updateChecklist();
    rebuildFormView();
    updateFilledCount();
  });

  // Auto-check S1 empty
  if (d.condition_code === "S1" && d.questionnaire_answers !== null && Object.keys(d.questionnaire_answers || {}).length === 0) {
    d.questionnaire_answers = null;
    emptyToggle.checked = true;
  }

  // ── Mode toggle: Form | JSON ──
  const headingRow = el("div", { style: "display:flex;align-items:center;gap:10px;margin:12px 0 4px;" });
  const modeToggle = el("div", { class: "gold-mode-toggle" });
  const btnForm = el("button", { class: "gold-mode-btn active", "data-mode": "form" }, "Form");
  const btnJson = el("button", { class: "gold-mode-btn", "data-mode": "json" }, "JSON");
  modeToggle.append(btnForm, btnJson);
  const filledBadge = el("span", { class: "ctx-filled-badge", style: "font-size:11px;color:var(--help);font-style:italic;" }, "");
  headingRow.append(modeToggle, filledBadge);

  // Form view
  const formWrap = el("div", { class: "gold-form ctx-state-form" });
  // JSON view
  const jsonWrap = el("div", { class: "gold-json-wrap", style: "display:none;" });
  const jsonError = el("div", { class: "gold-json-error" });
  const jsonArea = el("textarea", { class: "gold-json-textarea", spellcheck: "false", placeholder: "Paste or edit questionnaire_answers JSON here…" });
  jsonWrap.append(jsonError, jsonArea);

  let currentMode = "form";

  function updateFilledCount() {
    if (d.questionnaire_answers === null) {
      filledBadge.textContent = "(empty state)";
      return;
    }
    const ans = d.questionnaire_answers || {};
    const filled = Object.values(ans).filter(v => v != null && v !== "").length;
    const total = fieldsMeta.length;
    filledBadge.textContent = `(${filled} of ${total} fields filled)`;
  }

  // Collect structural fields (gates / repeat group counts)
  const structuralFields = new Set();
  function collectStructural(questions) {
    for (const q of questions) {
      const st = q.structure_type || "regular";
      if (st === "gate" && q.gate?.gate_on) structuralFields.add(q.gate.gate_on);
      if (st === "repeat_group" && q.repeat?.from_slot) structuralFields.add(q.repeat.from_slot);
      if (st === "group" || st === "gate" || st === "repeat_group") {
        collectStructural(q.fields || q.questions || []);
      }
    }
  }
  collectStructural(qTree);

  /** Read a value from questionnaire_answers. */
  function ansGet(fieldId) {
    return d.questionnaire_answers?.[fieldId];
  }
  /** Write a value to questionnaire_answers. */
  function ansSet(fieldId, val) {
    if (!d.questionnaire_answers) d.questionnaire_answers = {};
    d.questionnaire_answers[fieldId] = val;
  }
  /** Read from repeat group instance. */
  function ansRepeatGet(groupId, idx, fieldId) {
    const arr = d.questionnaire_answers?.[groupId];
    if (!Array.isArray(arr) || idx >= arr.length) return undefined;
    return arr[idx]?.[fieldId];
  }
  /** Write to repeat group instance. */
  function ansRepeatSet(groupId, idx, fieldId, val) {
    if (!d.questionnaire_answers) d.questionnaire_answers = {};
    if (!Array.isArray(d.questionnaire_answers[groupId])) d.questionnaire_answers[groupId] = [];
    while (d.questionnaire_answers[groupId].length <= idx) d.questionnaire_answers[groupId].push({});
    d.questionnaire_answers[groupId][idx][fieldId] = val;
  }

  /** Check if a gate is open. */
  function isGateOpen(gate) {
    const currentVal = ansGet(gate.gate_on);
    if (currentVal === undefined || currentVal === null) return false;
    for (const wv of gate.when_values) {
      if (wv === true && (currentVal === true || currentVal === "true")) return true;
      if (wv === false && (currentVal === false || currentVal === "false")) return true;
      if (String(wv) === String(currentVal)) return true;
    }
    return false;
  }

  /** Get repeat count. */
  function getRepeatCount(repeatConfig, groupId) {
    if (repeatConfig.mode === "fixed") return repeatConfig.count || 0;
    const val = ansGet(repeatConfig.from_slot);
    let n = parseInt(val, 10);
    if ((isNaN(n) || n <= 0) && groupId) {
      const arr = d.questionnaire_answers?.[groupId];
      if (Array.isArray(arr) && arr.length > 0) {
        n = arr.length;
        ansSet(repeatConfig.from_slot, n);
      }
    }
    return isNaN(n) || n < 0 ? 0 : Math.min(n, 20);
  }

  function normalizeFieldType(type) {
    return String(type || "text").trim().toLowerCase().replace(/[-\s]+/g, "_");
  }

  /** Create typed input for a field. */
  function makeFormInput(meta, currentVal, onChange) {
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
        const cb = el("input", { type: "checkbox", "data-opt": String(opt), ...(curArr.some(v => String(v) === String(opt)) ? { checked: "" } : {}) });
        cb.addEventListener("change", collect);
        lbl.append(cb, " " + String(opt));
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
        const headerRow = el("div", { class: "gold-table-header" });
        for (const col of meta.columns) {
          headerRow.appendChild(el("span", { class: "gold-table-col-header", title: col.gold_standard || "" }, col.question_text || col.id));
        }
        headerRow.appendChild(el("span", { class: "gold-table-col-action" }, ""));
        wrap.appendChild(headerRow);

        rows.forEach((row, rowIdx) => {
          const rowEl = el("div", { class: "gold-table-row" });
          for (const col of meta.columns) {
            const cell = el("div", { class: "gold-table-cell" });
            const colVal = row[col.id] ?? "";
            const colType = normalizeFieldType(col.type);
            const colOptions = Array.isArray(col.options) ? col.options : [];
            if (colType === "single_choice" && colOptions.length) {
              const sel = el("select", { class: "gold-form-input gold-table-input" });
              sel.append(el("option", { value: "" }, "\u2014"));
              for (const opt of colOptions) {
                sel.append(el("option", { value: String(opt), ...(String(colVal) === String(opt) ? { selected: "" } : {}) }, String(opt)));
              }
              if (colVal !== "" && !colOptions.some(opt => String(opt) === String(colVal))) {
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
          const rmBtn = el("button", { class: "gold-table-rm", title: "Remove row", type: "button" }, "\u00d7");
          rmBtn.addEventListener("click", () => { rows.splice(rowIdx, 1); renderTableRows(); fireChange(); });
          rowEl.appendChild(rmBtn);
          wrap.appendChild(rowEl);
        });

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
    if (inputType === "time") inp.addEventListener("change", fire);
    return inp;
  }

  /** Render a single field row. */
  function renderFieldRow(q, currentVal, onChangeCallback) {
    const meta = fieldMetaMap[q.id] || {
      id: q.id,
      label: q.question_text || q.label || q.id,
      type: q.type || "text",
      options: q.options || [],
      other_specify: !!q.other_specify,
      columns: q.columns || [],
    };
    const isFilled = currentVal != null && currentVal !== "";

    const row = el("div", { class: `gold-form-field${isFilled ? " ctx-field-filled" : ""}` });

    const labelCol = el("div", { class: "gold-form-label-col" });
    const fieldLabel = meta.label && meta.label !== q.id
      ? el("label", { class: "gold-form-label", title: q.id }, meta.label)
      : el("label", { class: "gold-form-label" }, q.id);
    labelCol.appendChild(fieldLabel);
    labelCol.appendChild(el("div", { class: "gold-form-badges" },
      el("span", { class: "gold-type-badge" }, meta.type || "text"),
      el("span", { style: "font-size:10px;color:var(--text-3);font-family:var(--mono);" }, q.id),
    ));

    const inputCol = el("div", { class: "gold-form-input-col" });
    const input = makeFormInput(meta, currentVal, (newVal) => {
      onChangeCallback(newVal);
      const filled = newVal != null && newVal !== "";
      row.classList.toggle("ctx-field-filled", filled);
      updateFilledCount();
      if (structuralFields.has(q.id)) rebuildFormView();
    });
    inputCol.appendChild(input);

    row.append(labelCol, inputCol);
    return row;
  }

  /** Walk the tree and render all fields into formWrap. */
  function rebuildFormView() {
    formWrap.innerHTML = "";
    if (d.questionnaire_answers === null) {
      formWrap.appendChild(el("p", { class: "context-empty", style: "padding:16px;" }, "\u2713 This state is intentionally empty \u2014 no fields pre-filled."));
      return;
    }

    if (!qTree.length) {
      formWrap.appendChild(el("p", { class: "context-empty", style: "padding:16px;" }, "No questionnaire tree available. Switch to JSON mode to edit directly."));
      return;
    }

    function walkQuestions(questions, parentLabel) {
      for (const q of questions) {
        const st = q.structure_type || "regular";

        if (st === "regular") {
          const currentVal = ansGet(q.id);
          formWrap.appendChild(renderFieldRow(q, currentVal, (newVal) => {
            ansSet(q.id, newVal);
            markDirty(); updateChecklist();
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
            const hintEl = el("div", { class: "gold-gate-hint" },
              `Set "${gate.gate_on}" to ${gate.when_values.map(v => JSON.stringify(v)).join(" or ")} to reveal these fields.`);
            formWrap.appendChild(hintEl);
            continue;
          }

        } else if (st === "repeat_group") {
          const repeat = q.repeat;
          if (!repeat) continue;
          const groupId = q.id;
          const count = getRepeatCount(repeat, groupId);
          const childFields = q.fields || q.questions || [];

          // Ensure array
          if (!Array.isArray(d.questionnaire_answers[groupId])) {
            d.questionnaire_answers[groupId] = [];
          }
          const arr = d.questionnaire_answers[groupId];
          while (arr.length < count) arr.push({});

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

            for (const cf of childFields) {
              if ((cf.structure_type || "regular") !== "regular") continue;
              if (arr[i] && !(cf.id in arr[i])) arr[i][cf.id] = null;
              const currentVal = ansRepeatGet(groupId, i, cf.id);
              const capturedIdx = i;
              instanceEl.appendChild(renderFieldRow(cf, currentVal, (newVal) => {
                ansRepeatSet(groupId, capturedIdx, cf.id, newVal);
                markDirty(); updateChecklist();
              }));
            }
            formWrap.appendChild(instanceEl);
          }
        }
      }
    }

    walkQuestions(qTree, "");
  }

  /** Build ordered JSON from questionnaire tree. */
  function buildOrderedJson() {
    const ans = d.questionnaire_answers;
    if (ans === null) return null;
    const result = {};
    function walk(questions) {
      for (const q of questions) {
        const st = q.structure_type || "regular";
        if (st === "regular") {
          if (q.id in ans && ans[q.id] != null && ans[q.id] !== "") result[q.id] = ans[q.id];
        } else if (st === "group") {
          walk(q.fields || q.questions || []);
        } else if (st === "gate") {
          walk(q.fields || q.questions || []);
        } else if (st === "repeat_group") {
          if (q.id in ans && Array.isArray(ans[q.id])) {
            const filtered = ans[q.id].filter(inst =>
              Object.values(inst).some(v => v != null && v !== "")
            );
            if (filtered.length > 0) result[q.id] = filtered;
          }
        }
      }
    }
    walk(qTree);
    // Extra keys not in tree
    for (const k of Object.keys(ans)) {
      if (!(k in result) && ans[k] != null && ans[k] !== "") {
        result[k] = ans[k];
      }
    }
    return Object.keys(result).length ? result : {};
  }

  function setMode(mode) {
    currentMode = mode;
    btnForm.classList.toggle("active", mode === "form");
    btnJson.classList.toggle("active", mode === "json");
    formWrap.style.display = mode === "form" ? "" : "none";
    jsonWrap.style.display = mode === "json" ? "" : "none";
    if (mode === "json") {
      const ordered = buildOrderedJson();
      jsonArea.value = ordered === null ? "null" : JSON.stringify(ordered, null, 2);
      jsonError.textContent = "";
    } else {
      applyJsonToAnswers();
      rebuildFormView();
      updateFilledCount();
    }
  }

  function applyJsonToAnswers() {
    try {
      const parsed = JSON.parse(jsonArea.value);
      if (parsed === null) {
        d.questionnaire_answers = null;
        emptyToggle.checked = true;
        markDirty();
        jsonError.textContent = "";
        return true;
      }
      if (typeof parsed !== "object" || Array.isArray(parsed)) {
        jsonError.textContent = "JSON must be an object or null.";
        return false;
      }
      d.questionnaire_answers = parsed;
      emptyToggle.checked = false;
      markDirty();
      jsonError.textContent = "";
      return true;
    } catch (e) {
      jsonError.textContent = "Invalid JSON: " + e.message;
      return false;
    }
  }

  jsonArea.addEventListener("input", () => {
    applyJsonToAnswers();
    updateChecklist();
    updateFilledCount();
  });

  btnForm.addEventListener("click", () => setMode("form"));
  btnJson.addEventListener("click", () => setMode("json"));

  // ── Clear all button ──
  const clearBtn = el("button", { class: "btn sm danger", type: "button", style: "margin-left:auto;" }, "Clear all values");
  clearBtn.addEventListener("click", () => {
    if (!confirm("Clear all field values in this state?")) return;
    if (d.questionnaire_answers && typeof d.questionnaire_answers === "object") {
      for (const key of Object.keys(d.questionnaire_answers)) {
        if (Array.isArray(d.questionnaire_answers[key])) {
          d.questionnaire_answers[key] = d.questionnaire_answers[key].map(inst => {
            if (typeof inst === "object" && inst !== null) {
              return Object.fromEntries(Object.keys(inst).map(k => [k, null]));
            }
            return inst;
          });
        } else {
          d.questionnaire_answers[key] = null;
        }
      }
    }
    markDirty(); updateChecklist();
    rebuildFormView();
    updateFilledCount();
  });

  const actionBar = el("div", { style: "display:flex;align-items:center;gap:8px;margin-top:8px;" });
  actionBar.appendChild(clearBtn);

  container.append(
    hintBanner,
    el("div", { class: "context-toggle", style: "margin-top:12px;" }, emptyToggle, el("span", {}, "Use an empty form state (no pre-filled answers)")),
    headingRow,
    el("p", { style: "font-size:12px;color:var(--help);margin-bottom:6px;line-height:1.5;" },
      "All questionnaire fields are shown below. Fill only the fields that should be pre-populated in this state context."),
    formWrap,
    jsonWrap,
    actionBar,
  );

  rebuildFormView();
  updateFilledCount();
}

// ── History context builder ──────────────────────────────────────────────────

export function renderHistoryContextSection(container) {
  const d = S.currentContextData;
  if (!Array.isArray(d.turns)) d.turns = [];

  // Condition-specific guidance
  const guidance = {
    H1: { text: "H1 = No prior turns. This is the first message in the session. Leave the turn list empty.", color: "#10b981" },
    H2: { text: "H2 = Recent supporting. Add 1–2 recent turns that directly support or set up the current item's utterance.", color: "#0ea5e9" },
    H3: { text: "H3 = Distant supporting. Add earlier evidence turns, then intervening turns on unrelated topics, so relevant info is buried in the history.", color: "#8b5cf6" },
    H4: { text: "H4 = Conflicting. Add a prior turn where the user stated something that contradicts what the current item's utterance will claim.", color: "#f97316" },
  };
  const guide = guidance[d.condition_code] || guidance.H1;
  const guideBanner = el("div", { class: "ctx-condition-hint", style: `border-left:3px solid ${guide.color};padding:8px 12px;background:#f8fafc;border-radius:0 6px 6px 0;margin-top:12px;font-size:12px;color:var(--text-2);line-height:1.5;` }, guide.text);

  // Template loading
  const templateBtnWrap = el("div", { style: "margin-top:10px;display:flex;gap:8px;align-items:center;" });
  if (d.condition_code !== "H1" && d.questionnaire) {
    const templateName = d.condition_code === "H2"
      ? "H2_recent_support"
      : d.condition_code === "H3"
        ? "H3_distant_support"
        : "H4_conflicting";
    const templateRef = `${d.questionnaire}/${templateName}`;
    const loadBtn = el("button", { class: "btn sm", type: "button" }, "📋 Load starter turns from " + templateRef);
    loadBtn.addEventListener("click", async () => {
      try {
        const tmpl = await apiGet(`/api/contexts/history/${templateRef}`);
        if (tmpl.turns && tmpl.turns.length) {
          if (d.turns.length && !confirm("Replace existing turns with template?")) return;
          d.turns = tmpl.turns.map(t => ({ speaker: t.speaker, text: t.text }));
          markDirty(); updateChecklist();
          renderTurns();
          showToast(`Loaded ${d.turns.length} template turns — edit them for your scenario`, "ok");
        } else {
          showToast("Template has no turns", "err");
        }
      } catch {
        showToast(`Could not load template ${templateRef}`, "err");
      }
    });
    templateBtnWrap.append(loadBtn, el("span", { style: "font-size:11px;color:var(--help);" }, "Loads reusable starter turns if a matching template exists"));
  }

  const list = el("div", { class: "context-builder" });

  function renderTurns() {
    list.innerHTML = "";
    if (d.condition_code === "H1" && !d.turns.length) {
      list.appendChild(el("p", { class: "context-empty" }, "✓ H1 = no prior turns needed. The turn list should stay empty."));
      return;
    }
    if (!d.turns.length) list.appendChild(el("p", { class: "context-empty" }, "No prior turns yet. Add the turns the agent should already have seen."));
    d.turns.forEach((turn, idx) => {
      const row = el("div", { class: "context-row" });
      const speakerSel = el("select", {});
      ["user", "assistant", "system"].forEach(s => speakerSel.append(el("option", { value: s, ...(s === turn.speaker ? { selected: "" } : {}) }, s)));
      speakerSel.addEventListener("change", () => { turn.speaker = speakerSel.value; markDirty(); updateChecklist(); });
      const textTa = el("textarea", { placeholder: "Turn text", style: "min-height:84px;" });
      textTa.value = turn.text || "";
      textTa.addEventListener("input", () => { turn.text = textTa.value; markDirty(); updateChecklist(); });
      const controls = el("div", { class: "context-turn-controls" },
        el("button", { class: "btn sm", type: "button", onclick: () => { if (idx > 0) { [d.turns[idx - 1], d.turns[idx]] = [d.turns[idx], d.turns[idx - 1]]; markDirty(); renderTurns(); } } }, "↑"),
        el("button", { class: "btn sm", type: "button", onclick: () => { if (idx < d.turns.length - 1) { [d.turns[idx + 1], d.turns[idx]] = [d.turns[idx], d.turns[idx + 1]]; markDirty(); renderTurns(); } } }, "↓"),
        el("button", { class: "btn sm", type: "button", onclick: () => { d.turns.splice(idx + 1, 0, { ...turn }); markDirty(); updateChecklist(); renderTurns(); } }, "Duplicate"),
        el("button", { class: "btn sm danger", type: "button", onclick: () => { d.turns.splice(idx, 1); markDirty(); updateChecklist(); renderTurns(); } }, "Remove"),
      );
      row.append(el("div", { class: "context-grid history" }, el("div", {}, el("label", {}, `Turn ${idx + 1} speaker`), speakerSel), el("div", {}, el("label", {}, "Turn text"), textTa)), controls);
      list.appendChild(row);
    });
  }

  const addBtn = el("button", { class: "btn sm", type: "button" }, "+ Add turn");
  addBtn.addEventListener("click", () => { d.turns.push({ speaker: d.turns.length % 2 === 0 ? "user" : "assistant", text: "" }); markDirty(); updateChecklist(); renderTurns(); });
  container.append(guideBanner, templateBtnWrap, list, addBtn);
  renderTurns();
}

// ── Context preview ──────────────────────────────────────────────────────────

export function renderContextPreviewSection(container) {
  const pane = el("div", { class: "context-pane" });
  const body = el("div", { class: "context-pane-body" });
  pane.append(el("div", { class: "context-pane-header" }, `Preview of ${CONTEXT_KIND_LABELS[S.currentContextKind].toLowerCase()}`, el("span", { class: "ref-path" }, S.currentContextRef || "—")), body);
  if (S.currentContextKind === "state") renderStateBody(body, S.currentContextData); else renderHistoryBody(body, S.currentContextData);
  container.append(el("p", { class: "help-text", style: "margin-top:12px;" }, "This is what item authors will see when they reference this shared context."), pane);
}

// ── Context save ─────────────────────────────────────────────────────────────

export function renderContextSaveSection(container) {
  container.append(el("div", { class: "save-bar" },
    el("div", { class: "status-group" }, el("label", {}, "Shared context:"), el("span", { class: "status-badge ready", style: "display:inline-block;" }, S.currentContextKind)),
    el("div", { class: "help-text" }, "Save this shared context, then reference it from an item and run python main.py materialize-pilot."),
  ));
}

// ── Section registry ─────────────────────────────────────────────────────────

export function getContextSections() {
  return [
    { id: "section-context-identity", n: 1, title: "Context identity", sub: "Name and classify this reusable state/history context.", render: body => renderContextIdentitySection(body) },
    { id: "section-context-content", n: 2, title: S.currentContextKind === "state" ? "Form state builder" : "Conversation history builder", sub: S.currentContextKind === "state" ? "Create the pre-filled form state interns will reuse." : "Create the prior turns the agent should already have seen.", render: body => S.currentContextKind === "state" ? renderStateContextSection(body) : renderHistoryContextSection(body) },
    { id: "section-context-preview", n: 3, title: "Preview", sub: "What item authors will see when they reference this shared context.", render: body => renderContextPreviewSection(body) },
    { id: "section-context-save", n: 4, title: "Save", sub: "Save the shared context, then attach it to an item.", render: body => renderContextSaveSection(body) },
  ];
}
