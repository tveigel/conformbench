// ── Evidence Tagging Popup ────────────────────────────────────────────────────
// Shows a popup where the item author can highlight text spans from the
// conversation history + current utterance as evidence for a value-source tag,
// optionally add a free-text note, and annotate item-level semantic IUs.
//
// Data stored per field in gold_annotations[fieldId]:
//   { value_source: "stated"|"inferred_obvious"|"inferred_non_obvious",
//     evidence: { spans: [ { turn_index, start, end, text } ], note: "" } }
//
// Item-level semantic IUs are stored separately as top-level semantic_ius rows:
//   { field_path, iu_id, schema_iu_id, gold_content, required, evidence_spans }
// ─────────────────────────────────────────────────────────────────────────────

import { S } from "./state.js";
import { apiGet } from "./api.js";
import { el } from "./dom-utils.js";
import { VALUE_SOURCES, formatClaimAnchorLabel, getClaimAnchorMeta } from "./constants.js";

function humanizeEntityRole(entityRole) {
  return entityRole ? entityRole.replace(/_/g, " ") : "";
}

function buildAnchorInstruction(claimAnchor, entityRole) {
  const roleLabel = humanizeEntityRole(entityRole);
  switch (claimAnchor) {
    case "primary_subject":
      return roleLabel
        ? `Tag spans about the main record subject for this field (${roleLabel}) when they support the annotation decision.`
        : "Tag spans about the main record subject for this field when they support the annotation decision.";
    case "event":
      return "Tag spans about the incident or event itself for this field, not just about whoever is speaking.";
    case "repeat_instance":
      return roleLabel
        ? `Tag spans for this specific ${roleLabel} entry, not just ${roleLabel} information in general.`
        : "Tag spans for this specific repeated entry, not just any similar information elsewhere in the conversation.";
    case "role_entity":
      return roleLabel
        ? `Tag spans about the ${roleLabel} for this field, not automatically about the main record subject.`
        : "Tag spans about the role-based entity for this field, not automatically about the main record subject.";
    default:
      return "Tag spans that support this field's actual record target.";
  }
}

/**
 * Open the evidence tagging popup for a specific field.
 *
 * @param {string}   fieldId       - The field key (flat) or "groupId[idx].fieldId" (repeat)
 * @param {string}   fieldLabel    - Human-readable field label
 * @param {string}   currentSource - Current value_source (stated | inferred_obvious | inferred_non_obvious)
 * @param {string}   claimAnchor   - Schema-defined claim anchor for the field
 * @param {string}   entityRole    - Optional role/entity qualifier for the anchor
 * @param {object}   annotation    - Reference to gold_annotations entry for this field
 * @param {Array?}   turnsOverride - Optional explicit turns for non-item-authoring surfaces
 * @param {function} onDone        - Called with (newSource, evidence) when the user confirms
 * @param {object?}  labels        - Optional UI label overrides (for localization)
 * @param {object?}  semanticIu    - Optional item-level semantic IU editor config
 */
export async function openEvidencePopup({ fieldId, fieldLabel, currentSource, claimAnchor = null, entityRole = null, annotation, turnsOverride = null, onDone, labels = null, semanticIu = null }) {
  // ── Gather conversation turns ──
  const turns = Array.isArray(turnsOverride) ? turnsOverride : await gatherConversationTurns();

  // ── Existing evidence (deep-clone) ──
  const existingEvidence = annotation?.evidence
    ? clone(annotation.evidence)
    : { spans: [], note: "" };
  const semanticFieldSpans = semanticIuFieldSpans(semanticIu, turns);

  // Mutable working copy
  let selectedSpans = Array.isArray(existingEvidence.spans) && existingEvidence.spans.length
    ? clone(existingEvidence.spans)
    : semanticFieldSpans;
  let note = existingEvidence.note || "";
  let activeSource = currentSource;
  let semanticRows = buildSemanticIuRows(semanticIu, turns);

  // ── Build overlay ──
  const overlay = el("div", { class: "ev-overlay" });
  const popup = el("div", { class: "ev-popup" });

  // ── Header ──
  const srcInfo = VALUE_SOURCES.find(s => s.value === currentSource);
  const anchorMeta = getClaimAnchorMeta(claimAnchor);
  const anchorLabel = formatClaimAnchorLabel(claimAnchor, entityRole);
  const header = el("div", { class: "ev-header" });
  const headerLeft = el("div", { class: "ev-header-left" });
  headerLeft.append(
    el("h3", { class: "ev-title" }, labels?.title || "Tag Evidence"),
    el("div", { class: "ev-subtitle" },
      el("span", { class: "ev-field-label" }, fieldLabel),
      anchorLabel ? el("span", {
        class: "ev-anchor-badge",
        title: anchorMeta?.desc || "",
      }, anchorLabel) : null,
      el("span", {
        class: "ev-source-badge",
        style: `--badge-color: ${srcInfo?.color || "#6b7280"};`,
      }, srcInfo?.label || currentSource),
    ),
  );
  const closeBtn = el("button", { class: "ev-close-btn", type: "button", title: "Close without saving" }, "\u00d7");
  closeBtn.addEventListener("click", () => overlay.remove());
  header.append(headerLeft, closeBtn);

  // ── Source pills (allow changing source from within popup) ──
  const pillRow = el("div", { class: "ev-pill-row" });
  const pillLabel = el("span", { class: "ev-pill-label" }, labels?.sourceLabel || "Source:");
  pillRow.appendChild(pillLabel);
  for (const vs of VALUE_SOURCES) {
    const vsLabel = labels?.valueSources?.[vs.value]?.label || vs.label;
    const vsDesc = labels?.valueSources?.[vs.value]?.desc || vs.desc;
    const pill = el("button", {
      type: "button",
      class: `ev-source-pill${activeSource === vs.value ? " active" : ""}`,
      "data-source": vs.value,
      title: vsDesc,
      style: activeSource === vs.value ? `--pill-color: ${vs.color};` : "",
    }, vsLabel);
    pill.addEventListener("click", () => {
      activeSource = activeSource === vs.value ? null : vs.value;
      pillRow.querySelectorAll(".ev-source-pill").forEach(p => {
        const pv = p.dataset.source;
        const active = pv === activeSource;
        p.classList.toggle("active", active);
        const src = VALUE_SOURCES.find(s => s.value === pv);
        p.style.setProperty("--pill-color", active && src ? src.color : "");
      });
      updateBadge();
    });
    pillRow.appendChild(pill);
  }
  function updateBadge() {
    const si = VALUE_SOURCES.find(s => s.value === activeSource);
    const badge = header.querySelector(".ev-source-badge");
    if (badge) {
      badge.textContent = (si ? (labels?.valueSources?.[si.value]?.label || si.label) : "None");
      badge.style.setProperty("--badge-color", si?.color || "#6b7280");
    }
  }

  // ── Instructions ──
  const anchorInstruction = buildAnchorInstruction(claimAnchor, entityRole);
  const instructions = el("div", { class: "ev-instructions" },
    el("div", { class: "ev-instructions-text" },
      el("div", {}, labels?.selectInstruction || "Select the shortest span or spans that justify this annotation decision. You can mark multiple spans."),
      el("div", { class: "ev-instructions-anchor" }, anchorInstruction),
    ),
  );

  // ── Conversation body ──
  const convBody = el("div", { class: "ev-conversation" });

  function renderConversation() {
    convBody.innerHTML = "";

    if (turns.length === 0) {
      convBody.appendChild(el("div", { class: "ev-empty" }, "No conversation history available. Only the current utterance is shown below."));
    }

    turns.forEach((turn, turnIdx) => {
      const turnEl = el("div", { class: `ev-turn ev-turn-${turn.speaker}` });
      const speakerLabel = turn.speaker === "state"
        ? "Prior state"
        : turn.speaker === "user"
          ? `\ud83d\udde3\ufe0f ${labels?.currentUtterance ? "Nutzer" : "User"}`
          : `\ud83e\udd16 ${labels?.currentUtterance ? "Assistent" : "Assistant"}`;
      const speakerEl = el("div", { class: "ev-turn-speaker" },
        speakerLabel,
        turn._isCurrent ? el("span", { class: "ev-current-badge" }, labels?.currentUtterance || "current utterance") : null,
      );
      turnEl.appendChild(speakerEl);

      const textEl = el("div", { class: "ev-turn-text", "data-turn-index": String(turnIdx) });
      // Render text with existing highlights
      renderHighlightedText(textEl, turn.text, turnIdx, selectedSpans);
      turnEl.appendChild(textEl);
      convBody.appendChild(turnEl);
    });
  }

  // Set up text selection handler on the conversation body
  convBody.addEventListener("mouseup", () => {
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed) return;

    const range = sel.getRangeAt(0);
    // Find the turn text container
    const turnTextEl = findParentWithClass(range.startContainer, "ev-turn-text");
    if (!turnTextEl || !convBody.contains(turnTextEl)) return;

    const endTurnTextEl = findParentWithClass(range.endContainer, "ev-turn-text");
    if (endTurnTextEl !== turnTextEl) {
      // Cross-turn selection — only take the start turn portion
      sel.removeAllRanges();
      return;
    }

    const turnIdx = parseInt(turnTextEl.dataset.turnIndex, 10);
    const turnText = turns[turnIdx].text;

    // Calculate character offsets within the raw turn text
    const offsets = getSelectionOffsetsInTurn(turnTextEl, turnText);
    if (!offsets || offsets.start === offsets.end) return;

    const selectedText = turnText.slice(offsets.start, offsets.end);

    // Check for overlapping spans and merge
    const newSpan = {
      turn_index: turnIdx,
      start: offsets.start,
      end: offsets.end,
      text: selectedText,
    };

    // Merge overlapping spans for the same turn
    selectedSpans = mergeSpan(selectedSpans, newSpan, turns);
    sel.removeAllRanges();
    renderConversation();
    renderSpanList();
    renderIuSection();
  });

  renderConversation();

  // ── Selected spans list ──
  const spanSection = el("div", { class: "ev-span-section" });
  const spanHeader = el("div", { class: "ev-span-header" },
    el("span", { class: "ev-span-title" }, labels?.markedEvidence || "Marked evidence"),
    el("span", { class: "ev-span-count" }, ""),
  );
  const spanList = el("div", { class: "ev-span-list" });
  spanSection.append(spanHeader, spanList);

  function renderSpanList() {
    spanList.innerHTML = "";
    spanHeader.querySelector(".ev-span-count").textContent = selectedSpans.length > 0
      ? `(${selectedSpans.length} ${labels?.spans || "span"}${selectedSpans.length !== 1 ? (labels?.spans ? "" : "s") : ""})`
      : `(${labels?.none || "none"})`;

    if (selectedSpans.length === 0) {
      spanList.appendChild(el("div", { class: "ev-span-empty" }, labels?.noSpansYet || "No evidence marked yet. Select text above to add spans."));
      return;
    }

    selectedSpans.forEach((sp, i) => {
      const turn = turns[sp.turn_index];
      const speakerLabel = turn
        ? (turn.speaker === "state" ? "Prior state" : (turn.speaker === "user" ? (labels ? "Nutzer" : "User") : (labels ? "Assistent" : "Assistant"))) + (turn._isCurrent ? ` (${labels?.currentUtterance || "current"})` : "")
        : `Turn ${sp.turn_index}`;
      const chip = el("div", { class: "ev-span-chip" });
      chip.append(
        el("span", { class: "ev-span-speaker" }, speakerLabel),
        el("span", { class: "ev-span-text" }, `"${sp.text}"`),
        el("button", {
          class: "ev-span-remove",
          type: "button",
          title: "Remove this span",
          onclick: () => {
            selectedSpans.splice(i, 1);
            renderConversation();
            renderSpanList();
            renderIuSection();
          },
        }, "\u00d7"),
      );
      spanList.appendChild(chip);
    });
  }

  renderSpanList();

  // ── Semantic IU editor ──
  const iuSection = el("div", { class: "ev-iu-section" });

  function renderIuSection() {
    iuSection.innerHTML = "";
    const hasIuConfig = semanticIu && (
      (Array.isArray(semanticIu.schemaIus) && semanticIu.schemaIus.length > 0)
      || semanticRows.length > 0
      || semanticIu.allowCustom !== false
    );
    if (!hasIuConfig) {
      iuSection.style.display = "none";
      return;
    }
    iuSection.style.display = "";
    const includedCount = semanticRows.filter(row => row.included).length;
    const header = el("div", { class: "ev-iu-header" },
      el("div", {},
        el("div", { class: "ev-iu-title" }, "Semantic information units"),
        el("div", { class: "ev-iu-help" }, "Select the IUs expected in the gold value. Each included IU should have gold content and evidence."),
      ),
      el("span", { class: "ev-iu-count" }, includedCount ? `${includedCount} included` : "none included"),
    );
    iuSection.appendChild(header);

    const list = el("div", { class: "ev-iu-list" });
    semanticRows.forEach((row, idx) => {
      const rowEl = el("div", { class: `ev-iu-row${row.included ? " included" : ""}` });
      const check = el("input", { type: "checkbox" });
      check.checked = Boolean(row.included);
      check.addEventListener("change", () => {
        row.included = check.checked;
        renderIuSection();
      });

      const idInput = row.custom
        ? el("input", {
            class: "ev-iu-id-input",
            type: "text",
            value: row.schema_iu_id || "",
            placeholder: "IU id",
          })
        : null;
      if (idInput) {
        idInput.addEventListener("input", () => {
          row.schema_iu_id = idInput.value.trim();
        });
      }

      const meta = el("div", { class: "ev-iu-meta" },
        el("label", { class: "ev-iu-checkline" },
          check,
          el("span", { class: "ev-iu-name" }, row.name || row.schema_iu_id || row.iu_id || "Custom IU"),
        ),
        row.description ? el("div", { class: "ev-iu-desc" }, row.description) : null,
        idInput,
      );

      const contentInput = el("textarea", {
        class: "ev-iu-content",
        placeholder: "Gold content for this IU",
        value: row.gold_content || "",
      });
      contentInput.value = row.gold_content || "";
      contentInput.disabled = !row.included;
      contentInput.addEventListener("input", () => {
        row.gold_content = contentInput.value;
      });

      const useField = el("input", { type: "checkbox" });
      useField.checked = row.use_field_evidence !== false;
      useField.disabled = !row.included;
      useField.addEventListener("change", () => {
        row.use_field_evidence = useField.checked;
        renderIuSection();
      });
      const evidenceMode = el("label", { class: "ev-iu-mode" },
        useField,
        el("span", {}, "Use field evidence"),
      );
      const copyBtn = el("button", {
        type: "button",
        class: "btn xs",
        disabled: !row.included,
        title: "Use the currently marked field spans for this IU only",
      }, "Copy marked spans");
      copyBtn.addEventListener("click", () => {
        row.use_field_evidence = false;
        row.custom_popup_spans = clone(selectedSpans);
        renderIuSection();
      });
      const clearBtn = el("button", {
        type: "button",
        class: "btn xs",
        disabled: !row.included || row.use_field_evidence !== false,
        title: "Clear this IU's custom evidence and fall back to field evidence",
      }, "Reset evidence");
      clearBtn.addEventListener("click", () => {
        row.use_field_evidence = true;
        row.custom_popup_spans = [];
        renderIuSection();
      });
      const evidenceSpans = row.use_field_evidence === false ? row.custom_popup_spans || [] : selectedSpans;
      const evidencePreview = el("div", { class: "ev-iu-evidence-preview" },
        evidenceSpans.length
          ? evidenceSpans.map(sp => `"${sp.text || ""}"`).join("; ")
          : "No spans yet",
      );

      const controls = el("div", { class: "ev-iu-controls" }, evidenceMode, copyBtn, clearBtn);
      if (row.custom) {
        const removeBtn = el("button", { type: "button", class: "btn xs danger" }, "Remove");
        removeBtn.addEventListener("click", () => {
          semanticRows.splice(idx, 1);
          renderIuSection();
        });
        controls.appendChild(removeBtn);
      }

      rowEl.append(meta, contentInput, controls, evidencePreview);
      list.appendChild(rowEl);
    });
    iuSection.appendChild(list);

    const addBtn = el("button", { type: "button", class: "btn sm" }, "Add custom IU");
    addBtn.addEventListener("click", () => {
      semanticRows.push({
        custom: true,
        included: true,
        schema_iu_id: "",
        iu_id: "",
        name: "Custom IU",
        description: "",
        gold_content: "",
        required: true,
        use_field_evidence: true,
        custom_popup_spans: [],
      });
      renderIuSection();
    });
    iuSection.appendChild(el("div", { class: "ev-iu-actions" }, addBtn));
  }

  renderIuSection();

  // ── Note ──
  const noteSection = el("div", { class: "ev-note-section" });
  const noteLabel = el("label", { class: "ev-note-label" }, labels?.noteLabel || "Note (optional)");
  const noteInput = el("textarea", {
    class: "ev-note-input",
    placeholder: labels?.notePlaceholder || "Why is this stated/inferred? Any clarification for future readers...",
    value: note,
  });
  noteInput.value = note;
  noteInput.addEventListener("input", () => { note = noteInput.value; });
  noteSection.append(noteLabel, noteInput);

  // ── Footer buttons ──
  const footer = el("div", { class: "ev-footer" });
  const clearBtn = el("button", { class: "btn sm ev-clear-btn", type: "button" }, labels?.clearAll || "Clear all");
  clearBtn.addEventListener("click", () => {
    selectedSpans = [];
    note = "";
    noteInput.value = "";
    renderConversation();
    renderSpanList();
    renderIuSection();
  });

  const cancelBtn = el("button", { class: "btn sm", type: "button" }, labels?.cancel || "Cancel");
  cancelBtn.addEventListener("click", () => overlay.remove());

  const saveBtn = el("button", { class: "btn sm primary ev-save-btn", type: "button" }, `\u2713 ${labels?.saveEvidence || "Save Evidence"}`);
  saveBtn.addEventListener("click", () => {
    const evidence = {
      spans: selectedSpans,
      note: note.trim(),
    };
    onDone(activeSource, evidence, compileSemanticIus(semanticIu, semanticRows, selectedSpans, turns));
    overlay.remove();
  });

  footer.append(clearBtn, el("div", { style: "flex:1;" }), cancelBtn, saveBtn);

  // ── Assemble popup ──
  popup.append(header, pillRow, instructions, convBody, spanSection, iuSection, noteSection, footer);
  overlay.appendChild(popup);

  // Close on overlay click
  overlay.addEventListener("click", (e) => { if (e.target === overlay) overlay.remove(); });
  // Close on Escape
  function onKey(e) { if (e.key === "Escape") { overlay.remove(); document.removeEventListener("keydown", onKey); } }
  document.addEventListener("keydown", onKey);

  document.body.appendChild(overlay);
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function clone(value) {
  if (value == null) return value;
  return JSON.parse(JSON.stringify(value));
}

function compactId(value) {
  return String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "") || "iu";
}

function spanKey(span) {
  return [
    Number(span?.turn_index ?? -1),
    Number(span?.start ?? -1),
    Number(span?.end ?? -1),
    span?.text || "",
  ].join(":");
}

function normalizePopupSpan(span, turns) {
  let turnIndex = Number(span?.turn_index ?? 0);
  if (!Number.isFinite(turnIndex) || turnIndex < 0 || turnIndex >= turns.length) {
    turnIndex = turns.findIndex(turn => turn._isCurrent) >= 0 ? turns.findIndex(turn => turn._isCurrent) : 0;
  }
  const turn = turns[turnIndex] || {};
  const text = String(turn.text || span?.text || "");
  const start = Math.max(0, Number(span?.start ?? span?.char_start ?? 0));
  const end = Math.max(start, Number(span?.end ?? span?.char_end ?? start));
  return {
    turn_index: turnIndex,
    start,
    end: Math.min(end, text.length || end),
    text: String(span?.text || span?.span_text || text.slice(start, end)),
  };
}

function popupSpansToSemanticSpans(spans, turns) {
  const seen = new Set();
  const out = [];
  for (const raw of spans || []) {
    const span = normalizePopupSpan(raw, turns);
    const turn = turns[span.turn_index] || {};
    const key = spanKey(span);
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({
      source: turn._source || (turn._isCurrent ? "current_utterance" : "visible_history"),
      turn_id: turn._turnId || (turn._isCurrent ? "current" : `turn_${span.turn_index}`),
      char_start: span.start,
      char_end: span.end,
      span_text: span.text,
    });
  }
  return out;
}

function semanticSpansToPopupSpans(spans, turns) {
  const out = [];
  for (const raw of spans || []) {
    if (!raw || typeof raw !== "object") continue;
    let turnIdx = turns.findIndex(turn => String(turn._turnId || "") === String(raw.turn_id || ""));
    if (turnIdx < 0 && raw.source === "current_utterance") turnIdx = turns.findIndex(turn => turn._isCurrent);
    if (turnIdx < 0 && raw.source === "prior_state") turnIdx = turns.findIndex(turn => turn._source === "prior_state");
    if (turnIdx < 0 && raw.source === "visible_history") turnIdx = turns.findIndex(turn => turn._source === "visible_history");
    if (turnIdx < 0) turnIdx = turns.length ? turns.length - 1 : 0;
    const turnText = turns[turnIdx]?.text || "";
    const spanText = String(raw.span_text || raw.text || "");
    let start = Math.max(0, Number(raw.char_start ?? raw.start ?? 0));
    let end = Math.max(start, Number(raw.char_end ?? raw.end ?? start));
    if (spanText && turnText.slice(start, end) !== spanText) {
      const found = turnText.indexOf(spanText);
      if (found >= 0) {
        start = found;
        end = found + spanText.length;
      }
    }
    out.push({
      turn_index: turnIdx,
      start,
      end: Math.min(end, turnText.length || end),
      text: spanText || turnText.slice(start, end),
    });
  }
  return out;
}

function schemaIuKey(iu) {
  return String(iu?.id || iu?.schema_iu_id || iu?.schema_id || "").trim();
}

function findExistingForSchema(schemaIu, existing, used) {
  const key = schemaIuKey(schemaIu);
  if (!key) return null;
  const idx = existing.findIndex((iu, index) => !used.has(index) && (
    String(iu.schema_iu_id || iu.schema_id || "") === key
    || String(iu.iu_id || "") === key
  ));
  if (idx < 0) return null;
  used.add(idx);
  return existing[idx];
}

function preserveIuExtras(source, target) {
  for (const key of ["accepted_variants", "acceptable_alternatives", "normalization_guidance", "allowed_extra_commitments", "notes"]) {
    if (source?.[key] !== undefined) target[key] = clone(source[key]);
  }
  return target;
}

function semanticIuFieldSpans(config, turns) {
  if (!config || !Array.isArray(config.existingIus)) return [];
  const byKey = new Map();
  for (const iu of config.existingIus) {
    for (const span of semanticSpansToPopupSpans(iu?.evidence_spans || [], turns)) {
      byKey.set(spanKey(span), span);
    }
  }
  return Array.from(byKey.values());
}

function buildSemanticIuRows(config, turns) {
  if (!config) return [];
  const existing = Array.isArray(config.existingIus) ? clone(config.existingIus) : [];
  const schemaRows = Array.isArray(config.schemaIus) ? config.schemaIus : [];
  const used = new Set();
  const rows = [];

  for (const schemaIu of schemaRows) {
    const existingIu = findExistingForSchema(schemaIu, existing, used);
    const customSpans = semanticSpansToPopupSpans(existingIu?.evidence_spans || [], turns);
    rows.push(preserveIuExtras(existingIu, {
      custom: false,
      included: Boolean(existingIu),
      schema_iu_id: schemaIuKey(schemaIu),
      iu_id: existingIu?.iu_id || "",
      name: schemaIu.name || schemaIu.id || existingIu?.schema_iu_id || "Information unit",
      description: schemaIu.description || "",
      gold_content: existingIu?.gold_content || "",
      required: existingIu?.required ?? true,
      use_field_evidence: customSpans.length === 0,
      custom_popup_spans: customSpans,
    }));
  }

  if (schemaRows.length === 0) {
    existing.forEach((iu, idx) => {
      if (used.has(idx)) return;
      const customSpans = semanticSpansToPopupSpans(iu.evidence_spans || [], turns);
      rows.push(preserveIuExtras(iu, {
        custom: true,
        included: true,
        schema_iu_id: iu.schema_iu_id || iu.schema_id || "",
        iu_id: iu.iu_id || "",
        name: iu.schema_iu_id || iu.schema_id || iu.iu_id || "Custom IU",
        description: iu.description || "",
        gold_content: iu.gold_content || "",
        required: iu.required ?? true,
        use_field_evidence: customSpans.length === 0,
        custom_popup_spans: customSpans,
      }));
    });
  }

  return rows;
}

function compileSemanticIus(config, rows, selectedSpans, turns) {
  if (!config) return null;
  const fieldPath = config.fieldPath || config.fieldId;
  return (rows || [])
    .filter(row => row.included)
    .map((row) => {
      const schemaId = String(row.schema_iu_id || row.iu_id || "").trim();
      const baseId = row.iu_id || `${compactId(config.itemId || "item")}:${compactId(fieldPath)}:${compactId(schemaId || row.name)}`;
      const popupSpans = row.use_field_evidence === false ? row.custom_popup_spans || [] : selectedSpans;
      return preserveIuExtras(row, {
        field_path: fieldPath,
        iu_id: baseId,
        schema_iu_id: schemaId,
        gold_content: String(row.gold_content || "").trim(),
        required: row.required !== false,
        evidence_spans: popupSpansToSemanticSpans(popupSpans, turns),
        status: "ready",
      });
    })
    .filter(iu => iu.schema_iu_id || iu.iu_id || iu.gold_content || iu.evidence_spans.length);
}

/**
 * Gather the full conversation: history turns + current utterance.
 * Returns array of { speaker, text, _isCurrent?, _source?, _turnId? }.
 */
async function gatherConversationTurns() {
  const turns = [];
  const d = S.currentData;
  if (!d) return turns;

  if (d.prior_state && Object.keys(d.prior_state).length > 0) {
    turns.push({
      speaker: "state",
      text: JSON.stringify(d.prior_state, null, 2),
      _isCurrent: false,
      _source: "prior_state",
      _turnId: "prior",
    });
  }

  if (Array.isArray(d.visible_history)) {
    d.visible_history.forEach((t, idx) => {
      if (!t) return;
      const text = typeof t === "string" ? t : t.text;
      if (!text) return;
      turns.push({
        speaker: typeof t === "object" ? (t.speaker || "user") : "user",
        text: String(text),
        _isCurrent: false,
        _source: "visible_history",
        _turnId: typeof t === "object" ? (t.turn_id || t.id || `history_${idx}`) : `history_${idx}`,
      });
    });
  }

  // Load history context
  if (turns.length === 0 && d.history_ref) {
    try {
      const ctx = await apiGet(`/api/contexts/history/${d.history_ref}`);
      if (ctx.turns && Array.isArray(ctx.turns)) {
        ctx.turns.forEach((t, idx) => {
          turns.push({
            speaker: t.speaker || "user",
            text: t.text || "",
            _isCurrent: false,
            _source: "visible_history",
            _turnId: t.turn_id || t.id || `history_${idx}`,
          });
        });
      }
    } catch { /* no history */ }
  }

  // Append current utterance
  const currentText = typeof d.current_utterance === "string"
    ? d.current_utterance
    : d.current_utterance?.text;
  if (currentText) {
    turns.push({
      speaker: typeof d.current_utterance === "object" ? (d.current_utterance.speaker || "user") : "user",
      text: String(currentText),
      _isCurrent: true,
      _source: "current_utterance",
      _turnId: "current",
    });
  }

  return turns;
}

/**
 * Render turn text with highlighted spans.
 */
function renderHighlightedText(container, fullText, turnIndex, spans) {
  const turnSpans = spans
    .filter(sp => sp.turn_index === turnIndex)
    .sort((a, b) => a.start - b.start);

  if (turnSpans.length === 0) {
    container.textContent = fullText;
    return;
  }

  let cursor = 0;
  for (const sp of turnSpans) {
    if (sp.start > cursor) {
      container.appendChild(document.createTextNode(fullText.slice(cursor, sp.start)));
    }
    const mark = el("mark", { class: "ev-highlight" }, fullText.slice(sp.start, sp.end));
    container.appendChild(mark);
    cursor = sp.end;
  }
  if (cursor < fullText.length) {
    container.appendChild(document.createTextNode(fullText.slice(cursor)));
  }
}

/**
 * Walk up from a node to find a parent with a specific class.
 */
function findParentWithClass(node, className) {
  let el = node.nodeType === Node.TEXT_NODE ? node.parentElement : node;
  while (el) {
    if (el.classList?.contains(className)) return el;
    el = el.parentElement;
  }
  return null;
}

/**
 * Calculate the character offsets of the selection within the turn's raw text.
 * The turn text element may contain mixed text nodes and <mark> elements,
 * so we walk through all children to compute the offset.
 */
function getSelectionOffsetsInTurn(turnTextEl, fullText) {
  const sel = window.getSelection();
  if (!sel || sel.rangeCount === 0) return null;
  const range = sel.getRangeAt(0);

  // Walk the DOM children of turnTextEl and sum text lengths to find offset
  function offsetOf(node, offsetInNode) {
    let total = 0;
    const walker = document.createTreeWalker(turnTextEl, NodeFilter.SHOW_TEXT);
    let textNode;
    while ((textNode = walker.nextNode())) {
      if (textNode === node) {
        return total + offsetInNode;
      }
      total += textNode.length;
    }
    // If node is an element, find its first/last text child
    if (node.nodeType === Node.ELEMENT_NODE) {
      // offsetInNode means nth child; find the text position right before that child
      let count = 0;
      const walker2 = document.createTreeWalker(turnTextEl, NodeFilter.SHOW_TEXT);
      let tn;
      while ((tn = walker2.nextNode())) {
        if (node.contains(tn)) {
          if (offsetInNode === 0) return total; // before first child
          return total + tn.length;
        }
        total += tn.length;
      }
      return total;
    }
    return null;
  }

  const start = offsetOf(range.startContainer, range.startOffset);
  const end = offsetOf(range.endContainer, range.endOffset);

  if (start == null || end == null || start >= end) return null;

  // Clamp to text bounds
  return {
    start: Math.max(0, Math.min(start, fullText.length)),
    end: Math.max(0, Math.min(end, fullText.length)),
  };
}

/**
 * Add a new span, merging any overlapping or adjacent spans in the same turn.
 * @param {Array} existing - Current spans
 * @param {object} newSpan - New span to add
 * @param {Array} turns - Full conversation turns (to recalculate text after merge)
 */
function mergeSpan(existing, newSpan, turns) {
  const sameTurn = existing.filter(s => s.turn_index === newSpan.turn_index);
  const other = existing.filter(s => s.turn_index !== newSpan.turn_index);

  // Merge all overlapping/adjacent
  let merged = [...sameTurn, newSpan].sort((a, b) => a.start - b.start);
  const result = [];
  for (const sp of merged) {
    if (result.length === 0) { result.push({ ...sp }); continue; }
    const last = result[result.length - 1];
    if (sp.start <= last.end) {
      // Overlap or adjacent — extend
      last.end = Math.max(last.end, sp.end);
    } else {
      result.push({ ...sp });
    }
  }

  // Fix text for merged spans using the turn text
  const turnText = turns[newSpan.turn_index]?.text || "";
  for (const sp of result) {
    sp.text = turnText.slice(sp.start, sp.end);
  }

  return [...other, ...result];
}
