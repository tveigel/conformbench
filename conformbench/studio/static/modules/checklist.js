// -- Checklist ----------------------------------------------------------------
import { S } from "./state.js";
import { el, ensureDifficultyProfile, readyItemRequiresFailureContract, cleanTargetedFailureModes } from "./dom-utils.js";
import { DIMENSIONS, DIMENSION_DEFAULTS } from "./constants.js";

export const CHECKLIST_ITEMS = [
  { id: "cl-utt", label: "Utterance written", sectionId: "section-context-utt", check: d => !!d.current_utterance?.text && !d.current_utterance.text.includes("TODO") },
  { id: "cl-cond", label: "Conditions set", sectionId: "section-scenario", check: d => !!d.state_condition && !!d.primary_delta_type && !!d.difficulty_tier },
  { id: "cl-outcome", label: "Gold resulting state", sectionId: "section-gold", check: d => {
    const grs = d.gold_resulting_state;
    if (grs && typeof grs === "object") {
      const vals = Object.values(grs);
      return vals.length > 0 && vals.some(v => v != null && v !== "");
    }
    return false;
  }},
  { id: "cl-dims", label: "Item stressor profile", sectionId: "section-gold", check: d => {
    ensureDifficultyProfile(d);
    const dp = d.difficulty_profile;
    for (const dim of DIMENSIONS) {
      const val = dp.dimensions[dim.key];
      if (!dim.values.includes(val)) return false;
      if (val !== DIMENSION_DEFAULTS[dim.key] && !(dp.dimension_notes?.[dim.key] || "").trim()) return false;
    }
    return true;
  }},
  { id: "cl-fail", label: "Failure modes", sectionId: "section-gold", check: d => {
    if (!readyItemRequiresFailureContract(d)) return true;
    return cleanTargetedFailureModes(d).length > 0;
  }},
];

export function renderChecklist() {
  const panel = document.getElementById("checklist-panel");
  panel.innerHTML = "";
  if (S.appMode !== "items" || !S.currentData) return;

  const card = el("div", { class: "checklist-card" });
  card.appendChild(el("h3", {}, "Authoring checklist"));
  for (const ci of CHECKLIST_ITEMS) {
    const done = ci.check(S.currentData);
    card.appendChild(el("div", {
      class: `checklist-item${done ? " done" : ""}`,
      id: ci.id,
      onclick: () => {
        const sec = document.getElementById(ci.sectionId);
        if (sec) { sec.classList.remove("collapsed"); sec.scrollIntoView({ behavior: "smooth", block: "start" }); }
      },
    }, el("div", { class: "check" }, done ? "\u2713" : ""), ci.label));
  }
  const s = S.currentData?.status || "template";
  card.appendChild(el("div", { style: "margin-top:14px;padding-top:12px;border-top:1px solid var(--border);" },
    el("div", { style: "font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.04em;color:var(--text-3);margin-bottom:6px;" }, "Current status"),
    el("div", { class: `status-badge ${s}`, style: "display:inline-block;" }, s),
  ));
  panel.appendChild(card);
}

export function updateChecklist() {
  if (S.appMode !== "items" || !S.currentData) return;
  for (const ci of CHECKLIST_ITEMS) {
    const node = document.getElementById(ci.id);
    if (!node) continue;
    const done = ci.check(S.currentData);
    node.className = `checklist-item${done ? " done" : ""}`;
    const check = node.querySelector(".check");
    if (check) check.textContent = done ? "\u2713" : "";
  }
}
