// Public Studio sidebar: item library plus evaluation mode.
import { S, persistState } from "./state.js";
import { el, setModeTabs, updateEmptyState } from "./dom-utils.js";
import { openItem } from "./navigation.js";
import { copyItemPrompt, deleteItemPrompt, createNewItem } from "./dialogs.js";

const HOWTO_SECTIONS = [
  { id: "howto-orientation", label: "Start here" },
  { id: "howto-questionnaire", label: "Questionnaire" },
  { id: "howto-state", label: "State shape" },
  { id: "howto-dataset", label: "Dataset" },
  { id: "howto-run", label: "Run and score" },
  { id: "howto-checklist", label: "Checklist" },
];

const ITEM_GROUP_MODES = [
  { value: "questionnaire", label: "Questionnaire" },
  { value: "primary_delta_type", label: "Primary type" },
  { value: "status", label: "Status" },
  { value: "state_condition", label: "Prior state" },
  { value: "evidence_profile", label: "Evidence" },
  { value: "none", label: "No grouping" },
];

const PRIMARY_TYPE_META = {
  add: { label: "Add", order: 0 },
  refine: { label: "Refine", order: 1 },
  correct: { label: "Correct", order: 2 },
  retract: { label: "Retract", order: 3 },
  repair: { label: "Repair", order: 4 },
  unknown: { label: "Unspecified", order: 99 },
};

const STATUS_META = {
  ready: { label: "Ready", order: 0 },
  draft: { label: "Draft", order: 1 },
  template: { label: "Template", order: 2 },
  unknown: { label: "Unknown", order: 99 },
};

const STATE_META = {
  S1: { label: "S1 Empty", order: 0 },
  S2: { label: "S2 Partial-correct", order: 1 },
  S3: { label: "S3 Partial-incorrect", order: 2 },
  S4: { label: "S4 Inconsistent", order: 3 },
  unknown: { label: "Unknown state", order: 99 },
};

const EVIDENCE_PROFILE_META = {
  current_only: { label: "Current utterance only", order: 0 },
  recent_history: { label: "Recent history required", order: 1 },
  distant_history: { label: "Distant history required", order: 2 },
  conflict_present: { label: "Conflict present", order: 3 },
  unknown: { label: "Unknown evidence", order: 99 },
};

function _itemPrimaryType(item) {
  const raw = (item.primary_delta_type || item.family || "").toLowerCase();
  return PRIMARY_TYPE_META[raw] ? raw : "unknown";
}

function _itemEvidenceProfile(item) {
  const evidence = item.evidence || {};
  if (evidence.conflict_present) return "conflict_present";
  if (evidence.history_required) {
    return Number(evidence.support_distance || 0) >= 2 ? "distant_history" : "recent_history";
  }
  return "current_only";
}

function _itemGroupValue(item, mode) {
  if (mode === "status") return item.status || "unknown";
  if (mode === "state_condition") return item.state_condition || "unknown";
  if (mode === "questionnaire") return item.questionnaire || "unknown";
  if (mode === "evidence_profile") return _itemEvidenceProfile(item);
  return _itemPrimaryType(item);
}

function _itemGroupLabel(mode, value) {
  if (mode === "status") return (STATUS_META[value] || STATUS_META.unknown).label;
  if (mode === "state_condition") return (STATE_META[value] || STATE_META.unknown).label;
  if (mode === "questionnaire") return value === "unknown" ? "Unknown questionnaire" : value.replace(/_/g, " ");
  if (mode === "evidence_profile") return (EVIDENCE_PROFILE_META[value] || EVIDENCE_PROFILE_META.unknown).label;
  return (PRIMARY_TYPE_META[value] || PRIMARY_TYPE_META.unknown).label;
}

function _itemGroupOrder(mode, value) {
  if (mode === "status") return (STATUS_META[value] || STATUS_META.unknown).order;
  if (mode === "state_condition") return (STATE_META[value] || STATE_META.unknown).order;
  if (mode === "evidence_profile") return (EVIDENCE_PROFILE_META[value] || EVIDENCE_PROFILE_META.unknown).order;
  if (mode === "primary_delta_type") return (PRIMARY_TYPE_META[value] || PRIMARY_TYPE_META.unknown).order;
  return 999;
}

function _sortItemGroups(mode, entries) {
  return entries.sort(([a], [b]) => {
    if (mode === "questionnaire") return a.localeCompare(b);
    const orderDiff = _itemGroupOrder(mode, a) - _itemGroupOrder(mode, b);
    return orderDiff || a.localeCompare(b);
  });
}

function _renderItemGroupToken(mode, value) {
  if (mode === "primary_delta_type") {
    return el("span", { class: `sidebar-group-token delta ${value}` }, _itemGroupLabel(mode, value));
  }
  if (mode === "status") {
    return el("span", { class: `status-badge ${value}` }, _itemGroupLabel(mode, value));
  }
  if (mode === "questionnaire") {
    return el("span", { class: `q-badge ${value}` }, _itemGroupLabel(mode, value));
  }
  return el("span", { class: "sidebar-group-token neutral" }, _itemGroupLabel(mode, value));
}

function _getItemTypeFilters() {
  return Array.isArray(S.sidebarItemTypeFilters) ? S.sidebarItemTypeFilters : [];
}

function _hasItemTypeFilters() {
  return _getItemTypeFilters().length > 0;
}

function _hasItemTypeFilter(value) {
  return _getItemTypeFilters().includes(value);
}

function _setItemTypeFilters(values, { groupByType = false } = {}) {
  S.sidebarItemTypeFilters = [...new Set(values)]
    .filter(value => PRIMARY_TYPE_META[value])
    .sort((a, b) => _itemGroupOrder("primary_delta_type", a) - _itemGroupOrder("primary_delta_type", b) || a.localeCompare(b));
  if (groupByType) S.sidebarItemGroupMode = "primary_delta_type";
  persistState();
  renderSidebarActions();
  renderSidebarList();
}

function _toggleItemTypeFilter(value) {
  const next = _hasItemTypeFilter(value)
    ? _getItemTypeFilters().filter(v => v !== value)
    : [..._getItemTypeFilters(), value];
  _setItemTypeFilters(next, { groupByType: true });
}

function _clearItemTypeFilters() {
  _setItemTypeFilters([]);
}

function _applyItemTypeFilters(items) {
  if (!_hasItemTypeFilters()) return items;
  const active = _getItemTypeFilters();
  return items.filter(item => active.includes(_itemPrimaryType(item)));
}

function _describeItemTypeFilters() {
  const labels = _getItemTypeFilters().map(value => _itemGroupLabel("primary_delta_type", value).toLowerCase());
  if (!labels.length) return "";
  if (labels.length <= 2) return labels.join(" + ");
  return `${labels.length} types`;
}

function _renderItemCoverageSummary(items) {
  const card = el("div", { class: "sidebar-summary-card" });
  const readyCount = items.filter(item => item.status === "ready").length;
  const draftCount = items.filter(item => item.status === "draft").length;
  const templateCount = items.filter(item => item.status === "template").length;
  const typeCounts = Object.fromEntries(Object.keys(PRIMARY_TYPE_META).map(key => [key, 0]));

  for (const item of items) {
    const key = _itemPrimaryType(item);
    typeCounts[key] = (typeCounts[key] || 0) + 1;
  }

  const header = el("div", { class: "sidebar-summary-header" },
    el("div", {},
      el("div", { class: "sidebar-summary-title" }, "Primary update coverage"),
      el("div", { class: "sidebar-summary-subtitle" }, `${items.length} item${items.length !== 1 ? "s" : ""}`),
    ),
    el("div", { class: "sidebar-summary-meta" }, `${readyCount} ready`),
  );

  const chips = el("div", { class: "sidebar-chip-row" });
  chips.appendChild(el("button", {
    class: `sidebar-chip${!_hasItemTypeFilters() ? " active" : ""}`,
    type: "button",
    onclick: () => _clearItemTypeFilters(),
  }, "All", el("span", { class: "sidebar-chip-count" }, String(items.length))));

  for (const key of ["add", "refine", "correct", "retract"]) {
    chips.appendChild(el("button", {
      class: `sidebar-chip delta ${key}${_hasItemTypeFilter(key) ? " active" : ""}`,
      type: "button",
      title: `Filter to ${PRIMARY_TYPE_META[key].label.toLowerCase()} items`,
      onclick: () => _toggleItemTypeFilter(key),
    }, PRIMARY_TYPE_META[key].label, el("span", { class: "sidebar-chip-count" }, String(typeCounts[key] || 0))));
  }

  const statusRow = el("div", { class: "sidebar-summary-statuses" },
    el("span", { class: "sidebar-inline-stat ready" }, `Ready ${readyCount}`),
    el("span", { class: "sidebar-inline-stat draft" }, `Draft ${draftCount}`),
    el("span", { class: "sidebar-inline-stat template" }, `Template ${templateCount}`),
  );

  card.append(header, chips, statusRow);
  return card;
}

function _renderItemCard(item) {
  const qLabel = item.questionnaire ? item.questionnaire.replace(/_/g, " ") : "unknown";
  const qClass = item.questionnaire || "unknown";
  const evidenceProfile = _itemEvidenceProfile(item);
  const evidenceMeta = EVIDENCE_PROFILE_META[evidenceProfile] || EVIDENCE_PROFILE_META.unknown;
  const itemMeta = [
    item.source || "",
    item.read_only ? "read-only" : "",
    item.history_ref || item.visible_history ? "history" : "",
  ].filter(Boolean).join(" / ");
  const semanticMeta = [
    item.state_condition,
    _itemGroupLabel("primary_delta_type", _itemPrimaryType(item)),
    evidenceProfile === "current_only" ? "" : evidenceMeta.label,
  ].filter(Boolean).join(" · ");

  return el("div", {
    class: `item-card${item.item_id === S.currentId ? " active" : ""}`,
    "data-id": item.item_id,
    "data-family": item.primary_delta_type || item.family,
    onclick: event => { if (!event.target.closest(".card-actions")) openItem(item.item_id); },
  },
    el("div", { class: "item-card-dot" }),
    el("div", { class: "item-card-body" },
      el("div", { class: "item-card-id" },
        item.item_id,
        el("span", { class: `q-badge ${qClass}` }, qLabel),
      ),
      el("div", { class: "item-card-title", title: item.title || "(untitled)" }, item.title || "(untitled)"),
      el("div", { class: "item-card-meta" }, itemMeta || semanticMeta),
      semanticMeta && itemMeta ? el("div", { class: "item-card-meta" }, semanticMeta) : null,
    ),
    el("div", { class: "item-card-right" },
      el("div", { class: `status-badge ${item.status}` }, item.status || "draft"),
      el("div", { class: "card-actions" },
        el("button", { class: "card-action-btn", title: "Duplicate item", type: "button", onclick: () => copyItemPrompt(item.item_id) }, "⧉"),
        item.read_only ? null : el("button", { class: "card-action-btn danger", title: "Delete item", type: "button", onclick: () => deleteItemPrompt(item.item_id) }, "x"),
      ),
    ),
  );
}

export function renderSidebarActions() {
  const host = document.getElementById("sidebar-actions");
  host.innerHTML = "";

  if (S.appMode !== "items") return;

  const searchRow = el("div", { class: "sidebar-search-row" });
  const searchInput = el("input", {
    type: "text",
    class: "sidebar-search-input",
    placeholder: "Search items...",
    value: S.sidebarSearchQuery,
  });
  const clearBtn = el("button", {
    class: "sidebar-search-clear",
    type: "button",
    style: S.sidebarSearchQuery ? "" : "display:none;",
    onclick: () => {
      S.sidebarSearchQuery = "";
      searchInput.value = "";
      clearBtn.style.display = "none";
      renderSidebarList();
    },
  }, "x");
  searchInput.addEventListener("input", () => {
    S.sidebarSearchQuery = searchInput.value;
    clearBtn.style.display = searchInput.value ? "" : "none";
    renderSidebarList();
  });
  searchRow.append(searchInput, clearBtn);

  const row = el("div", { class: "sidebar-action-row" });
  row.appendChild(el("button", { class: "btn sm primary", type: "button", onclick: () => createNewItem() }, "+ New item"));
  if (_hasItemTypeFilters()) {
    row.appendChild(el("button", {
      class: "btn sm",
      type: "button",
      onclick: () => _clearItemTypeFilters(),
    }, _getItemTypeFilters().length === 1
      ? `Clear ${_itemGroupLabel("primary_delta_type", _getItemTypeFilters()[0])}`
      : `Clear ${_getItemTypeFilters().length} filters`));
  }

  const groupWrap = el("div", { class: "sidebar-select-block" });
  const groupSel = el("select", { class: "sidebar-group-select" });
  for (const mode of ITEM_GROUP_MODES) {
    groupSel.append(el("option", {
      value: mode.value,
      ...(mode.value === S.sidebarItemGroupMode ? { selected: "" } : {}),
    }, `Group by ${mode.label}`));
  }
  groupSel.addEventListener("change", () => {
    S.sidebarItemGroupMode = groupSel.value;
    persistState();
    renderSidebarList();
  });
  groupWrap.appendChild(groupSel);

  host.append(searchRow, row, groupWrap, _renderItemCoverageSummary(S.allItems));
}

export function renderSidebarList() {
  const list = document.getElementById("item-list");
  const progressOuter = document.getElementById("progress-bar-outer");
  const progressInner = document.getElementById("progress-bar-inner");
  const progressLabel = document.getElementById("progress-label");
  list.innerHTML = "";

  if (S.appMode === "eval") {
    progressOuter.style.display = "none";
    progressLabel.textContent = "";
    list.appendChild(el("div", { class: "sidebar-no-results", style: "margin-top:2em;text-align:center;" },
      el("div", { style: "font-size:2em;margin-bottom:.5em" }, "Eval"),
      el("div", {}, "Run and inspect benchmark results in the main panel."),
    ));
    return;
  }

  if (S.appMode === "howto") {
    progressOuter.style.display = "none";
    progressLabel.textContent = "Custom questionnaire and dataset guide";
    const nav = el("div", { class: "howto-sidebar-nav" },
      el("div", { class: "sidebar-summary-title" }, "How-To sections"),
    );
    for (const section of HOWTO_SECTIONS) {
      nav.appendChild(el("a", {
        class: "howto-sidebar-link",
        href: `#${section.id}`,
        onclick: event => {
          event.preventDefault();
          document.getElementById(section.id)?.scrollIntoView({ behavior: "smooth", block: "start" });
        },
      }, section.label));
    }
    list.appendChild(nav);
    list.appendChild(el("div", { class: "sidebar-no-results" },
      "Use this page when you want to bring your own form schema or curate a project-specific benchmark split.",
    ));
    return;
  }

  progressOuter.style.display = "block";
  let baseItems = _applyItemTypeFilters(S.allItems);
  const readyCount = baseItems.filter(item => item.status === "ready").length;
  progressInner.style.width = baseItems.length ? `${(readyCount / baseItems.length) * 100}%` : "0%";

  const groupLabel = ITEM_GROUP_MODES.find(mode => mode.value === S.sidebarItemGroupMode)?.label || "Questionnaire";
  const filterLabel = _hasItemTypeFilters() ? ` · filtered to ${_describeItemTypeFilters()}` : "";
  progressLabel.textContent = `${readyCount} / ${baseItems.length} ready · grouped by ${groupLabel.toLowerCase()}${filterLabel}`;

  const query = S.sidebarSearchQuery.toLowerCase();
  const filtered = query
    ? baseItems.filter(item => `${item.item_id} ${item.title} ${item.primary_delta_type || item.family} ${_itemEvidenceProfile(item)} ${item.status} ${item.state_condition} ${item.questionnaire || ""}`.toLowerCase().includes(query))
    : baseItems;

  if (S.sidebarItemGroupMode !== "none") {
    const groups = {};
    for (const item of filtered) {
      const key = _itemGroupValue(item, S.sidebarItemGroupMode);
      (groups[key] ??= []).push(item);
    }
    for (const [groupKey, groupItems] of _sortItemGroups(S.sidebarItemGroupMode, Object.entries(groups))) {
      const groupReadyCount = groupItems.filter(item => item.status === "ready").length;
      list.appendChild(el("div", { class: "sidebar-group-header" },
        _renderItemGroupToken(S.sidebarItemGroupMode, groupKey),
        el("span", { class: "sidebar-group-count" }, `${groupItems.length} item${groupItems.length !== 1 ? "s" : ""}`),
        el("span", { class: "sidebar-group-subcount" }, `${groupReadyCount} ready`),
      ));
      for (const item of groupItems) list.appendChild(_renderItemCard(item));
    }
  } else {
    for (const item of filtered) list.appendChild(_renderItemCard(item));
  }

  if (!filtered.length) {
    list.appendChild(el("div", { class: "sidebar-no-results" },
      query ? `No items matching "${query}"` : "No items loaded",
    ));
  }
}

export function renderSidebar() {
  setModeTabs();
  updateEmptyState();
  renderSidebarActions();
  renderSidebarList();
}
