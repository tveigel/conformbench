// Shared mutable application state for the public Studio.

const _PERSIST_KEYS = [
  "appMode",
  "sidebarSearchQuery",
  "sidebarItemGroupMode",
  "sidebarItemTypeFilters",
  "currentId",
];
const _LS_KEY = "benchStudioState";

function _loadPersisted() {
  try {
    const raw = localStorage.getItem(_LS_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

const _saved = _loadPersisted();
const _savedMode = ["items", "eval", "howto"].includes(_saved.appMode) ? _saved.appMode : "items";

export const S = {
  appMode: _savedMode,
  allItems: [],
  allContexts: { state: [], history: [] },
  allScenarios: [],
  questionnaireNames: [],

  currentId: _saved.currentId ?? null,
  currentData: null,

  // The item editor uses these while an inline state/history modal is open.
  currentContextKind: null,
  currentContextRef: null,
  currentContextData: null,

  qFieldsCache: {},
  qFieldMetaCache: {},
  qTreeCache: {},
  isDirty: false,
  _ctxContainer: null,

  sidebarSearchQuery: _saved.sidebarSearchQuery ?? "",
  sidebarItemGroupMode: _saved.sidebarItemGroupMode || "questionnaire",
  sidebarItemTypeFilters: _saved.sidebarItemTypeFilters || [],
};

export function persistState() {
  try {
    const snap = {};
    for (const k of _PERSIST_KEYS) snap[k] = S[k];
    localStorage.setItem(_LS_KEY, JSON.stringify(snap));
  } catch {
    // Local storage is a convenience only.
  }
}
