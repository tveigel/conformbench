// API helpers for the public Studio.
import { S } from "./state.js";

async function errorFor(response, fallback) {
  let detail = "";
  try {
    const data = await response.json();
    detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
  } catch {
    detail = await response.text();
  }
  return new Error(detail || fallback);
}

export async function apiGet(url) {
  const response = await fetch(url);
  if (!response.ok) throw await errorFor(response, `GET ${url} -> ${response.status}`);
  return response.json();
}

export async function apiPut(url, body) {
  const response = await fetch(url, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw await errorFor(response, `PUT ${url} -> ${response.status}`);
  return response.json();
}

export async function apiPost(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw await errorFor(response, `POST ${url} -> ${response.status}`);
  return response.json();
}

export async function apiDelete(url) {
  const response = await fetch(url, { method: "DELETE" });
  if (!response.ok) throw await errorFor(response, `DELETE ${url} -> ${response.status}`);
  return response.json();
}

export async function getQFields(qname) {
  if (!qname) return [];
  if (S.qFieldsCache[qname]) return S.qFieldsCache[qname];
  try {
    const data = await apiGet(`/api/questionnaire/${qname}/fields`);
    S.qFieldsCache[qname] = data.fields || [];
  } catch {
    S.qFieldsCache[qname] = [];
  }
  return S.qFieldsCache[qname];
}

export async function getQFieldMeta(qname) {
  if (!qname) return [];
  if (S.qFieldMetaCache[qname]) return S.qFieldMetaCache[qname];
  try {
    const data = await apiGet(`/api/questionnaire/${qname}/field-meta`);
    S.qFieldMetaCache[qname] = data.fields || [];
  } catch {
    S.qFieldMetaCache[qname] = [];
  }
  return S.qFieldMetaCache[qname];
}

export async function getQTree(qname) {
  if (!qname) return [];
  if (S.qTreeCache[qname]) return S.qTreeCache[qname];
  try {
    const data = await apiGet(`/api/questionnaire/${qname}/tree`);
    S.qTreeCache[qname] = data.questions || [];
  } catch {
    S.qTreeCache[qname] = [];
  }
  return S.qTreeCache[qname];
}

export async function loadContexts() {
  const [states, histories] = await Promise.all([
    apiGet("/api/contexts/state").catch(() => []),
    apiGet("/api/contexts/history").catch(() => []),
  ]);
  S.allContexts = { state: states, history: histories };
}

export async function loadScenarios() {
  S.allScenarios = await apiGet("/api/scenarios").catch(() => []);
}
