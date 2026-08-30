// How-To page for custom questionnaires and datasets.
import { S } from "./state.js";
import { el, markClean, setEditorHeader } from "./dom-utils.js";

function codeBlock(text) {
  return el("pre", { class: "howto-code" }, el("code", {}, String(text).trim()));
}

function inlineCode(text) {
  return el("code", {}, text);
}

function section(id, kicker, title, copy, ...children) {
  return el("section", { class: "howto-section", id },
    el("div", { class: "howto-section-kicker" }, kicker),
    el("h2", {}, title),
    copy ? el("p", { class: "howto-copy" }, copy) : null,
    ...children,
  );
}

function bulletList(items) {
  return el("ul", { class: "howto-list" }, ...items.map(item => el("li", {}, ...item)));
}

function pathTable(rows) {
  const body = rows.map(row => el("tr", {},
    el("td", {}, inlineCode(row.path)),
    el("td", {}, row.purpose),
  ));
  return el("table", { class: "howto-table" },
    el("thead", {}, el("tr", {}, el("th", {}, "Path"), el("th", {}, "Purpose"))),
    el("tbody", {}, ...body),
  );
}

function renderQuickPanel() {
  const panel = document.getElementById("checklist-panel");
  panel.innerHTML = "";

  const questionnaireText = S.questionnaireNames.length
    ? S.questionnaireNames.join(", ")
    : "No questionnaires loaded yet";

  panel.append(
    el("aside", { class: "howto-quick-panel" },
      el("h3", {}, "Quick Reference"),
      el("div", { class: "howto-quick-item" },
        el("strong", {}, "Loaded questionnaires"),
        el("span", {}, questionnaireText),
      ),
      el("div", { class: "howto-quick-item" },
        el("strong", {}, "Questionnaire files"),
        el("span", {}, "data/schema/questionnaires/*.json"),
      ),
      el("div", { class: "howto-quick-item" },
        el("strong", {}, "Dataset items"),
        el("span", {}, "data/items/benchmark/<questionnaire>/<item>/ground_truth.json"),
      ),
      el("div", { class: "howto-quick-item" },
        el("strong", {}, "Custom data root"),
        el("span", {}, "CONFORMBENCH_DATA_DIR=/path/to/data"),
      ),
    ),
  );
}

export async function renderHowToView() {
  document.getElementById("no-selection").style.display = "none";
  const editor = document.getElementById("editor");
  editor.style.display = "flex";

  setEditorHeader({
    idText: "HOW-TO",
    titleText: "Custom questionnaires and datasets",
    saveLabel: "",
  });
  document.getElementById("btn-save").style.display = "none";
  document.getElementById("btn-prev").style.display = "none";
  document.getElementById("btn-next").style.display = "none";

  const form = document.getElementById("editor-form");
  form.innerHTML = "";
  renderQuickPanel();

  const page = el("article", { class: "howto-page" },
    el("header", { class: "howto-header", id: "howto-orientation" },
      el("div", { class: "howto-eyebrow" }, "Release workflow"),
      el("h1", {}, "Bring your own form, then author benchmark turns against it."),
      el("p", {},
        "ConFormBench treats a questionnaire as the schema for a structured record. ",
        "A dataset is a folder of item packets that pair that schema with prior state, visible history, a user turn, and the expected resulting state.",
      ),
    ),

    section(
      "howto-questionnaire",
      "Step 1",
      "Add a Questionnaire",
      "Create one JSON schema file per form. The file stem is the questionnaire id used everywhere else.",
      pathTable([
        {
          path: "data/schema/questionnaires/support_ticket.json",
          purpose: "Questionnaire schema loaded by list-questionnaires and Studio.",
        },
        {
          path: "data/schema/schema.md",
          purpose: "Reference for structure_type, field types, gates, branches, repeat groups, and IUs.",
        },
      ]),
      bulletList([
        ["Use stable lowercase ids such as ", inlineCode("support_ticket"), " and ", inlineCode("customer_name"), "."],
        ["Every real field is a ", inlineCode("structure_type: \"regular\""), " node with ", inlineCode("id"), ", ", inlineCode("question_text"), ", ", inlineCode("type"), ", and ", inlineCode("gold_standard"), "."],
        ["Use ", inlineCode("group"), " for layout, ", inlineCode("gate"), " for conditional follow-ups, ", inlineCode("branch"), " for route-specific questions, and ", inlineCode("repeat_group"), " for arrays of repeated records."],
        ["Refresh Studio after adding the file. The questionnaire should appear in the Items tab and in ", inlineCode("conformbench list-questionnaires"), "."],
      ]),
      codeBlock(`
{
  "title": "Support Ticket Intake",
  "record_context": {
    "domain": "support",
    "primary_subject": "customer",
    "speaker_role": "customer",
    "event_scope": "current ticket"
  },
  "questions": [
    {
      "id": "customer_name",
      "structure_type": "regular",
      "question_text": "Customer name",
      "type": "text",
      "gold_standard": "The customer's stated name; null if not given."
    },
    {
      "id": "issue_count",
      "structure_type": "regular",
      "question_text": "Number of issues",
      "type": "number",
      "gold_standard": "Count distinct issues the customer wants handled."
    },
    {
      "id": "issues",
      "structure_type": "repeat_group",
      "label": "Issues",
      "repeat": { "mode": "from_slot", "from_slot": "issue_count", "item_label": "Issue {{index}}" },
      "fields": [
        {
          "id": "summary",
          "structure_type": "regular",
          "question_text": "Issue summary",
          "type": "text",
          "gold_standard": "Short grounded summary of this issue."
        }
      ]
    }
  ]
}`),
    ),

    section(
      "howto-state",
      "Step 2",
      "Define the Record State Shape",
      "The runner and evaluator score full post-turn record states. Keep prior and gold states explicit and schema-shaped.",
      bulletList([
        ["Top-level regular fields become top-level JSON keys."],
        ["Repeat groups become arrays. Each row contains that repeat group's child field ids."],
        ["Use ", inlineCode("null"), " for unknown scalar values and ", inlineCode("[]"), " for empty repeat or table fields."],
        ["Gold state should preserve unchanged prior values and only differ where the turn licenses a change."],
      ]),
      codeBlock(`
{
  "customer_name": null,
  "issue_count": 0,
  "issues": []
}`),
      codeBlock(`
{
  "customer_name": "Ana Kim",
  "issue_count": 1,
  "issues": [
    { "summary": "Cannot reset password" }
  ]
}`),
    ),

    section(
      "howto-dataset",
      "Step 3",
      "Create Dataset Items",
      "Use Studio's New item flow after the questionnaire exists, or write ground_truth.json packets directly. Studio saves editable items under the configured items directory.",
      pathTable([
        {
          path: "data/items/benchmark/support_ticket/support_ticket_001/ground_truth.json",
          purpose: "One benchmark item packet.",
        },
        {
          path: "data/items/public/<questionnaire>/<item>/ground_truth.json",
          purpose: "Packaged public development examples. Studio treats these as read-only.",
        },
      ]),
      codeBlock(`
{
  "item_id": "support_ticket_001",
  "questionnaire_id": "support_ticket",
  "status": "ready",
  "primary_delta_type": "add",
  "prior_state": {
    "customer_name": null,
    "issue_count": 0,
    "issues": []
  },
  "visible_history": [],
  "current_utterance": "My name is Ana Kim and I cannot reset my password.",
  "gold_resulting_state": {
    "customer_name": "Ana Kim",
    "issue_count": 1,
    "issues": [
      { "summary": "Cannot reset password" }
    ]
  }
}`),
    ),

    section(
      "howto-run",
      "Step 4",
      "Run and Score the Dataset",
      "Use the Eval tab for local inspection, or run the same packets from the CLI for reproducible artifacts.",
      codeBlock(`
conformbench list-questionnaires
conformbench studio --items-dir data/items/benchmark --runs-dir data/reports/runs
conformbench run \\
  --items data/items/benchmark \\
  --solver conformbench.systems.YOUR_SYSTEM:solve \\
  --output-dir data/reports/runs/support_ticket_smoke \\
  --score`),
      bulletList([
        ["Use ", inlineCode("CONFORMBENCH_DATA_DIR=/path/to/data"), " when your schema, items, and reports live outside the installed package data directory."],
        ["The Eval tab can run public, Studio-authored, or selected items and then inspect per-field verdicts."],
        ["Generated run artifacts live under ", inlineCode("data/reports/runs/<run_id>"), " unless you override the runs directory."],
      ]),
    ),

    section(
      "howto-checklist",
      "Quality Gate",
      "Before sharing a dataset, make the contract boring and repeatable.",
      bulletList([
        ["Questionnaire ids and item ids are stable, unique, and filesystem-safe."],
        ["Each prior state and gold state uses the same shape for the same questionnaire."],
        ["All repeat rows include the expected child keys, even when a child value is ", inlineCode("null"), "."],
        ["The current utterance contains enough evidence for every intended change."],
        ["Forbidden or tempting non-gold commitments are documented in item notes or metadata when they matter."],
        ["A smoke run with a trivial solver and one real solver produces readable reports in Studio."],
      ]),
    ),
  );

  form.appendChild(page);
  markClean();
}
