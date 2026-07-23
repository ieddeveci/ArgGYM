import random
import traceback
import uuid

from flask import Flask, request, jsonify, render_template_string

from aspic_api import ASPICVerifier
from aspic_engine import contrary
from aspic_gym import (create_dataset, score_answer, ALL_TASKS, CONTENT_KINDS,
                       MAX_LEVEL, ATTACK_CHAIN_LEVEL, TASK_MIN_LEVEL)
from evals import template

app = Flask(__name__)

TASKS = {}

ORDERINGS = ["last_link_elitist", "last_link_democratic",
             "weakest_link_elitist", "weakest_link_democratic"]

TASK_KINDS = [
    ("syntax", "Syntax", "Write one directive or token in the required format (a format / vocabulary drill; ignores level)."),
    ("status_query", "Status Query", "Classify the grounded status of several claims."),
    ("claim_identification", "Claim ID", "Trace the argument(s); list the claims it establishes."),
    ("attackers_of", "Attackers", "Name every attacker of the target argument (by its conclusion)."),
    ("enthymeme", "Enthymeme", "Supply the missing element(s) of the argument."),
    ("attack", "Attack", "Mount a specified attack (undermine/rebut/undercut) that defeats the conclusion."),
    ("counter_argumentation", "Counter-Argument", "Build a winning case for the opposite claim."),
    ("preference_construction", "Preference", "Add the preference(s) that settle an undecided claim."),
    ("formalization", "Formalize", "Write the statement/argument in the proper format."),
    ("perturbation_prediction", "Perturbation",
     "Predict how statuses change after the theory is perturbed (belief revision)."),
    ("robustness", "Robustness",
     "Find the load-bearing premise(s): what must be removed to flip the target."),
    ("evidence_construction", "Evidence Construction",
     "Build a verified argument from a shared pool of statements, using the right-stance ones."),
    ("ordering_sensitivity", "Ordering Sensitivity",
     "Give a claim's status under BOTH last-link and weakest-link ordering (they differ)."),
]

_listed = {k for k, _, _ in TASK_KINDS}
_missing_from_ui = set(ALL_TASKS) - _listed
assert not _missing_from_ui, (
    f"TASK_KINDS is missing {sorted(_missing_from_ui)} — add them so they appear in the viewer")
ANSWER_HINT = {
    "syntax": "Write exactly what the instruction asks for inside [answer] ... [/answer]: "
              "usually one directive (e.g. [premise: i5], [strict: a -> b]), sometimes a bare "
              "token (justified / none / d7 / -a), a `number: status` line, or two directives "
              "in a stated order.",
    "status_query": "Put the answer inside [answer] ... [/answer], one line per claim as "
                    "`number: status` (e.g. `1: justified`); statuses: justified / overruled / "
                    "undecided / unsatisfiable.",
    "claim_identification": "Numbers of the established claims, e.g.  1, 3",
    "attackers_of": "Each attacker's conclusion, e.g. -a1, -d2 (or the statements, in content mode).",
    "enthymeme": "The MISSING directive(s), one per line — restore the intended argument; "
                 "a working shortcut that skips the gaps only earns partial credit.",
    "attack": "Directives, one per line, e.g. [premise: -p3] to undermine, [premise: b] + "
              "[defeasible: b => -x] to rebut by derivation, [premise: -d2] to undercut rule d2.",
    "counter_argumentation": "Directive(s) making the opposite claim justified.",
    "preference_construction": "Preference directive(s), e.g. [prefer_rule: d1 > d2].",
    "perturbation_prediction": "Put the answer inside [answer] ... [/answer], one line per "
                               "statement as `number: status` (e.g. `1: overruled`) predicting "
                               "the status AFTER the described change(s); statuses: justified / "
                               "overruled / undecided / unsatisfiable.",
    "robustness": "Name the premise statement(s) inside [answer] ... [/answer] "
                  "(one for single/reinstate, the full list for the set variant, exactly two "
                  "comma-separated for a pair); symbolic mode uses atom ids, content mode the "
                  "statements themselves.",
    "formalization": "semantic equivalence \u2014 the engine compares the BEHAVIOR of your theory to the gold theory (statuses under counterfactual probes); renamed or reordered rules score 1.0, and non-equivalent theories get at most 0.25 shaped credit",
    "evidence_construction": "Assemble an argument from the listed statements, finishing "
                             "at the target conclusion (symbolic mode uses item ids; content mode "
                             "uses the statements themselves).",
}
GRADING_MODE = {
    "syntax": "exact — the [answer] block must contain exactly the one required directive "
              "(parsed equivalently; whitespace/label-insensitive); 0 otherwise. A pure "
              "format / vocabulary drill — level is ignored.",
    "status_query": "partial — fraction of claims labelled correctly (strict: only the "
                    "exact [answer] ... [/answer] block is read; 0 if the tags are missing)",
    "claim_identification": "partial — Jaccard over established claims",
    "attackers_of": "partial — Jaccard / F1 over attacker conclusions",
    "enthymeme": "graded — goal gate (target justified, additions alone must not suffice) "
                 "times component-F1 against the actually-missing elements",
    "attack": "gated — the requested attack TYPE is a hard gate (wrong-type success scores 0); "
              "right type earns 0.3, +0.2 chain milestone, +0.5 for actually defeating C; from "
              "level {} rebut/undercut must be DERIVED through a rule".format(ATTACK_CHAIN_LEVEL),
    "counter_argumentation": "goal-state — any answer that makes the opposite claim justified "
                             "using DEFEASIBLE means only (added axioms / strict rules score 0)",
    "preference_construction": "goal-state — any preference set that makes the target justified",
    "formalization": "semantic equivalence — the engine compares the BEHAVIOR of your theory "
                     "to the gold theory (grounded status of every literal, on the base "
                     "theories and under one contrary-premise probe per atom); renamed or "
                     "reordered rules score 1.0, non-equivalent theories at most 0.25 shaped "
                     "credit (0.25 × component-F1)",
    "perturbation_prediction": "partial — fraction of queried statements whose predicted "
                               "post-change status matches the engine recomputed on the changed "
                               "theory (case-insensitive; only the [answer] block is read)",
    "robustness": "goal-state — the named premise(s) are removed and the engine recomputes the "
                  "target: ANY answer achieving the goal (defeat, or reinstatement at L4) scores "
                  "1.0; the set variant is Jaccard against the critical set; a pair must also be "
                  "MINIMAL (neither premise alone suffices) or it earns 0.3",
    "evidence_construction": "gated — justified under grounded semantics, standing on at "
                             "least the required number of pool statements; ANY wrong-stance "
                             "load-bearing premise zeroes the score",
}
GOAL_STATE = {"counter_argumentation", "preference_construction", "robustness"}


def task_meta(entry):
    """Human-readable (label, value) rows revealing the gold solution / key facts."""
    md = entry["metadata"]; k = md["kind"]; rows = []
    if k == "status_query":
        rows.append(("claims", ", ".join(md["queries"])))
        rows.append(("gold statuses",
                     "   ".join(f"{c}: {s}" for c, s in zip(md["queries"], md["gold_statuses"]))))
    elif k == "claim_identification":
        rows.append(("established (numbers)",
                     (", ".join(f"{n}:{md['universe'][n-1]}" for n in md["established"])) or "none"))
    elif k == "attackers_of":
        rows.append(("attackers", ", ".join(md.get("gold_sents", md["gold_lits"]))))
    elif k == "enthymeme":
        rows.append(("target", md["target"]))
        rows.append(("missing elements", str(md["n_missing"])))
        rows.append(("gold components", " ; ".join(md.get("gold_components", []))))
    elif k == "attack":
        rows.append(("conclusion (justified)", md["target"]))
        rows.append(("required attack", md["req"]
                     + (" (must be chained: derive it through a rule)"
                        if md.get("level", 2) >= ATTACK_CHAIN_LEVEL and md["req"] in ("rebut", "undercut")
                        else "")))
        wit = entry["answer"].replace("[answer]", "").replace("[/answer]", "").strip()
        rows.append(("a valid attacking argument", wit))
    elif k == "preference_construction":
        rows.append(("target (currently undecided)", md["target"]))
        wit = entry["answer"].replace("[answer]", "").replace("[/answer]", "").strip()
        rows.append(("a valid preference set", wit))
    elif k == "counter_argumentation":
        rows.append(("target (currently justified)", md["target"]))
        rows.append(("goal — make justified", contrary(md["target"])))
        ns, sup = md.get("n_supports"), md.get("supports")
        if ns is not None and sup is not None:
            rows.append((f'independent supports ({ns})', ", ".join(sup)))
    elif k == "perturbation_prediction":
        rows.append(("perturbation(s)", " ; ".join(md["perturbation"])))
        rows.append(("pre → post (queried)",
                     "   ".join(f"{c}: {pre}→{post}" for c, pre, post
                                in zip(md["queries"], md["pre_statuses"], md["post_statuses"]))))
        cg = md.get("changed_gold", {})
        rows.append(("gold (changed only)",
                     ("   ".join(f"#{int(i)+1}: {s}" for i, s in cg.items())) if cg else "none"))
    elif k == "ordering_sensitivity":
        rows.append(("claim", str(md["claim"])))
        rows.append(("last-link status", md["last_status"]))
        rows.append(("weakest-link status", md["weak_status"]))
    elif k == "robustness":
        rows.append(("target", md["target"]))
        rows.append(("variant / direction", f'{md["variant"]} / {md["direction"]}'))
        rows.append(("critical set (single-removal)", ", ".join(md["critical_set"]) or "none"))
        rows.append(("a valid answer", ", ".join(md["gold_names"])))
    elif k == "formalization":
        rows.append(("elements to formalize", str(md["n_elements"])))
        rows.append(("mode", md.get("mode", "")))
    elif k == "evidence_construction":
        atoms = md["atoms"]
        concl = atoms["c0"]["pos"] if md["stance"] == "support" else atoms["c0"]["neg"]
        rows.append(("stance → target", f'{md["stance"]} → {md["target"]}'))
        rows.append(("conclusion to reach", concl))
        rows.append(("pool size", str(len(md["pool"]))))
        rows.append(("pool (id: leaning — hidden from prompt)",
                     ", ".join(f'{i}:{atoms[i]["polarity"]}' for i in md["pool"])))
        if all("tier" in atoms[i] for i in md["pool"]):
            rows.append(("pool tiers (0=base … higher=derived; hidden from prompt)",
                         ", ".join(f'{i}:t{atoms[i]["tier"]}' for i in md["pool"])))
        if md.get("seeded_rules"):
            rows.append(("seeded opposing rules", ", ".join(md["seeded_rules"])))
    return rows


@app.route("/")
def index():
    return render_template_string(HTML, orderings=ORDERINGS, task_kinds=TASK_KINDS,
                                  levels=list(range(1, MAX_LEVEL + 1)), content_kinds=sorted(CONTENT_KINDS),
                                  answer_hint=ANSWER_HINT)


@app.route("/api/verify", methods=["POST"])
def api_verify():
    try:
        data = request.json or {}
        dsl = data.get("dsl", "")
        query = (data.get("query") or "").strip()
        ordering = data.get("ordering", "last_link_elitist")
        v = ASPICVerifier.from_dsl(dsl, ordering=ordering)
        out = v.verdict(query=query or None, check=True).to_dict()
        out["attack_graph"] = v.attack_graph()
        if query:
            out["explain"] = v.explain(query)
        return jsonify(out)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/generate", methods=["POST"])
def api_generate():
    try:
        data = request.json or {}
        kind = data["kind"]
        level = int(data.get("level", 2))
        with_content = bool(data.get("with_content", False)) and kind in CONTENT_KINDS
        seed_in = str(data.get("seed") or "").strip()
        seed = int(seed_in) if seed_in.isdigit() else random.randrange(1, 2_000_000_000)
        # optional explicit ordering for ordering-aware tasks; None -> per-item 50/50 coin.
        ord_in = str(data.get("ordering") or "").strip()
        ordering = ord_in if ord_in in ("last_link_elitist", "weakest_link_elitist") else None
        min_lv = TASK_MIN_LEVEL.get(kind, 1)
        if level < min_lv:
            return jsonify({"error": f"{kind} needs level ≥ {min_lv} "
                            f"(it requires preference-bearing conflicts, which start at "
                            f"level {min_lv}). Pick a higher level."}), 400
        try:
            entry = create_dataset(kind, seed=seed, size=1, level=level,
                                   with_content=with_content, ordering=ordering)[0]
        except RuntimeError as ex:
            return jsonify({"error": f"could not generate a valid {kind} item at level "
                            f"{level}: {ex}"}), 422
        task_id = uuid.uuid4().hex
        TASKS[task_id] = entry
        # The taskset prompt no longer pins a submission template; show the default
        # answer contract so a human solver sees the same prompt a run would send.
        return jsonify({"task_id": task_id, "kind": kind,
                        "prompt": template.apply(entry["question"]),
                        "level": level, "with_content": with_content, "seed": seed,
                        "answer_hint": ANSWER_HINT.get(kind, ""),
                        "grading_mode": GRADING_MODE.get(kind, ""),
                        "goal_state": kind in GOAL_STATE,
                        "reference": entry.get("answer", ""),
                        "meta": task_meta(entry)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/grade", methods=["POST"])
def api_grade():
    try:
        data = request.json or {}
        task_id = data.get("task_id")
        response = data.get("response", "")
        entry = TASKS.get(task_id)
        if entry is None:
            return jsonify({"error": "unknown or expired task_id"}), 404
        wrapped = False
        if not data.get("raw") and "[answer]" not in response.lower():
            response = "[answer]\n" + response.strip() + "\n[/answer]"
            wrapped = True
        score = score_answer(response, entry)
        kind = entry["metadata"]["kind"]
        ref = entry.get("answer", "")
        if kind in GOAL_STATE:
            detail = "goal verified by the solver — any valid solution scores 1.0"
        elif kind == "evidence_construction":
            detail = ("conclusion must hold under grounded semantics, on enough pool premises; "
                      "any wrong-stance load-bearing premise zeroes the score")
        else:
            detail = ("reference: " + ref) if ref else "checked by the solver"
        if wrapped:
            detail = "(answer auto-wrapped in [answer] tags)  " + detail
        return jsonify({"score": score, "success": score >= 0.999, "detail": detail})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ArgGym v2 Studio</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>
  body { font-family: ui-sans-serif, system-ui, sans-serif; }
  textarea, input, select { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  .chip { display:inline-flex; align-items:center; padding:2px 10px; border-radius:9999px;
          font-size:12px; font-weight:600; margin:2px; }
  .JUSTIFIED { background:#dcfce7; color:#166534; }
  .OVERRULED { background:#fee2e2; color:#991b1b; }
  .UNDECIDED { background:#fef9c3; color:#854d0e; }
  .UNSATISFIABLE { background:#f1f5f9; color:#475569; }
  .IN { background:#dcfce7; color:#166534; }
  .OUT { background:#fee2e2; color:#991b1b; }
  .UNDEC { background:#fef9c3; color:#854d0e; }
  pre { white-space: pre-wrap; }
  ::-webkit-scrollbar { width:8px; height:8px; }
  ::-webkit-scrollbar-thumb { background:#cbd5e1; border-radius:4px; }
</style>
</head>
<body class="bg-slate-100 text-slate-800 min-h-screen">

<header class="bg-indigo-700 text-white px-6 py-3 flex items-center justify-between shadow">
  <div>
    <h1 class="text-xl font-bold">ArgGym v2 Studio</h1>
    <p class="text-xs text-indigo-200">Grounded-semantics sandbox &amp; task lab</p>
  </div>
  <nav class="flex gap-2">
    <button id="tab-sandbox" onclick="showTab('sandbox')"
      class="px-4 py-1.5 rounded-md text-sm font-semibold bg-white text-indigo-700">Sandbox</button>
    <button id="tab-lab" onclick="showTab('lab')"
      class="px-4 py-1.5 rounded-md text-sm font-semibold bg-indigo-600 text-white">Task Lab</button>
  </nav>
</header>

<!-- ===================== SANDBOX ===================== -->
<main id="view-sandbox" class="p-6 grid grid-cols-1 lg:grid-cols-5 gap-6">
  <!-- input -->
  <section class="lg:col-span-2 bg-white rounded-xl shadow p-5 flex flex-col">
    <h2 class="font-bold mb-3">Build an argument</h2>
    <label class="text-xs font-semibold text-slate-500 mb-1">Target claim (optional)</label>
    <input id="query" placeholder="e.g. liable" class="border rounded-md p-2 mb-3 text-sm">
    <label class="text-xs font-semibold text-slate-500 mb-1">DSL</label>
    <textarea id="dsl" rows="10" class="border rounded-md p-3 text-sm resize-none flex-1"
      placeholder="[premise: signed]&#10;[premise: breached]&#10;[defeasible: signed AND breached => liable]"></textarea>
    <div class="flex items-center gap-2 mt-3">
      <select id="ordering" class="border rounded-md p-2 text-xs">
        {% for o in orderings %}<option value="{{o}}">{{o}}</option>{% endfor %}
      </select>
      <button onclick="runVerify()"
        class="flex-1 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-2 rounded-md">
        Check validity</button>
    </div>
    <div class="mt-4 bg-slate-50 border rounded-md p-3 text-xs leading-5">
      <strong>Syntax</strong><br>
      <code>[premise: X]</code> &nbsp; <code>[axiom: X]</code><br>
      <code>[defeasible: A AND B =&gt; C]</code><br>
      <code>[strict: A -&gt; C]</code><br>
      <code>[prefer_rule: d1 &gt; d2]</code> &nbsp; <code>[prefer_premise: P1 &gt; P2]</code><br>
      <span class="text-slate-500">Rules auto-name d1,d2,… / s1,s2,… (or label them, e.g.
      <code>[defeasible d1: A =&gt; C]</code>)  ·  <code>-X</code> negates  ·
      <code>-d1</code> undercuts rule d1</span>
    </div>
  </section>

  <!-- output -->
  <section class="lg:col-span-3 space-y-4">
    <div id="sb-target" class="bg-white rounded-xl shadow p-5 hidden">
      <div class="flex items-center justify-between">
        <h3 class="font-bold">Target</h3>
        <div id="sb-badges" class="flex gap-2"></div>
      </div>
      <div id="sb-target-body" class="mt-2"></div>
      <div id="sb-explain" class="mt-3 text-sm"></div>
    </div>

    <div class="bg-white rounded-xl shadow p-5">
      <h3 class="font-bold mb-2">Status of every claim</h3>
      <div id="sb-statusmap" class="text-sm"><span class="text-slate-400 italic">Run a check.</span></div>
    </div>

    <div id="sb-dropped" class="bg-rose-50 border border-rose-200 rounded-xl shadow p-5 hidden">
      <h3 class="font-bold text-rose-800 mb-2">Parse / rejected directives</h3>
      <ul id="sb-dropped-list" class="text-sm text-rose-700 list-disc pl-5 space-y-1"></ul>
    </div>

    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div class="bg-white rounded-xl shadow p-5">
        <h3 class="font-bold mb-2">Knowledge base</h3>
        <div id="sb-kb" class="text-xs space-y-2"></div>
      </div>
      <div class="bg-white rounded-xl shadow p-5">
        <h3 class="font-bold mb-2">Arguments &amp; defeats</h3>
        <div id="sb-graph" class="text-xs"></div>
      </div>
    </div>
  </section>
</main>

<!-- ===================== TASK LAB ===================== -->
<main id="view-lab" class="p-6 hidden grid grid-cols-1 lg:grid-cols-5 gap-6">
  <section class="lg:col-span-2 bg-white rounded-xl shadow p-5">
    <h2 class="font-bold mb-3">Generate a task</h2>
    <div class="flex items-center gap-3 mb-4 text-sm">
      <label class="font-semibold text-slate-500">Level</label>
      <select id="level" class="border rounded-md p-2">
        {% for l in levels %}<option value="{{l}}">{{l}}</option>{% endfor %}
      </select>
      <label class="flex items-center gap-1"><input type="checkbox" id="content"> content</label>
      <label class="font-semibold text-slate-500 ml-2">Seed</label>
      <input id="seed" placeholder="random" size="10" class="border rounded-md p-2 text-xs w-24">
    </div>
    <div class="grid grid-cols-2 gap-2">
      {% for kind, label, desc in task_kinds %}
      <button onclick="genTask('{{kind}}')"
        class="text-left border rounded-lg p-3 hover:bg-indigo-50 hover:border-indigo-300 transition">
        <div class="font-semibold text-sm text-indigo-700">{{label}}</div>
        <div class="text-xs text-slate-500">{{desc}}</div>
      </button>
      {% endfor %}
    </div>
    <p class="text-xs text-slate-400 mt-3">Content skin applies to:
      {{ content_kinds|join(', ') }}.</p>
  </section>

  <section class="lg:col-span-3 space-y-4">
    <div id="task-card" class="bg-white rounded-xl shadow p-5 hidden">
      <div class="flex items-center justify-between mb-2">
        <h3 id="task-kind" class="font-bold"></h3>
        <button onclick="regen()" class="text-xs text-indigo-600 hover:underline">↻ new question</button>
      </div>
      <pre id="task-prompt" class="bg-slate-50 border rounded-md p-3 text-sm"></pre>
      <div id="task-grademode" class="text-xs text-slate-500 mt-2"></div>
      <label class="text-xs font-semibold text-slate-500 mt-3 block" id="task-hint"></label>
      <textarea id="task-answer" rows="4"
        class="border rounded-md p-3 text-sm w-full resize-none mt-1"></textarea>
      <div class="text-xs text-slate-400 mt-1">Bare answers are auto-wrapped in
        <code>[answer]...[/answer]</code>; include the tags yourself to test the exact model-facing contract.</div>
      <button onclick="submitAnswer()"
        class="mt-2 bg-emerald-600 hover:bg-emerald-700 text-white font-semibold py-2 px-4 rounded-md">
        Submit answer</button>
      <div id="task-result" class="mt-3"></div>
      <div class="mt-3 border-t pt-2">
        <button id="task-ref-toggle" onclick="toggleRef()"
          class="text-xs text-slate-500 hover:text-indigo-600 hover:underline">▸ Reference &amp; details (peek)</button>
        <div id="task-ref" class="hidden mt-2"></div>
      </div>
    </div>
    <div id="task-empty" class="bg-white rounded-xl shadow p-10 text-center text-slate-400">
      Pick a task type on the left to generate a question.
    </div>
  </section>
</main>

<script>
function showTab(t){
  document.getElementById('view-sandbox').classList.toggle('hidden', t!=='sandbox');
  document.getElementById('view-lab').classList.toggle('hidden', t!=='lab');
  const a=document.getElementById('tab-sandbox'), b=document.getElementById('tab-lab');
  const on='bg-white text-indigo-700', off='bg-indigo-600 text-white';
  a.className='px-4 py-1.5 rounded-md text-sm font-semibold '+(t==='sandbox'?on:off);
  b.className='px-4 py-1.5 rounded-md text-sm font-semibold '+(t==='lab'?on:off);
}
function esc(s){return (s+'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function chip(text,cls){return `<span class="chip ${cls}">${esc(text)}</span>`;}

async function runVerify(){
  const body={dsl:document.getElementById('dsl').value,
              query:document.getElementById('query').value,
              ordering:document.getElementById('ordering').value};
  const r=await fetch('/api/verify',{method:'POST',headers:{'Content-Type':'application/json'},
            body:JSON.stringify(body)});
  const d=await r.json();
  if(d.error){alert('Error: '+d.error);return;}
  renderVerify(d);
}

function renderVerify(d){
  // badges
  const badges=document.getElementById('sb-badges');
  badges.innerHTML =
    chip(d.consistent?'consistent':'INCONSISTENT', d.consistent?'JUSTIFIED':'OVERRULED')+
    chip(d.invariants_ok?'invariants ok':'INVARIANTS FAIL', d.invariants_ok?'IN':'OUT')+
    chip(d.parse_ok?'parsed':'parse errors', d.parse_ok?'UNSATISFIABLE':'OVERRULED');

  // target
  const tcard=document.getElementById('sb-target');
  const tbody=document.getElementById('sb-target-body');
  const exp=document.getElementById('sb-explain');
  if(d.query){
    tcard.classList.remove('hidden');
    tbody.innerHTML = `<span class="text-sm">Claim <code>${esc(d.query)}</code> is </span>`+
                      chip(d.query_status, d.query_status)+
                      `<span class="text-sm ml-2 text-slate-500">(negation `+
                      `<code>${esc(contraryOf(d.query))}</code>: </span>`+
                      chip(d.query_negation_status, d.query_negation_status)+
                      `<span class="text-sm text-slate-500">)</span>`;
    if(d.explain){
      let h='<div class="font-semibold text-slate-600 mb-1">Supporting arguments</div>';
      if(d.explain.arguments.length===0){ h+='<span class="text-slate-400 italic">none</span>'; }
      d.explain.arguments.forEach(a=>{
        h+=`<div class="border rounded-md p-2 mb-1">`+chip(a.label,a.label)+
           `<code class="ml-1">${esc(a.id)}</code>`;
        if(a.defeated_by.length) h+=`<div class="text-rose-700 mt-1">defeated by: ${a.defeated_by.map(esc).join(', ')}</div>`;
        else if(a.attacked_by.length) h+=`<div class="text-amber-700 mt-1">attacked by (not winning): ${a.attacked_by.map(esc).join(', ')}</div>`;
        h+=`</div>`;
      });
      exp.innerHTML=h;
    } else exp.innerHTML='';
  } else { tcard.classList.add('hidden'); }

  // status map
  const sm=document.getElementById('sb-statusmap');
  const keys=Object.keys(d.status_map).sort();
  sm.innerHTML = keys.length ? keys.map(k=>chip(k+': '+d.status_map[k], d.status_map[k])).join('')
                            : '<span class="text-slate-400 italic">No derivable claims.</span>';

  // dropped
  const dc=document.getElementById('sb-dropped'), dl=document.getElementById('sb-dropped-list');
  if(d.dropped && d.dropped.length){ dc.classList.remove('hidden');
    dl.innerHTML=d.dropped.map(x=>`<li><strong>${esc(x.reason)}</strong>: <code>${esc(x.raw)}</code></li>`).join('');
  } else dc.classList.add('hidden');

  // kb
  const kb=d.kb; const k=document.getElementById('sb-kb');
  const rule=(r,a)=>`<code>[${esc(r[0])}] ${r[1].map(esc).join(' AND ')} ${a} ${esc(r[2])}</code>`;
  let kh='';
  if(kb.axioms.length) kh+=`<div><b>Axioms:</b> ${kb.axioms.map(x=>`<code>${esc(x)}</code>`).join(', ')}</div>`;
  if(kb.ordinary_premises.length) kh+=`<div><b>Premises:</b> ${kb.ordinary_premises.map(x=>`<code>${esc(x)}</code>`).join(', ')}</div>`;
  if(kb.strict_rules.length) kh+=`<div><b>Strict:</b><br>${kb.strict_rules.map(r=>rule(r,'-&gt;')).join('<br>')}</div>`;
  if(kb.defeasible_rules.length) kh+=`<div><b>Defeasible:</b><br>${kb.defeasible_rules.map(r=>rule(r,'=&gt;')).join('<br>')}</div>`;
  if(kb.rule_preferences.length) kh+=`<div><b>Rule prefs:</b> ${kb.rule_preferences.map(p=>`<code>${esc(p[0])} &gt; ${esc(p[1])}</code>`).join(', ')}</div>`;
  if(kb.premise_preferences.length) kh+=`<div><b>Premise prefs:</b> ${kb.premise_preferences.map(p=>`<code>${esc(p[0])} &gt; ${esc(p[1])}</code>`).join(', ')}</div>`;
  k.innerHTML = kh || '<span class="text-slate-400 italic">empty</span>';

  // graph
  const g=document.getElementById('sb-graph');
  const ag=d.attack_graph;
  let gh='<div class="mb-2">'+(ag.arguments.length?ag.arguments.map(a=>chip(a.id+' ⇒ '+a.conclusion, a.label)).join(''):'<span class="text-slate-400 italic">no arguments</span>')+'</div>';
  if(ag.defeats.length){ gh+='<div class="font-semibold text-slate-600">Defeats</div>';
    gh+=ag.defeats.map(e=>`<div><code>${esc(e.from)}</code> ⟶ <code>${esc(e.to)}</code></div>`).join('');
  }
  g.innerHTML=gh;
}
function contraryOf(s){return s.startsWith('-')?s.slice(1):'-'+s;}

// ---------- Task Lab ----------
let currentTask=null;
async function genTask(kind){
  const body={kind, level:parseInt(document.getElementById('level').value),
              with_content:document.getElementById('content').checked,
              seed:document.getElementById('seed').value};
  const r=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},
            body:JSON.stringify(body)});
  const d=await r.json();
  if(d.error){alert('Error: '+d.error);return;}
  currentTask=d;
  document.getElementById('task-empty').classList.add('hidden');
  document.getElementById('task-card').classList.remove('hidden');
  document.getElementById('task-kind').textContent=kind.replace(/_/g,' ')+'  (level '+d.level+(d.with_content?', content':'')+', seed '+d.seed+')';
  document.getElementById('task-prompt').textContent=d.prompt;
  document.getElementById('task-grademode').innerHTML =
    chip(d.goal_state?'goal-state · many valid answers':'graded answer','UNSATISFIABLE')+
    '<span class="ml-1">'+esc(d.grading_mode||'')+'</span>';
  document.getElementById('task-hint').textContent=d.answer_hint||'Your answer:';
  document.getElementById('task-answer').value='';
  document.getElementById('task-result').innerHTML='';
  // reference & details panel (collapsed)
  let rh='';
  (d.meta||[]).forEach(row=>{
    rh+=`<div class="flex gap-2 mb-0.5"><span class="text-slate-400 w-44 shrink-0">${esc(row[0])}</span>`+
        `<span class="text-slate-700 font-mono break-all">${esc(row[1])}</span></div>`;
  });
  if(d.reference){
    const lbl=d.goal_state?'one valid solution':'reference answer';
    rh+=`<div class="mt-2 pt-2 border-t"><div class="text-slate-400 mb-0.5">${lbl}</div>`+
        `<pre class="bg-slate-50 border rounded-md p-2 text-xs">${esc(d.reference)}</pre></div>`;
  }
  const refbox=document.getElementById('task-ref');
  refbox.innerHTML=rh||'<span class="text-slate-400 italic text-xs">no extra details</span>';
  refbox.classList.add('hidden');
  document.getElementById('task-ref-toggle').textContent='▸ Reference & details (peek)';
}
function toggleRef(){
  const box=document.getElementById('task-ref'), btn=document.getElementById('task-ref-toggle');
  const hidden=box.classList.toggle('hidden');
  btn.textContent=(hidden?'▸':'▾')+' Reference & details (peek)';
}
function regen(){ if(currentTask) genTask(currentTask.kind); }
async function submitAnswer(){
  if(!currentTask) return;
  const r=await fetch('/api/grade',{method:'POST',headers:{'Content-Type':'application/json'},
            body:JSON.stringify({task_id:currentTask.task_id, response:document.getElementById('task-answer').value})});
  const d=await r.json();
  const box=document.getElementById('task-result');
  if(d.error){box.innerHTML=chip('error: '+d.error,'OVERRULED');return;}
  const cls=d.success?'JUSTIFIED':'OVERRULED';
  box.innerHTML=chip(d.success?'CORRECT':'not yet','+'+cls)+
    chip('score '+d.score.toFixed(2), 'UNSATISFIABLE')+
    `<div class="text-xs text-slate-500 mt-1">${esc(d.detail)}</div>`;
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print("\n" + "=" * 52)
    print(" ARGGYM v2 STUDIO  ->  http://127.0.0.1:5000")
    print("=" * 52 + "\n")
    app.run(debug=True, port=5000, host="0.0.0.0")
