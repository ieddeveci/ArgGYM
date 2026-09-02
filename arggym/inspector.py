from __future__ import annotations

import traceback
import uuid

from flask import Flask, jsonify, render_template_string, request

from arggym.tasks.attack_defense import make_item, TASK
from arggym.tasks import perturbation as perturb
from arggym.tasks import counter_argument as counterarg
from arggym.tasks import preference_construction as prefcon
from arggym.tasks import claim_chain as claimchain
from arggym.tasks import defeat_diagnosis as defeatdiag
from arggym.tasks import formalization as formalize
from arggym.tasks import status_query as statusquery
from arggym.tasks import semantics_query as semquery
from arggym.core.scoring import score_item
from arggym.core.curriculum import ATTACK, DEFENCE, MIXED, LAST_LINK, WEAKEST_LINK, describe, spec_for

app = Flask(__name__)
ITEMS = {}

PERTURB = "perturbation"
COUNTERARG = "counter_argument"
COUNTERARG_STRICT = "counter_argument_strict"
PREFCON = "preference_construction"
CLAIMCHAIN = "claim_chain"
DEFEATDIAG = "defeat_diagnosis"
FORMALIZE = "formalization"
STATUSQUERY = "status_query"
SEMQUERY = "semantics_query"

MODES = [
    (SEMQUERY, "Semantics query",
     "State the status of each claim under the semantics named beside it."),
    (STATUSQUERY, "Status query",
     "State the status of each named claim."),
    (FORMALIZE, "Formalization",
     "Formalize a natural-language argumentation as an ASPIC+ theory."),
    (DEFEATDIAG, "Defeat diagnosis",
     "Say why a claim is not justified, and where its support fails."),
    (CLAIMCHAIN, "Claim chain",
     "Write the argumentation line that justifies a claim."),
    (PREFCON, "Preference construction",
     "Add preference directives to move claims to a required status."),
    (COUNTERARG_STRICT, "Counter-argument (strict allowed)",
     "Make a claim's contrary justified, with strict rules permitted."),
    (COUNTERARG, "Counter-argument",
     "Make a claim's contrary justified."),
    (PERTURB, "Perturbation",
     "Predict which claims change status when directives are added."),
    (ATTACK, "Attack",
     "Make a justified claim overruled."),
    (DEFENCE, "Defence",
     "Make an attacked claim justified."),
    (MIXED, "Attack-Defense",
     "Make one claim overruled and another justified at the same time."),
]

def build_graph(base_ops, ordering, highlight=None):
    from arggym.aspic.api import ASPICVerifier
    try:
        v = ASPICVerifier.from_operations(list(base_ops), ordering=ordering)
        sm = {k: str(x) for k, x in v.status_map().items()}
    except Exception:
        sm = {}
    rules = {o.name for o in base_ops
             if o.kind in ("defeasible", "strict") and getattr(o, "name", None)}
    facts, edges, attacks, prefs = [], [], [], []
    for o in base_ops:
        if o.kind in ("premise", "axiom"):
            facts.append({"lit": o.content, "kind": o.kind,
                          "status": sm.get(o.content, "?")})
        elif o.kind in ("defeasible", "strict"):
            c = o.consequent or ""
            tgt = c.lstrip("-")
            if c.startswith("-") and tgt in rules:
                attacks.append({"rule": o.name, "kind": "undercut",
                                "src": list(o.antecedents or ()), "target": tgt,
                                "strict": o.kind == "strict"})
            elif c.startswith("-"):
                attacks.append({"rule": o.name, "kind": "rebut",
                                "src": list(o.antecedents or ()), "target": tgt,
                                "strict": o.kind == "strict"})
            else:
                edges.append({"rule": o.name, "src": list(o.antecedents or ()),
                              "dst": c, "strict": o.kind == "strict",
                              "status": sm.get(c, "?")})
        else:
            prefs.append({"kind": o.kind, "stronger": o.stronger, "weaker": o.weaker})
    return {"facts": facts, "edges": edges, "attacks": attacks, "prefs": prefs,
            "statuses": sm, "highlight": highlight or []}


def ascii_graph(g, max_rows=400):
    st = {"JUSTIFIED": "+", "OVERRULED": "-", "UNDECIDED": "?", "UNSATISFIABLE": "x"}
    hi = set(g.get("highlight") or [])
    out = []
    out.append("FACTS")
    for f in sorted(g["facts"], key=lambda x: x["lit"]):
        mark = "AX" if f["kind"] == "axiom" else "  "
        out.append(f"  [{st.get(f['status'],'?')}] {mark} {f['lit']}")
    out.append("")
    out.append("SUPPORT  (rule: antecedents -> consequent)")
    by_dst = {}
    for e in g["edges"]:
        by_dst.setdefault(e["dst"], []).append(e)
    for dst in sorted(by_dst):
        s = g["statuses"].get(dst, "?")
        star = " *" if dst in hi else ""
        out.append(f"  [{st.get(s,'?')}] {dst}{star}")
        for e in by_dst[dst]:
            arrow = "==>" if e["strict"] else "-->"
            src = " AND ".join(e["src"])
            tag = " (strict)" if e["strict"] else ""
            out.append(f"          {src} {arrow} {dst}   via {e['rule']}{tag}")
    out.append("")
    out.append("ATTACKS")
    if not g["attacks"]:
        out.append("  (none)")
    for a in sorted(g["attacks"], key=lambda x: (x["kind"], x["target"])):
        src = " AND ".join(a["src"])
        word = "undercuts rule" if a["kind"] == "undercut" else "rebuts claim"
        out.append(f"  {a['rule']}: {src}  {word}  {a['target']}")
    out.append("")
    out.append("PREFERENCES")
    if not g["prefs"]:
        out.append("  (none)")
    for p in g["prefs"]:
        kind = "rule" if p["kind"] == "prefer_rule" else "premise"
        out.append(f"  {kind}: {p['stronger']} > {p['weaker']}")
    return "\n".join(out[:max_rows])


PAGE = """
<!doctype html><html><head><meta charset="utf-8"><title>ArgGYM</title>
<style>
 :root{--bg:#0f1115;--panel:#171a21;--line:#2a2f3a;--fg:#e6e6e6;--dim:#9aa4b2;
       --ok:#3fb950;--warn:#d29922;--bad:#f85149;--acc:#58a6ff}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);
   font:14px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace}
 header{padding:13px 18px;border-bottom:1px solid var(--line);display:flex;gap:14px;align-items:baseline;flex-wrap:wrap}
 h1{font-size:15px;margin:0;letter-spacing:.5px}
 .sub{color:var(--dim);font-size:12px}
 main{display:grid;grid-template-columns:350px 1fr;height:calc(100vh - 52px)}
 aside{border-right:1px solid var(--line);padding:14px;overflow:auto}
 section{padding:16px;overflow:auto}
 label{display:block;color:var(--dim);font-size:12px;margin:10px 0 4px}
 select,input,textarea,button{width:100%;background:#0b0d11;color:var(--fg);
   border:1px solid var(--line);border-radius:6px;padding:8px;font:inherit}
 button{background:var(--acc);color:#04121f;border:0;font-weight:700;cursor:pointer;margin-top:10px}
 button.sec{background:#222834;color:var(--fg)}
 .card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px;margin-bottom:13px}
 .card h3{margin:0 0 8px;font-size:11px;color:var(--dim);letter-spacing:1.2px;text-transform:uppercase}
 pre{margin:0;white-space:pre-wrap;font:inherit}
 .chip{display:inline-block;padding:2px 8px;border-radius:99px;font-size:12px;margin:2px 4px 2px 0}
 .ok{background:rgba(63,185,80,.15);color:var(--ok)} .bad{background:rgba(248,81,73,.15);color:var(--bad)}
 .warn{background:rgba(210,153,34,.15);color:var(--warn)} .neu{background:#222834;color:var(--dim)}
 table{width:100%;border-collapse:collapse} td,th{text-align:left;padding:3px 6px;font-size:13px}
 th{color:var(--dim);font-weight:400;border-bottom:1px solid var(--line)}
 .desc{color:var(--dim);font-size:12px;margin-top:6px}
 .row{display:flex;gap:8px} .row>*{flex:1}
</style></head><body>
<header><h1>ArgGYM</h1><span class="sub">A procedural, engine-verified benchmark for defeasible reasoning, built on the ASPIC+ structured argumentation framework.</span></header>
<main>
<aside>
  <label>Mode</label><select id="mode"></select>
  <div class="desc" id="modedesc"></div>
  <label>Level (1&ndash;15)</label><input id="level" type="number" min="1" max="15" value="8">
  <label>Ordering</label>
  <select id="ordering">
    <option value="last_link_elitist">last-link elitist</option>
    <option value="last_link_democratic">last-link democratic</option>
    <option value="weakest_link_elitist">weakest-link elitist</option>
    <option value="weakest_link_democratic">weakest-link democratic</option>
  </select>
  <label>Seed</label>
  <div class="row"><input id="seed" type="number" value="1">
    <button class="sec" onclick="rnd()">random</button></div>
  <button onclick="gen()">Generate item</button>
  <button class="sec" onclick="fillRef()">Fill reference answer</button>
  <div class="card" style="margin-top:16px"><h3>Level spec</h3><div class="desc" id="spec">&mdash;</div></div>
  <div class="card"><h3>Scoring</h3><div class="desc">
    score = 0.5 &times; success + 0.5 &times; efficiency.<br>
    efficiency = verified minimum / directives used.<br>
    Strict parsing: any malformed line scores 0.
  </div></div>
</aside>
<section>
  <div id="status"></div>
  <div class="card"><h3>Question Prompt</h3><pre id="prompt">generate an item&hellip;</pre></div>
  <div class="card"><h3>Answer</h3>
    <textarea id="ans" rows="7" placeholder="[answer]&#10;...&#10;[/answer]"></textarea>
    <button onclick="grade()">Grade</button><div id="grade"></div></div>
  <div class="card"><h3>Theory structure</h3>
    <div class="row"><button onclick="toggleGraph()">show / hide</button>
      <span class="desc">statuses: + justified, - overruled, ? undecided, x unsatisfiable</span></div>
    <pre id="graph" style="display:none;max-height:520px;overflow:auto">&mdash;</pre></div>
  <div class="card"><h3>Verified minimum</h3><div id="minv">&mdash;</div>
    <div class="desc">Established by a lower-bound proof, a witness, and a bounded exhaustive
    cross-check. Efficiency is scored against this, never against the reference answer.</div></div>
  <div class="card"><h3>Goals &mdash; engine status</h3><pre id="goals">&mdash;</pre></div>
  <div class="card"><h3>Metadata</h3><div id="meta">&mdash;</div></div>
  <div class="card"><h3>Reference answer</h3><pre id="ref">&mdash;</pre></div>
</section></main>
<script>
const MODES={{modes|tojson}};
let TOKEN=null, REF="";
function chip(t,c){return '<span class="chip '+c+'">'+t+'</span>'}
function init(){const s=document.getElementById('mode');
  MODES.forEach(m=>{const o=document.createElement('option');o.value=m[0];o.textContent=m[1];s.appendChild(o)});
  s.onchange=onMode; onMode()}
function onMode(){const k=document.getElementById('mode').value;
  document.getElementById('modedesc').textContent=MODES.find(m=>m[0]===k)[2]}
function rnd(){document.getElementById('seed').value=Math.floor(Math.random()*100000)}
async function gen(){
  const b={mode:document.getElementById('mode').value,level:+document.getElementById('level').value,
           ordering:document.getElementById('ordering').value,seed:+document.getElementById('seed').value};
  document.getElementById('status').innerHTML=chip('generating (minimality search)...','neu');
  const r=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});
  const d=await r.json();
  if(!d.ok){document.getElementById('status').innerHTML=chip(d.error,'bad');return}
  TOKEN=d.token; REF=d.reference;
  document.getElementById('prompt').textContent=d.prompt;
  document.getElementById('ref').textContent=d.reference+'\\n\\nreference scores: '+d.ref_score.toFixed(3);
  document.getElementById('spec').textContent=d.spec;
  document.getElementById('goals').textContent=d.goals.map(g=>
     g.claim+': now '+g.current+'  ->  required '+g.want).join('\\n');
  document.getElementById('minv').innerHTML=
     chip('minimum '+d.min_directives+' directives','ok')
     +chip('moves available '+(d.meta.moves_available||'-'),'neu')
     +chip(d.meta.lower_bound_proven?'lower bound PROVEN':'lower bound unproven',
           d.meta.lower_bound_proven?'ok':'bad')
     +chip(d.meta.searched_exhaustively?'exhaustive check ran':'proof only',
           d.meta.searched_exhaustively?'ok':'warn');
  let mt='<table>';
  Object.entries(d.meta).forEach(([k,v])=>{mt+='<tr><td class="desc">'+k+'</td><td>'+JSON.stringify(v)+'</td></tr>'});
  document.getElementById('meta').innerHTML=mt+'</table>';
  document.getElementById('status').innerHTML=chip(d.mode+' L'+d.level,'neu')
     +chip(d.ordering.replace(/_/g,' '),'neu')
     +chip('reference '+d.ref_score.toFixed(3), d.ref_score>=0.999?'ok':'bad');
  document.getElementById('grade').innerHTML='';
  const gEl=document.getElementById('graph');
  gEl.textContent='\u2014'; gEl.style.display='none';
  document.getElementById('ans').value='';
}
function fillRef(){document.getElementById('ans').value=REF}
function toggleGraph(){
  var el=document.getElementById('graph');
  if(el.style.display==='none'){
    el.style.display='block';
    if(!TOKEN){ el.textContent='generate an item first'; return; }
    el.textContent='building...';
    fetch('/api/graph',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({token:TOKEN})}).then(r=>r.json()).then(d=>{
        el.textContent = d.ok ? d.graph : ('error: '+d.error);
      });
  } else { el.style.display='none'; }
}
async function grade(){
  if(!TOKEN)return;
  const r=await fetch('/api/grade',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({token:TOKEN,answer:document.getElementById('ans').value})});
  const d=await r.json();
  if(!d.ok){document.getElementById('grade').innerHTML=chip(d.error,'bad');return}
  const g=d.result, dg=g.diagnostics||{};
  let h=chip('score '+g.score.toFixed(3), g.score>=0.999?'ok':(g.score>0?'warn':'bad'))
       +chip(g.reason, g.reason==='ok'?'ok':'bad');
  if(g.efficiency!==undefined) h+=chip('efficiency '+g.efficiency.toFixed(3),'neu');
  if(g.precision!==undefined) h+=chip('P '+g.precision.toFixed(2)+' R '+g.recall.toFixed(2),'neu');
  if(g.exact_match!==undefined) h+=chip(g.exact_match?'exact match':'not exact', g.exact_match?'ok':'warn');
  h+='<table><tr><th>diagnostic</th><th>value</th></tr>';
  ['n_lines','n_unparseable','n_used','minimum','n_predicted','n_gold',
   'n_survivor_included','n_wrong_status','n_contradicted','n_missed','n_spurious','n_quoted','n_gold'].forEach(k=>{
    h+='<tr><td class="desc">'+k+'</td><td>'+(dg[k]===undefined?'-':dg[k])+'</td></tr>'});
  if(dg.illegal&&dg.illegal.length) h+='<tr><td class="desc">illegal</td><td>'+dg.illegal.join(', ')+'</td></tr>';
  if(dg.contradicted&&dg.contradicted.length) h+='<tr><td class="desc">contradicted</td><td>'+dg.contradicted.join(', ')+'</td></tr>';
  if(dg.wrong&&dg.wrong.length) h+='<tr><td class="desc">wrong</td><td>'+dg.wrong.join('<br>')+'</td></tr>';
  if(dg.goals_met) dg.goals_met.forEach(x=>{
    h+='<tr><td class="desc">'+x.claim+'</td><td>'+(x.got===x.want?'✓ ':'✗ ')+x.got+' (want '+x.want+')</td></tr>'});
  document.getElementById('grade').innerHTML=h+'</table>';
}
init();
</script></body></html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE, modes=MODES)


@app.route("/api/generate", methods=["POST"])
def api_generate():
    b = request.get_json(force=True) or {}
    mode = b.get("mode", ATTACK)
    level = max(1, min(15, int(b.get("level", 8))))
    ordering = b.get("ordering", LAST_LINK)
    seed = int(b.get("seed", 1))
    if mode == SEMQUERY:
        try:
            sq = semquery.make_item(level, seed, ordering)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"})
        if sq is None:
            return jsonify({"ok": False, "error": "no item for this cell; try another seed"})
        token = uuid.uuid4().hex
        ITEMS[token] = ("semquery", sq)
        ref = semquery.score(sq.reference, sq)
        return jsonify({"ok": True, "token": token, "task": semquery.TASK, "mode": SEMQUERY,
                        "level": sq.level, "ordering": sq.ordering, "prompt": sq.prompt,
                        "reference": sq.reference, "ref_score": ref["score"],
                        "min_directives": sq.metadata["n_queries"],
                        "goals": [{"claim": f"{c} under {s}", "current": "-",
                                   "want": sq.gold[(c, s)]} for c, s in sq.queries],
                        "meta": sq.metadata,
                        "spec": f"L{sq.level} semantics_query {sq.ordering.replace('_link_', '-link ').replace('_', '-')} | "
                                f"{sq.metadata['n_queries']} queries over "
                                f"{len(sq.metadata['semantics'])} semantics | "
                                f"{sq.metadata['n_items']} directives, "
                                f"{sq.metadata['n_diverging_claims']} diverging claims"})
    if mode == STATUSQUERY:
        try:
            sq = statusquery.make_item(level, seed, ordering)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"})
        if sq is None:
            return jsonify({"ok": False, "error": "no item for this cell; try another seed"})
        token = uuid.uuid4().hex
        ITEMS[token] = ("statusquery", sq)
        ref = statusquery.score(sq.reference, sq)
        return jsonify({"ok": True, "token": token, "task": statusquery.TASK, "mode": STATUSQUERY,
                        "level": sq.level, "ordering": sq.ordering, "prompt": sq.prompt,
                        "reference": sq.reference, "ref_score": ref["score"],
                        "min_directives": sq.metadata["n_queried"],
                        "goals": [{"claim": c, "current": "-", "want": sq.gold[c]}
                                  for c in sq.queried],
                        "meta": sq.metadata,
                        "spec": f"L{sq.level} status_query {sq.ordering.replace('_link_', '-link ').replace('_', '-')} | "
                                f"{sq.metadata['n_queried']} claims asked of "
                                f"{sq.metadata['n_items']} directives | modal share "
                                f"{sq.metadata['modal_share']:.2f}, tower "
                                f"{sq.metadata['max_tower']}"})
    if mode == FORMALIZE:
        try:
            fi = formalize.make_item(level, seed, ordering)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"})
        if fi is None:
            return jsonify({"ok": False, "error": "no item for this cell; try another seed"})
        token = uuid.uuid4().hex
        ITEMS[token] = ("formalize", fi)
        ref = formalize.score(fi.reference, fi)
        return jsonify({"ok": True, "token": token, "task": formalize.TASK, "mode": FORMALIZE,
                        "level": fi.level, "ordering": fi.ordering, "prompt": fi.prompt,
                        "reference": fi.reference, "ref_score": ref["score"],
                        "min_directives": fi.metadata["n_directives"],
                        "goals": [{"claim": l, "current": "-", "want": fi.gold_status[l]}
                                  for l in fi.queried],
                        "meta": fi.metadata,
                        "spec": f"L{fi.level} formalization {fi.ordering.replace('_link_', '-link ').replace('_', '-')} | "
                                f"{fi.metadata['n_directives']} directives "
                                f"(target {fi.metadata['target_range']}) | "
                                f"{fi.metadata['n_axioms']} axioms, "
                                f"{fi.metadata['n_strict']} strict"})
    if mode == DEFEATDIAG:
        try:
            dd = defeatdiag.make_item(level, seed, ordering)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"})
        if dd is None:
            return jsonify({"ok": False, "error": "no item for this cell; try another seed"})
        token = uuid.uuid4().hex
        ITEMS[token] = ("defeatdiag", dd)
        ref = defeatdiag.score(dd.reference, dd)
        return jsonify({"ok": True, "token": token, "task": defeatdiag.TASK, "mode": DEFEATDIAG,
                        "level": dd.level, "ordering": dd.ordering, "prompt": dd.prompt,
                        "reference": dd.reference, "ref_score": ref["score"],
                        "min_directives": len(dd.diagnoses),
                        "goals": [{"claim": dd.claim, "current": dd.claim_status.lower(),
                                   "want": "diagnose every failure"}],
                        "meta": dd.metadata,
                        "spec": f"L{dd.level} defeat_diagnosis {dd.ordering.replace('_link_', '-link ').replace('_', '-')} | "
                                f"{dd.metadata['n_routes']} broken routes, depth "
                                f"{dd.metadata['chain_depth']} | tower {dd.metadata['tower']}, "
                                f"kinds {dd.metadata['kinds']}"})
    if mode == CLAIMCHAIN:
        try:
            cc = claimchain.make_item(level, seed, ordering)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"})
        if cc is None:
            return jsonify({"ok": False, "error": "no item for this cell; try another seed"})
        token = uuid.uuid4().hex
        ITEMS[token] = ("claimchain", cc)
        ref = claimchain.score(cc.reference, cc)
        return jsonify({"ok": True, "token": token, "task": claimchain.TASK, "mode": CLAIMCHAIN,
                        "level": cc.level, "ordering": cc.ordering, "prompt": cc.prompt,
                        "reference": cc.reference, "ref_score": ref["score"],
                        "min_directives": cc.metadata["line_length"],
                        "goals": [{"claim": cc.claim, "current": "justified",
                                   "want": "trace the line"}],
                        "meta": cc.metadata,
                        "spec": f"L{cc.level} claim_chain {cc.ordering.replace('_link_', '-link ').replace('_', '-')} | "
                                f"{cc.metadata['n_items']} items, depth "
                                f"{cc.metadata['chain_depth']}, line {cc.metadata['line_length']} | "
                                f"tower {cc.metadata['tower_height_true']}, decoys "
                                f"{cc.metadata['n_decoys']}"})
    if mode == PREFCON:
        try:
            pc = prefcon.make_item(level, seed, ordering)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"})
        if pc is None:
            return jsonify({"ok": False, "error": "no item for this cell; try another seed"})
        token = uuid.uuid4().hex
        ITEMS[token] = ("prefcon", pc)
        ref = score_item(pc.reference, pc.as_score_input())
        return jsonify({"ok": True, "token": token, "task": prefcon.TASK, "mode": PREFCON,
                        "level": pc.level, "ordering": pc.ordering, "prompt": pc.prompt,
                        "reference": pc.reference, "ref_score": ref["score"],
                        "min_directives": pc.min_directives, "goals": pc.goals,
                        "meta": pc.metadata,
                        "spec": f"L{pc.level} preference_construction "
                                f"{pc.ordering.replace('_link_', '-link ').replace('_', '-')} | "
                                f"{pc.metadata['n_claims']} claims, "
                                f"{pc.metadata['n_conflicts']} conflicts | shared="
                                f"{pc.metadata['shared_conflict']} theory_prefs="
                                f"{pc.metadata['theory_has_preferences']}"})
    if mode in (COUNTERARG, COUNTERARG_STRICT):
        try:
            cit = counterarg.make_item(level, seed, ordering,
                                       allow_strict=(mode == COUNTERARG_STRICT))
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"})
        if cit is None:
            return jsonify({"ok": False, "error": "no item for this cell - every candidate was "
                                                  "rejected; try another seed"})
        token = uuid.uuid4().hex
        ITEMS[token] = ("counterarg", cit)
        ref = score_item(cit.reference, counterarg.as_score_input(cit))
        return jsonify({"ok": True, "token": token, "task": counterarg.TASK, "mode": FORMALIZE,
                        "level": cit.level, "ordering": cit.ordering, "prompt": cit.prompt,
                        "reference": cit.reference, "ref_score": ref["score"],
                        "min_directives": cit.min_directives,
                        "goals": [{"claim": "-" + cit.target, "current": "not justified",
                                   "want": "JUSTIFIED"},
                                  {"claim": cit.target, "current": "justified",
                                   "want": "OVERRULED"}],
                        "meta": cit.metadata,
                        "spec": f"L{cit.level} counter_argument {cit.ordering.replace('_link_', '-link ').replace('_', '-')} | "
                                f"{cit.metadata['n_chains']} chains "
                                f"({cit.metadata['n_strict_final']} strict-final) depth "
                                f"{cit.metadata['chain_depth']} | strategy "
                                f"{cit.metadata['strategy']}"})
    if mode not in {m[0] for m in MODES}:
        return jsonify({"ok": False,
                        "error": f"unknown mode {mode!r}; choose one of "
                                 f"{', '.join(sorted(x[0] for x in MODES))}"})

    if mode == PERTURB:
        try:
            pit = perturb.make_item(level, seed, ordering)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"})
        if pit is None:
            return jsonify({"ok": False, "error": "no item for this cell - every candidate was "
                                                  "rejected (no change, no survivor, or the per-item "
                                                  "status balance failed); try another seed"})
        token = uuid.uuid4().hex
        ITEMS[token] = ("perturb", pit)
        ref = perturb.score(pit.reference(), pit)
        return jsonify({"ok": True, "token": token, "task": perturb.TASK, "mode": PERTURB,
                        "level": pit.level, "ordering": pit.ordering, "prompt": pit.prompt,
                        "reference": pit.reference(), "ref_score": ref["score"],
                        "min_directives": len(pit.gold),
                        "goals": [{"claim": k, "current": pit.before.get(k, "-"), "want": v}
                                  for k, v in sorted(pit.gold.items())],
                        "meta": pit.metadata,
                        "spec": f"L{pit.level} perturbation {pit.ordering.replace('_link_', '-link ').replace('_', '-')} | "
                                f"{pit.metadata['n_components']} components depth "
                                f"{pit.metadata['chain_depth']} | "
                                f"{pit.metadata['n_perturbations']} perturbations | "
                                f"{pit.metadata['n_changed']} changed, "
                                f"{pit.metadata['n_survivors']} survivors"})
    try:
        it = make_item(level, seed, ordering, mode)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"})
    if it is None:
        return jsonify({"ok": False, "error": "no item for this cell - every candidate was rejected "
                                              "(a gate failed, or the minimality search did not "
                                              "terminate); try another seed"})
    token = uuid.uuid4().hex
    ITEMS[token] = ("attackdef", it)
    ref = score_item(it.reference, it.as_score_input())
    return jsonify({"ok": True, "token": token, "task": TASK, "mode": it.mode,
                    "level": it.level, "ordering": it.ordering, "prompt": it.prompt,
                    "reference": it.reference, "ref_score": ref["score"],
                    "min_directives": it.min_directives, "goals": it.goals,
                    "meta": it.metadata, "spec": describe(spec_for(level, ordering, seed % 5))})


@app.route("/api/graph", methods=["POST"])
def api_graph():
    b = request.get_json(force=True)
    entry = ITEMS.get(b.get("token"))
    if entry is None:
        return jsonify({"ok": False, "error": "unknown token"})
    kind, it = entry
    ops = getattr(it, "base_ops", None) or getattr(it, "reference_ops", None) or []
    hi = []
    if hasattr(it, "claim"):
        hi = [it.claim]
    elif hasattr(it, "target"):
        hi = [it.target]
    elif hasattr(it, "queried"):
        hi = list(it.queried)
    elif hasattr(it, "goals"):
        hi = [g["claim"] for g in it.goals]
    try:
        return jsonify({"ok": True,
                        "graph": ascii_graph(build_graph(ops, it.ordering, hi))})
    except Exception as e:
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"})


@app.route("/api/grade", methods=["POST"])
def api_grade():
    b = request.get_json(force=True) or {}
    entry = ITEMS.get(b.get("token"))
    if entry is None:
        return jsonify({"ok": False, "error": "item expired - generate again"})
    kind, it = entry
    try:
        if kind == "perturb":
            res = perturb.score(b.get("answer", ""), it)
        elif kind == "counterarg":
            res = score_item(b.get("answer", ""), counterarg.as_score_input(it))
        elif kind == "prefcon":
            res = score_item(b.get("answer", ""), it.as_score_input())
        elif kind == "claimchain":
            res = claimchain.score(b.get("answer", ""), it)
        elif kind == "defeatdiag":
            res = defeatdiag.score(b.get("answer", ""), it)
        elif kind == "formalize":
            res = formalize.score(b.get("answer", ""), it)
        elif kind == "semquery":
            res = semquery.score(b.get("answer", ""), it)
        elif kind == "statusquery":
            res = statusquery.score(b.get("answer", ""), it)
        else:
            res = score_item(b.get("answer", ""), it.as_score_input())
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"})
    return jsonify({"ok": True, "result": res})


if __name__ == "__main__":
    print("ArgGYM inspector -> http://127.0.0.1:5000")
    app.run(debug=False, port=5000)
