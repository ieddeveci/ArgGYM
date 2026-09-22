import json, sys, re, itertools
sys.path.insert(0,'.')
import arggym

ORD="last_link_elitist"
def build(task,level,seed):
    ds=arggym.create(task,level,ORD,size=1,seed=seed,profile="FULL")
    e,_=ds.build_at(0); return e

def ops(e): return e["metadata"]["base_ops"]

def premises(e):
    return [o["content"] for o in ops(e) if o.get("kind")=="premise"]

def neg(a): return a[1:] if a.startswith("-") else "-"+a

# ---------- construction: lazy deadlock ----------
def deadlock_answer(e, nmax=None):
    goals=e["metadata"]["state"]["goals"]
    prem=premises(e)
    if not prem: return None
    lines=[]
    for i,g in enumerate(goals):
        c=g["claim"]
        # rule from an arbitrary premise to the contrary of the goal claim
        src = prem[i % len(prem)]
        lines.append(f"[defeasible zz{i}: {src} => {neg(c)}]")
    return "\n".join(lines)

def score(ans,e):
    try:
        r=arggym.score_row(ans,e); return r.score, r.reason, r.diagnostics.get("progress"), r.diagnostics.get("deadlock_not_defeat")
    except Exception as ex:
        return None, "EXC:"+type(ex).__name__+":"+str(ex)[:80], None, None

CONSTR=["preference_construction","counter_argument","counter_argument_strict","attack","defence","attack_defense"]
print("=== lazy-deadlock exploit on construction tasks (one line per goal, no preferences) ===")
for t in CONSTR:
    out=[]
    for L in (1,3,6,9,12,15):
        e=build(t,L,101)
        a=deadlock_answer(e)
        s=score(a,e)
        out.append((L, s[0], s[1][:28]))
    print(f"{t:26s}", out)

# ---------- module tasks: constant-label ----------
print()
print("=== constant-label exploits on label tasks ===")
def queried_lines(e, task):
    st=e["metadata"].get("state",{})
    if task=="status_query":
        return [f"{c}: {{L}}" for c in st["queried"]]
    if task=="semantics_query":
        return [f"{c} under {s}: {{L}}" for c,s in st["queries"]]
    if task=="perturbation":
        # answer lists every claim; take them from the reference shape
        ref=e["reference_answer"]
        return [l.split(":")[0]+": {L}" for l in ref.splitlines() if l.strip()]
    return []

for task in ["status_query","semantics_query","perturbation"]:
    for L in (1,3,6,9,12,15):
        e=build(task,L,101)
        tmpl=queried_lines(e,task)
        best=[]
        for lab in ["justified","overruled","undecided"]:
            a="\n".join(x.format(L=lab) for x in tmpl)
            s=score(a,e)
            best.append((lab, s[0]))
        # also: copy every claim as undecided but keep 'no stable extension' out
        print(f"{task:18s} L{L:<3d} n={len(tmpl):3d}", best)
    print()

print("=== defeat_diagnosis: status-only answers ===")
for L in (1,3,6,9,12,15):
    e=build("defeat_diagnosis",L,101)
    for lab in ["justified","overruled","undecided"]:
        print(f"  L{L:<3d} status:{lab:11s}", score(f"status: {lab}", e)[:2])

print()
print("=== claim_chain: empty / single premise ===")
for L in (1,3,6,9,12,15):
    e=build("claim_chain",L,101)
    print(f"  L{L:<3d} ref_lines", len(e['reference_answer'].splitlines()), "| premise-only:", score("[premise: "+premises(e)[0]+"]", e)[:2] if premises(e) else None)
