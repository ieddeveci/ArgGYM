import sys, re, collections
sys.path.insert(0,'.')
import arggym
from arggym.core.scoring import parse_answer
from arggym.core.rows import score_input
from arggym.tasks import formalization as F

ORDS=["last_link_elitist","weakest_link_elitist"]
def rename(ref):
    # rename every model-chosen rule name q_N -> rN, including references -q_N
    return re.sub(r'\bq_(\d+)\b', r'rr\1', ref)

def all_strict_axiom(ref):
    out=[]
    for l in ref.splitlines():
        l=l.replace("[premise:","[axiom:")
        l=re.sub(r'\[defeasible (\w+):', r'[strict \1:', l).replace(" => "," -> ")
        out.append(l)
    return "\n".join(out)

def premises_only(ref):
    return "\n".join(l for l in ref.splitlines() if l.startswith("[premise") or l.startswith("[axiom"))

def axiom_flood(ref):
    # every literal mentioned, declared as both premise and axiom; no rules
    lits=set(re.findall(r'[-]?\b[a-z]{2}\d\b', ref))
    return "\n".join(f"[axiom: {x}]" for x in sorted(lits))

rows=[]
for L in (1,3,6,9,12,15):
    for o in ORDS:
        e,_=arggym.create("formalization",L,o,size=1,seed=7000+L,profile="FULL").build_at(0)
        ref=e["reference_answer"]
        r={}
        for name,fn in [("reference",lambda x:x),("renamed",rename),
                        ("all_strict_axiom",all_strict_axiom),
                        ("premises_only",premises_only),("axiom_flood",axiom_flood)]:
            a=fn(ref)
            sc=arggym.score_row(a,e)
            r[name]=(round(sc.score,4), round(sc.diagnostics.get("shape_f1") or 0,3),
                     round(sc.diagnostics.get("type_score") or 0,3),
                     round(sc.diagnostics.get("behavioural") or 0,3))
        rows.append((L,o,r))
        print(L,o)
        for k,v in r.items(): print(f"   {k:18s} total={v[0]:<8} shape={v[1]:<6} type={v[2]:<6} behav={v[3]}")
