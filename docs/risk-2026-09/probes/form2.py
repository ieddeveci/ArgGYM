import sys, re, collections
sys.path.insert(0,'.')
import arggym
ORDS=["last_link_elitist","last_link_democratic","weakest_link_elitist","weakest_link_democratic"]
def q_atoms(q):
    # only the case text, up to "Under a correct formalization"
    body=q.split("Under a correct formalization")[0]
    return sorted(set(re.findall(r'\b[a-z]{2}\d\b', body)))
def premises_only(ref):
    return "\n".join(l for l in ref.splitlines() if l.startswith("[premise") or l.startswith("[axiom"))
res=[]
for L in range(1,16):
    for o in ORDS:
        e,_=arggym.create("formalization",L,o,size=1,seed=7000+L,profile="FULL").build_at(0)
        q=e["question"]
        atoms=q_atoms(q)
        flood="\n".join(f"[axiom: {a}]" for a in atoms)
        s1=arggym.score_row(flood,e)
        s2=arggym.score_row(premises_only(e["reference_answer"]),e)
        gs=e["metadata"]["gold"]["gold_status"]
        res.append((L,o,round(s1.score,4),round(s2.score,4),collections.Counter(gs.values())))
import statistics
print(f"{'L':>3s} {'axiom_flood(from question text only)':>36s} {'premises_only':>14s}")
for L in range(1,16):
    sub=[r for r in res if r[0]==L]
    print(f"{L:3d} {statistics.mean(r[2] for r in sub):36.4f} {statistics.mean(r[3] for r in sub):14.4f}")
print()
print("OVERALL axiom_flood mean:", round(statistics.mean(r[2] for r in res),4), " premises_only mean:", round(statistics.mean(r[3] for r in res),4), " n=",len(res))
allc=collections.Counter()
for r in res: allc.update(r[4])
print("formalization gold_status distribution over queried literals:", allc)
