import sys
sys.path.insert(0,'.')
import arggym
ORDS=["last_link_elitist","last_link_democratic","weakest_link_elitist","weakest_link_democratic"]
n=0; zeroed=0; ex=[]
for L in range(1,16):
    for o in ORDS:
        for s in (5000+L,6100+L):
            e,_=arggym.create("defeat_diagnosis",L,o,size=1,seed=s,profile="FULL").build_at(0)
            if e is None: continue
            q=e["question"]
            asked = "Append `survives_because: <rule>` only for a failure point" in q
            if asked: continue
            n+=1
            ref=e["reference_answer"].splitlines()
            base=arggym.score_row("\n".join(ref),e)
            if len(ref)<2: continue
            ref2=list(ref); ref2[1]=ref2[1]+"; survives_because: zz"
            r=arggym.score_row("\n".join(ref2),e)
            if r.score==0.0: zeroed+=1
            if len(ex)<6: ex.append((L,o,round(base.score,3),base.reason,round(r.score,3),r.reason))
print(f"defeat_diagnosis rows whose prompt does NOT mention survives_because: {n}")
print(f"of those, a perfect answer + one spurious `survives_because` scores 0.0: {zeroed}/{n}")
for x in ex: print("  ",x)
