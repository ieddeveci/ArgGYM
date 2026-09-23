import sys, collections
sys.path.insert(0,'.')
import arggym
ORDS=["last_link_elitist","last_link_democratic","weakest_link_elitist","weakest_link_democratic"]
trap=0; asked=0; tot=0; goldhas=0
rows=[]
for L in range(1,16):
    for o in ORDS:
        for s in (5000+L,6100+L):
            e,_=arggym.create("defeat_diagnosis",L,o,size=1,seed=s,profile="FULL").build_at(0)
            if e is None: continue
            tot+=1
            q=e["question"]
            a = "Append `survives_because: <rule>` only for a failure point" in q
            g = any(d.get("survives_because") for d in e["metadata"]["gold"]["diagnoses"])
            asked += a; goldhas += g
            if a and not g:
                trap+=1
                # demonstrate: reference + a spurious survives_because on the first record
                ref=e["reference_answer"].splitlines()
                if len(ref)>1:
                    ref[1]=ref[1]+"; survives_because: zz"
                    r=arggym.score_row("\n".join(ref),e)
                    rows.append((L,o,s,r.score,r.reason))
print("defeat_diagnosis rows:",tot)
print("prompt asks for survives_because:",asked)
print("gold actually has a survives_because:",goldhas)
print("TRAP rows (prompt asks, gold has none):",trap)
print("what one spurious survives_because does to an otherwise perfect answer:")
for r in rows[:10]: print("  ",r)
# also: what does a perfect answer WITHOUT the field score on trap rows
