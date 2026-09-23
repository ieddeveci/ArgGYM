import sys, collections, time
sys.path.insert(0,'.')
import arggym
from arggym.core.freeze import minimum_unproven
ORDS=["last_link_elitist","weakest_link_democratic"]
CON=["preference_construction","counter_argument","counter_argument_strict","attack","defence","attack_defense"]
unproven=[]; n=0; bylevel=collections.Counter()
t0=time.time()
for t in CON:
    for L in range(1,16):
        for o in ORDS:
            e,_=arggym.create(t,L,o,size=1,seed=4400+L,profile="FULL").build_at(0)
            if e is None: continue
            n+=1
            if minimum_unproven(e):
                unproven.append((t,L,o,e["metadata"]["gold"]["min_directives"])); bylevel[L]+=1
    print(f"  {t} n={n} unproven={len(unproven)} {time.time()-t0:.0f}s",flush=True)
print("TOTAL construction rows:",n)
print("minimality_unproven:",len(unproven),f"({100*len(unproven)/max(n,1):.1f}%)")
print("  by level:",dict(sorted(bylevel.items())))
print("  examples:",unproven[:20])
