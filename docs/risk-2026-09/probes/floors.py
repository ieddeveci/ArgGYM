import json, sys
sys.path.insert(0,'.')
import arggym
ORDS=["last_link_elitist","last_link_democratic","weakest_link_elitist","weakest_link_democratic"]
TASKS=["status_query","semantics_query","perturbation","defeat_diagnosis","attack_defense","defence","counter_argument","attack","preference_construction","counter_argument_strict"]
def neg(a): return a[1:] if a.startswith("-") else "-"+a
out=open("/tmp/floors.jsonl","w")
for task in TASKS:
    for L in range(1,16):
        for o in ORDS:
            e=None
            try:
                e,_=arggym.create(task,L,o,size=1,seed=7000+L,profile="FULL").build_at(0)
            except Exception as ex:
                out.write(json.dumps({"task":task,"level":L,"ord":o,"err":str(ex)[:100]})+"\n"); continue
            if e is None:
                out.write(json.dumps({"task":task,"level":L,"ord":o,"err":"nobuild"})+"\n"); continue
            rec={"task":task,"level":L,"ord":o}
            st=e["metadata"].get("state",{})
            ref=e["reference_answer"]
            cands={}
            if task=="status_query":
                keys=[f"{c}: " for c in st["queried"]]
            elif task=="semantics_query":
                keys=[f"{c} under {s}: " for c,s in st["queries"]]
            elif task=="perturbation":
                keys=[l.split(":")[0]+": " for l in ref.splitlines() if l.strip()]
            else:
                keys=None
            if keys is not None:
                rec["n_items"]=len(keys)
                for lab in ["justified","overruled","undecided"]:
                    a="\n".join(k+lab for k in keys)
                    cands[lab]=arggym.score_row(a,e).score
            if task=="defeat_diagnosis":
                rec["gold_status"]=e["metadata"]["gold"]["claim_status"]
                for lab in ["justified","overruled","undecided"]:
                    cands["status:"+lab]=arggym.score_row(f"status: {lab}",e).score
            if task in ("attack_defense","defence","counter_argument","attack","preference_construction","counter_argument_strict"):
                goals=e["metadata"]["state"]["goals"]
                prem=[x["content"] for x in e["metadata"]["base_ops"] if x.get("kind")=="premise"]
                rec["min_directives"]=e["metadata"]["gold"].get("min_directives")
                if prem:
                    a="\n".join(f"[defeasible zz{i}: {prem[i%len(prem)]} => {neg(g['claim'])}]" for i,g in enumerate(goals))
                    r=arggym.score_row(a,e); cands["deadlock"]=r.score
                # empty
                cands["empty"]=arggym.score_row("",e).score
            rec["cands"]=cands
            rec["best"]=max(cands.values()) if cands else None
            out.write(json.dumps(rec)+"\n"); out.flush()
    print(task,"done",flush=True)
out.close(); print("ALLDONE")
