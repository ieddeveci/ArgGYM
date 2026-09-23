import json, sys, traceback, time
sys.path.insert(0,'.')
import arggym
from evals.prompt import compose, Elicitation

TASKS = ["preference_construction","counter_argument","counter_argument_strict","attack",
         "defence","attack_defense","formalization","status_query","semantics_query",
         "perturbation","claim_chain","defeat_diagnosis"]
ORD = "last_link_elitist"
out = open("/tmp/rl_measure.jsonl","w")
t0=time.time()
for task in TASKS:
    for level in range(1,16):
        for seed in (101,202):
            rec = {"task":task,"level":level,"seed":seed}
            try:
                ds = arggym.create(task, level, ORD, size=1, seed=seed, profile="FULL")
                entry, report = ds.build_at(0)
                if entry is None:
                    rec["built"]=False; rec["reasons"]=dict(report.reasons)
                    out.write(json.dumps(rec)+"\n"); out.flush(); continue
                rec["built"]=True
                g = entry["metadata"].get("gold",{})
                rec["min_directives"] = g.get("min_directives")
                rec["minimality_proven"] = g.get("minimality_proven")
                ref = entry.get("reference_answer") or ""
                rec["ref"] = ref
                rec["ref_lines"] = len([l for l in ref.splitlines() if l.strip()])
                rec["ref_has_prefer"] = ("prefer_" in ref)
                q = entry["question"]
                rec["qchars"] = len(q)
                rec["q_v7_rule"] = ("Any word in your answer outside these lines scores the whole answer zero." in q)
                rec["q_AND"] = ("join the antecedents with the uppercase word AND" in q)
                # reference scores 1.0?
                try:
                    r = arggym.score_row(ref, entry)
                    rec["ref_score"]=r.score; rec["ref_success"]=r.success; rec["ref_reason"]=r.reason
                except Exception as e:
                    rec["ref_score"]=None; rec["ref_reason"]="EXC:"+type(e).__name__+":"+str(e)[:120]
                # empty answer
                try:
                    r0 = arggym.score_row("", entry); rec["empty_score"]=r0.score; rec["empty_reason"]=r0.reason
                except Exception as e:
                    rec["empty_score"]=None; rec["empty_reason"]="EXC:"+type(e).__name__
                # reference + one stray word
                try:
                    r1 = arggym.score_row(ref+"\ntherefore", entry); rec["stray_score"]=r1.score; rec["stray_reason"]=r1.reason
                except Exception as e:
                    rec["stray_score"]=None; rec["stray_reason"]="EXC:"+type(e).__name__
                # metadata extras
                st = entry["metadata"].get("state",{})
                rec["state_keys"]=sorted(st.keys())
                rec["gold_keys"]=sorted(g.keys())
                rec["n_clusters"]=entry["metadata"].get("n_clusters")
                rec["n_base_ops"]=len(entry["metadata"].get("base_ops") or [])
                rec["meta_extra"]={k:v for k,v in entry["metadata"].items() if isinstance(v,(int,float,str,bool)) and k not in ("level","seed","ordering")}
                if task=="semantics_query":
                    rec["gold_pairs"]=g.get("gold")
                    rec["state_queries"]=st.get("queries") or st.get("queried")
            except Exception as e:
                rec["error"]=traceback.format_exc()[-400:]
            out.write(json.dumps(rec)+"\n"); out.flush()
    print(task,"done",round(time.time()-t0,1),flush=True)
out.close()
print("ALLDONE",round(time.time()-t0,1))
