# Related Benchmarks & Datasets for ArgGYM

Literature scan of prior work ArgGYM must position against and cite. ArgGYM = a
procedural, engine-verified benchmark + RL environment for **defeasible reasoning**,
built on **ASPIC+** structured argumentation under **grounded semantics**, with gold
answers computed symbolically (PyArg). Tasks: claim status classification
(justified/overruled/undecided/unsatisfiable), attacker identification
(rebut/undermine/undercut), attack/counter-argument/preference construction, NL→DSL
formalization, enthymeme completion, belief-revision/perturbation prediction.
Difficulty procedurally scaled; symbolic and NL ("content") variants.

All entries below were verified to exist via the paper-search tools (Semantic Scholar /
Google Scholar / arXiv). Where I saw an arXiv ID or DOI directly in tool output I record
it; otherwise I give the venue and a stable URL. IDs I did not see verbatim are marked
"(ID unverified)".

---

## Closest competitors (read these first)

These 4-5 works overlap most with ArgGYM. None combines *all* of: ASPIC+ structured
argumentation, grounded semantics, procedural difficulty scaling, symbolic engine
verification (PyArg), a full task suite beyond claim status, *and* an RL/self-play
environment. That intersection is ArgGYM's gap to fill.

1. **LLM-ASPIC+: A Neuro-Symbolic Framework for Defeasible Reasoning** (Fang, Li, Chen,
   Liao, 2025; IOS Press / ECAI-track ebook, https://ebooks.iospress.nl/volumearticle/75917).
   The single most similar work: explicitly couples LLMs with **ASPIC+** for defeasible
   reasoning over NL, evaluated on BoardgameQA (reports ~82-83% on BoardGameQA-2/3). It is
   a *method* (LLM extracts ASPIC+ structure, symbolic engine resolves conflicts), not a
   benchmark. ArgGYM differs by being a *benchmark + RL environment*: it procedurally
   generates ASPIC+ theories with engine-computed gold labels and a task suite (attacker
   typing, preference/attack construction, DSL formalization, enthymemes, perturbation),
   rather than proposing one pipeline. LLM-ASPIC+ is both the closest prior art and a
   natural baseline to run inside ArgGYM.

2. **DeFAb: A Verifiable Benchmark for Defeasible Abduction in Foundation Models** (Cooper
   & Velasquez, U. Colorado Boulder; arXiv 2606.18557, submitted 2026-06-17; MIT-licensed,
   dataset on HuggingFace). *Verified against the arXiv abstract page.* Task: **defeasible
   abduction** — construct hypotheses that explain anomalies by overriding defaults while
   preserving unrelated expectations; hypotheses pass polynomial-time checks for valid
   derivation, conservativity, minimality (ASP solver as oracle). Scale: 372k+ instances
   from 18 real knowledge bases (OpenCyc, YAGO, Wikidata, ConceptNet, UMLS), 3 difficulty
   levels, 4 surface renderings (rendering-robust = worst case over renderings). Headline:
   best frontier model 65% (drops to 23.5% rendering-robust) vs. 100% symbolic; also
   DeFAb-Hard and CONJURE variants. Training: **no training experiments are run** (all
   results are prompted evaluation of frozen models), but the release *ships training
   and self-play infrastructure* — SFT/DPO/GRPO/verifier-in-the-loop-RLHF scripts, an
   expert-iteration self-play trajectory collector ("runs are queued and no results are
   claimed"), and an adversarial-debate config — as a "pre-registered protocol" with
   results deferred to follow-on work (verified against the full arXiv HTML, 2026-07-18).
   Differs from ArgGYM: single-agent default-override abduction, **no argumentation** —
   no attack relations (rebut/undermine/undercut), no preferences/orderings, no grounded
   semantics or dialectical status, no counter-argument construction or belief-revision
   tasks, no RL environment. Direct competitor for the bare "engine-verified defeasible
   benchmark" claim — must cite and differentiate on the argumentation-theoretic task
   taxonomy; also a candidate transfer-eval for the training phase.

3. **BoardgameQA: A Dataset for Natural Language Reasoning with Contradictory Information**
   (Kazemi, Yuan, Bhatia, Kim et al., NeurIPS 2023 Datasets & Benchmarks;
   https://proceedings.neurips.cc/paper_files/paper/2023/hash/7adce80e86aa841490e6307109094de5-Abstract-Datasets_and_Benchmarks.html).
   The reference NL benchmark for **defeasible reasoning with conflicting rules and
   preferences** (rules have priorities; conflicts resolved defeasibly). Answers generated
   by a symbolic defeasible-reasoning procedure (engine-verified). Widely used as the
   defeasible-reasoning yardstick (LLM-ASPIC+ and others benchmark on it). ArgGYM differs
   by grounding in ASPIC+ (structured args, rebut/undermine/undercut, grounded semantics)
   with a richer task suite and RL environment, not just QA classification; BoardgameQA
   uses a fixed synthetic generator without ArgGYM's argumentation-theoretic task taxonomy.

4. **Enhancing Conflict Resolution in Language Models via Abstract Argumentation** (Z. Li,
   X. Fang, C. Chen, M. Li, B. Liao, 2025; Neurocomputing,
   https://www.sciencedirect.com/science/article/pii/S0925231225027651). Same group as
   LLM-ASPIC+; couples LLMs with **abstract argumentation** (argument acceptability
   computation) for conflict resolution, with CoT LLM baselines. Method paper, abstract
   (Dung) AFs rather than ASPIC+ structured argumentation; no procedural benchmark/RL. Key
   adjacent work confirming the "argumentation-as-conflict-resolution for LLMs" line.

5. **Reasoning Gym** (Stojanovski, Stanley, Sharratt, Jones, Adefioye, Kaddour, Köpf, 2025;
   arXiv 2505.24760). The template for the "gym" architecture ArgGYM adopts: 100+
   procedural data generators + verifiers (incl. a logic domain) with adjustable
   complexity and infinite data, usable for RLVR and evaluation. Not argumentation-
   specific and has no ASPIC+/defeasible semantics. ArgGYM is essentially a
   defeasible-argumentation instantiation of this paradigm with a symbolic (PyArg) oracle;
   Reasoning Gym is the methodological anchor and a likely integration target.

---

## 1. Defeasible / non-monotonic reasoning benchmarks for LLMs

- **δ-NLI — "Thinking Like a Skeptic: Defeasible Inference in Natural Language"** (Rudinger,
  Shwartz, Hwang, Bhagavatula, Forbes, Le Bras, Smith, Choi; EMNLP Findings 2020; DOI
  10.18653/v1/2020.findings-emnlp.418). Foundational NL defeasible-inference dataset:
  given premise+hypothesis, add an *update* that weakens (attenuator) or strengthens
  (intensifier) the inference; classification + generation tasks over SNLI/social-norms/
  ATOMIC. Verification: human-annotated crowd labels (not engine-verified). ArgGYM differs
  by symbolic engine verification, ASPIC+ structure (typed attacks, preferences, grounded
  status) and procedural scaling vs. δ-NLI's fixed crowdsourced single-step updates.

- **DeFAb: A Verifiable Benchmark for Defeasible Abduction in Foundation Models** (Cooper &
  Velasquez, 2026; arXiv 2606.18557). See closest-competitors #2. Verifiable, defeasible,
  symbolic→NL ladder, superiority relations.

- **Benchmarking Defeasible Reasoning with Large Language Models — Initial Experiments and
  Future Directions** (Tachmazidis, Batsakis, Antoniou, 2024; arXiv 2410.12509). Builds a
  benchmark from **defeasible logic** rule patterns (defeasible rules with superiority
  relations, teams of conflicting rules) and tests LLMs on them. Verification: symbolic
  (defeasible-logic ground truth). Very close in motivation (rule-based defeasible
  reasoning, superiority) but small/initial, defeasible-logic rather than ASPIC+, no
  grounded-semantics argument structure, attacker typing, or RL. Cite as prior
  engine-grounded defeasible-reasoning eval.

- **Belief-R / ΔR — "Belief Revision: The Adaptability of Large Language Models Reasoning"**
  (Wilie, Cahyawijaya, Ishii, He, Fung; EMNLP 2024; arXiv 2406.19764). Tests whether LLMs
  suppress/revise prior inferences when new premises arrive ("delta reasoning"), directly
  in the defeasible/non-monotonic tradition (inference retraction). Verification: designed
  premise sequences with defined expected revisions (rule-constructed, human-checked).
  Overlaps ArgGYM's belief-revision/perturbation task but as NL QA without ASPIC+ semantics
  or engine oracle. (Also relevant to §4.)

- **Generics and Default Reasoning in Large Language Models** (Kirkpatrick & Sterken, 2025;
  arXiv 2508.13718). Evaluates LLMs on generics ("birds fly") and default/defeasible
  inference, referencing δ-NLI-style deltas. Philosophy/linguistics-flavored eval, human/
  theoretical grounding, no symbolic engine or argumentation framework. Useful for framing
  the defeasibility motivation.

- **αNLI / ART — "Abductive Commonsense Reasoning"** (Bhagavatula, Le Bras, Malaviya et al.,
  2019; arXiv 1908.05739). ~20k-example abductive NLI/NLG benchmark (choose/generate the
  best explanatory hypothesis). Abductive, non-monotonic flavor; crowd-annotated (not
  engine-verified). Relevant background for ArgGYM's enthymeme-completion and
  counter-argument-construction tasks, which are abductive in spirit.

- **LLMs as ASP Programmers: Self-Correction Enables Task-Agnostic Nonmonotonic Reasoning**
  (Ishay & Lee, 2026; ACL Findings, https://aclanthology.org/2026.findings-acl.1151/).
  LLMs write Answer-Set-Programming encodings (with self-correction) to do **nonmonotonic
  reasoning**, tested on defeasible/contradictory NL incl. BoardgameQA. Neuro-symbolic
  *method* using ASP (stable-model semantics) rather than ASPIC+/grounded; no procedural
  benchmark or RL. Adjacent nonmonotonic-reasoning-via-symbolic-solver line to cite.

## 2. Computational argumentation + LLM benchmarks/evals

- **ArgLLMs — "Argumentative Large Language Models for Explainable and Contestable Claim
  Verification"** (Freedman, Dejl, Gorur, Yin, Rago, Toni; AAAI 2025; arXiv 2405.02079).
  LLMs construct argumentation frameworks with (quantitative bipolar) argumentation
  semantics to verify claims, yielding explainable/contestable decisions. Method +
  claim-verification eval; uses gradual/quantitative semantics, not ASPIC+ grounded
  status, and no procedural generator/RL. Central reference for "argumentation structure
  around LLMs."

- **Evaluating Uncertainty Quantification Methods in Argumentative LLMs** (Zhou, Dejl,
  Freedman, Chen, Rago, Toni; EMNLP Findings 2025;
  https://aclanthology.org/2025.findings-emnlp.1184.pdf). Extends ArgLLMs; studies UQ in
  argumentative LLMs on claim verification. Same family; incremental.

- **MArgE — "Meshing Argumentative Evidence from Multiple LLMs for Justifiable Claim
  Verification"** (Ng, Jiang, Freedman, Rago et al., 2025; arXiv 2508.02584). Aggregates
  argumentative evidence across multiple LLMs into a structured, grounded claim-verification
  process; evaluated on binary claim-verification datasets. Argumentation-structured
  ensemble method, not a procedural ASPIC+ benchmark.

- **Can Formal Argumentative Reasoning Enhance LLMs Performances?** (Castagna, Sassoon,
  Parsons, 2024; arXiv 2405.13036). Wraps LLMs in a formal argumentation pipeline and
  evaluates via **statement acceptability (justified/…)** — terminology overlapping
  ArgGYM's status classification. A method/eval showing formal argumentation can help LLM
  performance; no procedural generator, ASPIC+ engine oracle, or RL. Close on the
  "justified-status" framing.

- **Addressing the Right to Explanation… : Symbolic Constraints over LLMs** (Yu, Liga,
  Markovich, 2025; https://doi.org/10.1145/3769126.3769252). Prompts an LLM under **abstract
  argumentation + grounded semantics + discussion games** to explain and identify which
  argument to change to flip a decision — conceptually adjacent to ArgGYM's grounded-status
  and perturbation-prediction tasks, but a legal-explanation prompting study, not a
  benchmark. Notable for using *grounded semantics* explicitly.

- **A Unifying Framework for Learning Argumentation Semantics** (Mileva, Bikakis, D'Asaro,
  Law et al., 2023; arXiv 2310.12309). Learns to compute complete/grounded/preferred/stable
  extensions of abstract AFs (ILP-based), with empirical evaluation. Not LLM/NL, but the
  canonical "learn argumentation semantics" reference; ArgGYM's symbolic-variant tasks
  (compute grounded status of an AF) overlap its problem definition. Cite for AF-semantics
  computation baselines.

- **Large Language Models in Argument Mining: A Survey** (H. Li, Schlegel, Sun, Batista,
  2025; arXiv 2506.16383). Survey of LLMs for argument mining incl. benchmarks/evaluation
  protocols and gaps. Argument *mining* (extraction from text) is only tangential to
  ArgGYM's reasoning focus, but the survey is the right pointer for the mining-vs-reasoning
  boundary and for noting ArgGYM is *reasoning/semantics*, not mining.

- (Adjacent, non-core) **Evaluating LLM-driven summarisation of parliamentary debates with
  computational argumentation** (Cunningham, Greene, Cross, Rago, 2026; arXiv 2604.19331) —
  argument-mining + acceptability metrics for debate summarisation; peripheral.

## 3. Procedurally generated / engine-verified logic reasoning benchmarks

- **RuleTaker — "Transformers as Soft Reasoners over Language"** (Clark, Tafjord,
  Richardson; IJCAI 2020; DOI 10.24963/ijcai.2020/537). Seminal: synthetically generated
  NL rule theories, transformers emulate deduction; engine-generated gold via a symbolic
  reasoner over the theory (closed-world). ArgGYM shares the "procedural theory + symbolic
  gold" design but targets *defeasible* ASPIC+ (conflicts, attacks, preferences,
  under-determination) vs. RuleTaker's monotonic entailment; and adds argumentation-
  specific tasks + RL.

- **ProofWriter — "Generating Implications, Proofs, and Abductive Statements over Natural
  Language"** (Tafjord, Dalvi, Clark; ACL Findings 2021; DOI 10.18653/v1/2021.findings-
  acl.317). Extends RuleTaker with proof generation and abduction (find a missing fact to
  prove a conclusion) — the latter parallels ArgGYM's enthymeme completion. Engine-verified
  proofs. Still monotonic rule reasoning, not defeasible/argumentation.

- **"Pushing the Limits of Rule Reasoning in Transformers through Natural Language
  Satisfiability" (NLSat)** (Richardson & Sabharwal; AAAI 2022; DOI 10.1609/aaai.v36i10.21371).
  Generates *hard* NL-SAT instances via SAT-hardness sampling to systematically scale
  RuleTaker difficulty. Directly relevant to ArgGYM's procedural difficulty-scaling claim
  (how to make instances genuinely hard, not just uniformly sampled). Engine-verified
  (SAT). Propositional satisfiability, not defeasible argumentation.

- **FaiRR: Faithful and Robust Deductive Reasoning over Natural Language** (Sanyal, Singh,
  Ren; 2022; arXiv 2203.10261) and **RuleBERT** (Saeed, Ahmadi, Nakov, Papotti; EMNLP 2021;
  DOI 10.18653/v1/2021.emnlp-main.110) and **Neural Unification for Logic Reasoning over
  Natural Language** (Picco, Lam, Sbodio, López; EMNLP Findings 2021). RuleTaker-family
  methods/datasets: RuleBERT adds *soft* (probabilistic) Horn rules; FaiRR modularizes
  faithful reasoning; Neural Unifier does backward chaining. All monotonic NL rule
  reasoning; useful lineage for the "reasoning over NL theories" framing.

- **FOLIO: Natural Language Reasoning with First-Order Logic** (Han, Schoelkopf, Zhao et al.;
  2022; arXiv 2209.00840). Human-authored, logically complex NL reasoning with FOL
  annotations verified by an **FOL inference engine**; also an NL↔FOL translation dataset.
  Engine-verified but *human-authored* (not procedural) and monotonic FOL. FOLIO's NL↔FOL
  translation is the analogue of ArgGYM's NL→DSL formalization task — good comparison point
  (ArgGYM procedural + defeasible DSL vs. FOLIO curated + FOL).

- **LogicNLI: Diagnosing the First-Order Logical Reasoning Ability through LogicNLI** (Tian,
  Li, Chen, Xiao, He et al.; EMNLP 2021; https://aclanthology.org/2021.emnlp-main.303/).
  NLI-style diagnostic dataset that disentangles FOL reasoning from commonsense; procedurally
  constructed from FOL. Engine/rule-verified. Monotonic FOL, fixed difficulty; no
  defeasibility/argumentation.

- **PrOntoQA — "Language Models Are Greedy Reasoners: A Systematic Formal Analysis of
  Chain-of-Thought"** (Saparov & He; 2022; arXiv 2210.01240). Synthetic QA over generated
  ontologies with *parseable* CoT so each reasoning step is formally checkable against the
  proof. Procedural + step-level symbolic verification — methodologically close to ArgGYM's
  engine-verified stance. Monotonic ontological deduction, not defeasible.

- **LogicBench: Towards Systematic Evaluation of Logical Reasoning Ability of LLMs** (Parmar,
  Patel, Varshney et al.; ACL 2024; DOI 10.18653/v1/2024.acl-long.739). 25 inference rules
  over propositional, first-order, **and non-monotonic** logics as NL QA. The non-monotonic
  subset overlaps ArgGYM's remit, but LogicBench is a fixed single-inference-rule QA set
  with template/rule-verified labels, no argumentation structure, procedural scaling, or RL.
  Strong reference for "systematic logical-reasoning eval incl. non-monotonic."

- **ProverQA / ProverGen — "Large Language Models Meet Symbolic Provers for Logical Reasoning
  Evaluation"** (Qi, Ma, Li et al.; 2025; arXiv 2502.06563). LLM-generated + **symbolic-
  prover-verified** scalable FOL reasoning dataset with intermediate steps; also used for
  fine-tuning. The "LLM generator + symbolic verifier for a scalable reasoning benchmark"
  recipe is exactly ArgGYM's methodology, but for monotonic FOL, not defeasible ASPIC+.
  Close on *construction methodology*.

- **JustLogic: A Comprehensive Benchmark for Evaluating Deductive Reasoning in LLMs** (M.K.
  Chen, X. Zhang, D. Tao; 2025; arXiv 2501.14851). Programmatically constructed NL deductive
  reasoning benchmark controlling depth/complexity. Procedural + verified. Deductive/
  monotonic; contrast for difficulty-controlled deductive vs. ArgGYM's defeasible.

- **Enigmata: Scaling Logical Reasoning in LLMs with Synthetic Verifiable Puzzles** (J. Chen,
  He, Yuan et al.; 2025; arXiv 2505.19914). 36 puzzle tasks, each with a generator +
  rule-based verifier, built for **multi-task RLVR** with controllable difficulty. Same
  generator-verifier-RL architecture as ArgGYM but general puzzles, no argumentation/
  defeasibility. Good citation for the RLVR-with-verifiable-generators design.

- **QMFOL / QMFOLBench** (Zheng, Shi, Yu et al.; 2026; arXiv 2606.20227). Automated generation
  of monadic-FOL reasoning tasks with *quantifiable, controllable* complexity, NL-translated
  and prover-verified (round-trip). Procedural difficulty control + symbolic verification;
  monotonic FOL. Reinforces the procedural-controllable-difficulty design point.

### Gym-style RL reasoning environments

- **Reasoning Gym** (Stojanovski et al., 2025; arXiv 2505.24760). See closest-competitors #5.
- **RLVE: Scaling Up RL for Language Models with Adaptive Verifiable Environments** (Zeng,
  Ivison, Wang et al.; 2025; arXiv 2511.07317). RLVE-Gym: 400 procedural verifiable
  environments that **adapt difficulty to the policy** during training. Directly relevant to
  ArgGYM's "procedurally scaled difficulty for self-play RL" plan — cite for adaptive
  curriculum over verifiable environments. Not argumentation-specific.
- **Multilingual Reasoning Gym** (Dobler, Lehnerer, Scozzafava et al.; 2026; arXiv 2603.10793).
  Extends Reasoning Gym to 14 languages with parallel procedural generation. Relevant to
  ArgGYM's NL/"content" variants and any multilingual extension.

## 4. Belief revision / counterfactual-update evals for LLMs

- **Belief-R / ΔR** (Wilie et al., EMNLP 2024; arXiv 2406.19764). See §1 — the core LLM
  belief-revision benchmark (retract prior inference given new evidence). Closest to
  ArgGYM's belief-revision/perturbation-prediction task, but NL QA without ASPIC+ semantics
  or engine oracle; ArgGYM can compute the *exact* post-perturbation grounded status
  symbolically.

- **MQuAKE: Assessing Knowledge Editing in Language Models via Multi-Hop Questions** (Zhong,
  Wu, Manning, Potts, Chen; EMNLP 2023; https://aclanthology.org/2023.emnlp-main.971/).
  Counterfactual knowledge edits propagated through multi-hop questions (MQuAKE-CF / -T);
  the "ripple effect" of an update. Knowledge-editing / factual-update framing (edit a fact,
  check downstream answers), distinct from ArgGYM's argument-structure perturbation
  (add/remove an argument or preference and predict the new grounded extension). Cite as the
  knowledge-update analogue; ArgGYM's updates are structural and engine-verifiable.

- (Follow-on knowledge-editing works: PokeMQA (Gu et al., ACL 2024), retrieval-enhanced /
  error-accumulation editing (Shi et al. 2024; Guo et al. 2026). Peripheral — same
  factual-editing line, not argumentation.)

## 5. Argumentation frameworks combined with LLM training / self-play

- **LLM-ASPIC+** (Fang et al., 2025) and **Enhancing Conflict Resolution via Abstract
  Argumentation** (Li et al., 2025). See closest-competitors #1 and #4 — the two works that
  most directly wire argumentation frameworks into LLM reasoning. Both are inference-time
  neuro-symbolic methods; neither trains via RL/self-play on argumentation.

- **Training Language Models to Win Debates with Self-Play Improves Judge Accuracy** (Arnesen,
  Rein, Michael; 2024; arXiv 2409.16636). Self-play *debate* training (RL) improves the
  reliability of a judge — the clearest "argumentation-flavored self-play RL" precedent.
  Debate is informal two-agent persuasion, *not* structured ASPIC+ argumentation with
  grounded semantics; ArgGYM's proposed self-play would optimize against an engine-verified
  argumentation oracle rather than a learned judge. Key citation for the self-play-RL plan.

- **AI Debaters Are More Persuasive When Arguing in Alignment with Their Own Beliefs** (Carro,
  Mester, Nieto, Stanchi et al.; 2025; arXiv 2510.13912) and the **debate/scalable-oversight**
  line generally. Adjacent (persuasion/oversight), not structured argumentation semantics.
  Cite to distinguish ArgGYM (formal argumentation, engine-verified) from debate-style
  oversight (rhetorical, human/judge-verified).

- No work found that trains LLMs via **self-play RL directly on a structured (ASPIC+/ABA)
  argumentation engine with procedurally generated theories**. This is ArgGYM's most
  distinctive, apparently-unclaimed contribution as of this search. Caveat: **DeFAb has
  publicly pre-registered exactly this direction on its own (non-argumentation, defeasible-
  abduction) oracle** — self-play expert-iteration and DPO/GRPO scripts shipped, results
  "reserved for follow-on work." The results gap is still open, but the *idea* is now
  staked in print; ArgGYM's phase-2 claim should move quickly and differentiate on the
  argumentation substrate (dialectical self-play over ASPIC+ theories, not single-agent
  abduction).

---

## Notes on the gap ArgGYM fills

- **Engine-verified + defeasible + procedural** is occupied only sparsely: BoardgameQA
  (defeasible, symbolic gold, but fixed generator, QA-only, no ASPIC+ task taxonomy), DeFAb
  (verifiable defeasible abduction, but not ASPIC+/grounded or RL), and the initial
  Tachmazidis et al. defeasible-logic benchmark (small, defeasible-logic not ASPIC+).
- **ASPIC+/grounded semantics + LLMs** exists only as *methods* (LLM-ASPIC+, Li et al.
  conflict resolution, Yu et al. grounded-semantics prompting) — no *benchmark/environment*.
- **Task richness** (attacker typing rebut/undermine/undercut, attack/counter-argument/
  preference *construction*, NL→argumentation-DSL, enthymeme completion, perturbation
  prediction) appears in no single prior benchmark; existing defeasible/argumentation evals
  are almost all claim-status or claim-verification classification.
- **Self-play RL on a structured argumentation oracle**: no precedent found; debate-self-play
  (Arnesen et al.) is the nearest but uses a learned judge, not a symbolic argumentation
  engine.

Recommended must-cite core: LLM-ASPIC+, BoardgameQA, DeFAb, δ-NLI, RuleTaker/ProofWriter,
FOLIO, LogicBench, Reasoning Gym / RLVE, ArgLLMs, Belief-R, and (for methodology) ProverGen
/ NLSat / PrOntoQA.
