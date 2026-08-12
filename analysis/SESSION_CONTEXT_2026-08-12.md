# Session context — 2026-08-12: npj PD submission set, reference audit, and publication

Follows `SESSION_CONTEXT_2026-08-09.md`. No new analysis and no reported number changed. This session turned a drafted manuscript into a submission package and published the code.

---

## 1. Headline

**A near-miss is the most important thing here.** One command short of pushing the repository public, five files in `data/processed/` turned out to hold participant-level PPMI data, including a 19,450 × 212 table keyed by `PATNO` and `EVENT_ID` carrying `Death_Status`, `age_death` and `Death_Date`. Publishing it would have breached the data use agreement and contradicted the manuscript's own Data availability statement. Details in §5.

Otherwise: the manuscript was restructured to npj's actual requirements, all 50 references were verified against PubMed, and four submission documents now exist.

---

## 2. npj PD requirements, read from the journal rather than inferred

Both guideline pages redirect through Nature's SSO; the content is reachable after following the chain twice.

**The relief clause:** *"Manuscripts submitted to npj Parkinson's Disease do not need to adhere to our formatting requirements at the point of initial submission; formatting requirements only apply at the time of acceptance."* And **no word limit is specified for Articles**, so the 3,802-word main text is not a constraint.

**What did bite** is structure. npj's order is main text → **Data availability → References → Author contributions → Acknowledgements → Competing interests**. The draft had References last and Acknowledgements before Author contributions, so three sections moved. A **Code availability** section was also split out; npj requires one where custom code is central, and the draft had folded code into Data availability where a checklist would not find it.

**Two obligations still outstanding:** npj states that *"use of an LLM should be properly documented in the Methods section"* — the manuscript has no such statement — and a Reporting Summary is required for Life Sciences articles.

---

## 3. Reference audit — all 50 verified

Every PMID was resolved against PubMed and checked on title, journal, year and DOI. **No fabrications, no orphans, no misdirected citations.** Six apparent discrepancies were all artifacts: three were a parser leaving author names in the title field; one batch retrieval silently dropped five records, all of which exist when queried individually; and PMID 22534427 shows 2012 in PubMed's `publication_date` because that is the electronic publication, while the print issue is `23(1):91-106` in 2014, exactly as cited.

Two methods that had been named without a citation now have one, both verified rather than recalled:

- NULISA — Feng et al., *Nat Commun* 2023;14:7238, PMID 37945559
- SuSiE-coloc — Wallace, *PLoS Genet* 2021;17(9):e1009440, PMID 34587156 (single author, so no *et al.*)

Adding them mid-document broke first-appearance order, so the renumbering pass ran again. Now 50 references, sequential.

**Worth remembering:** on 08-09 three citations pointed at the *wrong papers* while every PMID was valid. A verified reference list is not necessarily a correctly-pointed one; those are two different checks.

---

## 4. Submission set

All four are **generated from `MANUSCRIPT.md` by scripts** — regenerate after any edit rather than editing the .docx, or the two drift. Copies live in `C:\Users\Kartic Mishra\Downloads\npjPD_submission\` because Word cannot reliably open files on `E:`.

| file | contents |
|---|---|
| `Manuscript_npjPD.docx` | 12 sections in npj order, 50 references, six figures embedded, continuous line numbering |
| `Cover_Letter_npjPD.docx` | 632 words, five suggested referees, no exclusions |
| `Supplementary_npjPD.docx` | Supplementary Notes 1 (deviations) and 2 (integrity sweep), 7 tables |
| `Reporting_Summary_answers_npjPD.docx` | 12 answered items; the form itself is a journal-issued fillable PDF that cannot be generated outside the submission system |

**Authorship**, as decided by the author: five authors, Kartic and Shivani Devi joint first (`#`), then Sanggyun Yi, Yeeun Seo, and Tae-Sik Park corresponding. Kartic is a mononym. Contributions for S.Y. and Y.S. were drafted by inference from work the project actually contains and **need confirming with them**. Funding: Gachon University research fund 202307940001 to Kartic, with Shivani's award left as a marked field.

Two front-matter defects were repaired: the author block used `1` and `#` as superscript markers with neither footnote present, and Word had autocorrected "Mendelian randomisation" to the American spelling in the keywords while the body uses the British form five times. The only remaining American spellings are inside three reference titles, which are correct as published.

---

## 5. Publishing the code — and what nearly went with it

The old repository, `kartic03/DBS-BBB-Multimodal-Fusion`, is **deliberately untouched**. Its single commit is the rejected DBS/LFP fusion study, and renaming it to match this paper would have left commit `3a3ff9b` reachable by SHA under a title that contradicts the manuscript.

A new repository was created instead: **github.com/kartic03/ppmi-csf-cognitive-decline**, public, 203 files.

**It could not simply take the local history.** The local repo's root commit `43b3978` still contains `train_fusion.py`, `lfp_preprocessing.py` and `data_fusion.py` — pushing 52 commits into a "fresh" repo would have carried the fabricated study in as commit one. The repo is a clean snapshot instead.

### The participant-level data catch

Five files in `data/processed/` held real participant data:

| file | what it was |
|---|---|
| `phase1/curated_cut.parquet` | 19,450 × 212, `PATNO` + `EVENT_ID`, with `SITE`, `study_status`, `Death_Status`, `age_death`, `Death_Date` |
| `phase2/outcome.parquet` | 816 per-participant slopes keyed by `subject_id` |
| `phase1/outcome.csv` | 1,597 rows, PATNO + UPDRS slopes + exact ages such as `67.38630136986302` |
| `phase1/qalbumin.csv` | 537 rows, PATNO + CSF/plasma albumin |
| `phase1/setA_patnos.csv` | 290 PATNOs |

**The parquets nearly escaped.** A text scan for `PATNO` reported clean on both, because compressed parquet does not expose column names to `grep` or `strings`. They were only found by loading the schema with pandas. All five were removed and the commit rebuilt from scratch, so **they never entered the pushed history**, and `.gitignore` now blocks the paths by name with a comment saying why.

A consequence: `src/phase1` and `src/phase2` regenerate these intermediates from raw PPMI rather than reading them from the repo. A user who clones can reproduce the **figures** immediately and the **analysis** only after obtaining PPMI access. This is correct behaviour and the README says so.

Also excluded: four `.joblib` frozen model bundles under `analysis/B5/`, which the project's own nested `.gitignore` excludes and which a blanket `git add -f` had briefly pulled in.

### History shape

The first push used one commit for all 203 files. That violates the project's commit convention — one commit per top-level path, message `<path> updated` — which exists because GitHub's file list shows each row's last-touching commit message. History was rebuilt as nine commits and force-pushed.

---

## 6. The repository was published wrong twice before it was right

Recorded because both mistakes were mine and both are the kind that repeat.

**First push: one commit for 203 files.** The project's commit convention is one commit per top-level path, message `<path> updated`, because GitHub's file list shows each row's last-touching commit message. I reasoned that an initial snapshot was a different case. It is not — a fresh repo is exactly where that column gets set for every row at once. Rebuilt as nine commits and force-pushed.

**Second push: 46 internal markdown files went public.** `revision/` was copied wholesale because the manuscript cites two documents inside it. That published the JNE rejection notes for the fabricated study, the simulated five-reviewer panel and its response, four session contexts, the 57 KB project plan, and every design spec — and **five files stating that a leaked Groq key still needed rotating**, which advertises an unrotated credential to anyone reading. No key value was ever exposed, but the disclosure was.

Now 140 files with **four** markdown: the README plus the three a reader needs — `DEVIATIONS_TABLE.md` (cited as Supplementary Note 1), `ARTIFACT_INTEGRITY_SWEEP.md` (Note 2) and `captions.md`. `B5/` and `figures/prism/` were dropped as internal. History was rebuilt each time, so nothing stripped exists in any reachable commit.

**Stripping files broke pointers in the files that were kept.** `DEVIATIONS_TABLE.md` — a document the paper cites — referenced `B5/OSF_PREREG_SUBMISSION.md`, its addendum, and `SESSION_CONTEXT_2026-08-04.md`, all deleted. Fixed at source and the Supplementary docx regenerated, since it is built from that file. Fifteen figure scripts also carried stale `revision/figures/` paths in their docstrings, and `fig1` cited a `checks/` note that exists in no copy of the repo.

**The folder is now `analysis/` in both places.** Renaming on E: rewrote 31 files; the trap is that `revision/audit-and-controls` is the git *branch* name and appears four times in the docs, so it was sentinel-protected through the replace. Figures verified to still render afterwards, and the submission builders were repointed.

**Reproduction was smoke-tested, not assumed:** taking only what the repo ships — `analysis/figures/`, `data/processed/` and the two lockfiles — the figures rebuild and pass their asserts, so the README's claim holds from a clean clone.

### The other six repositories are clean

All of `kartic03`'s public repos were cloned with full history and scanned on 2026-08-12: `eeg-fm-shortcut-audit`, `Toxbench`, `BBB-Trans-AI`, `DBS-Candidacy-Screening`, `DBS-BBB-Multimodal-Fusion`, `RATAN-PBind`. **No credentials in any tree or history** — the only two hits are `api_key: "YOUR_GROQ_API_KEY"` template placeholders in two config files, with zero `gsk_` matches anywhere. **No internal files**: each carries one README, plus legitimate content markdown in Toxbench (`tables/`) and eeg-fm-shortcut-audit (`checkpoints_manifest.md`). The exposure was confined to the repo created in this session.

`DBS-BBB-Multimodal-Fusion` is deliberately untouched; its single commit is the rejected fusion study.

## 7. The writing pass, and a selective report it uncovered

Researched how these documents should be written, then rebuilt them against that. Guidance is recorded in the `paper-craft-guidance` memory; the findings specific to this paper are below.

### A selective report, found in an artifact's own caveats field

`phase2/hlavnicka_head2head.json` carries a `caveats` list stating that the fainting item **"is NOT identifiable from the PPMI data dictionary; the reported H4 arm is the most favourable of four"**. Four candidate autonomic items had been tested — 0.576, 0.582, 0.627, **0.649** — and the manuscript reported 0.649, the maximum, as "their four items reached AUC 0.649", with no disclosure anywhere.

The spread across candidates (0.073) is nearly three times the difference the parsimony comparison turns on (0.676 − 0.649 = 0.027). Pick SCAU13 instead and the questionnaire scores 0.576 against our 0.676.

**The number verified perfectly against its artifact. The selection behind it was undisclosed.** That is the lesson worth carrying: a value can be real, traceable and reproducible and still be a selective report. Read `caveats`, `note` and `_SUPERSEDED` fields before quoting a file.

Now stated in Methods and Results, and the Discussion's parsimony paragraph was rebuilt to rest on the leg that survives the problem: the paired within-cohort comparison in which adding the CSF panel changed AUC by −0.003. **The parsimony finding is weaker than originally drafted, and that is its correct state.**

### The registration contradiction

The manuscript called the baseline "pre-registered" in **six** places — Abstract, Results, Discussion twice, Methods, and the Figure 1 legend — while Limitations stated that no analysis here was pre-registered. The PPMI work ran 23–29 June 2026; the OSF registration is stamped 17 July 2026, so the specification was not pre-specified relative to these analyses, exactly as `DEVIATIONS_TABLE.md` already warned. All six now read "original ten-variable". A Methods **Registration** subsection carries the statement TRIPOD+AI requires, and doubles as the explanation for the naming.

Keeping the registration disclosure *helps*: TRIPOD+AI requires registration status to be stated even when a study is unregistered, and the OSF record is public, so silence would look worse than disclosure.

### Structural audit against IMRaD

Proportions are sound (Introduction 17%, Results 46%, Discussion 36%), the Introduction funnels properly, and every Discussion paragraph opens on its point. Two real faults were found and fixed:

- **Two Results subsections had no Methods at all** — the Hlavnička refit and the trial-enrichment simulation. Four Methods subsections were added: Sample size adequacy and stability, Subgroup and equity audit, Comparison with a questionnaire-only model, Prognostic enrichment. Methods now mirrors the Results order.
- **Seven values appeared for the first time in the Discussion**, which IMRaD forbids. Each was verified against its artifact and moved into Results: equity spreads 0.011 / 0.050 / 0.207 (plus APOE 0.023 and GBA 0.024, previously reported nowhere), n = 98, instability 0.163, Riley 1,254 and 524.

### Abstract and cover letter

The abstract was rebuilt to the Springer Nature shape and now ends on an implication rather than a verdict; numbers cut from 12 to 6. The cover letter was rewritten after the Nature Portfolio editorial identified three faults in the draft: it repeated the abstract, it compared the work to other Nature Portfolio papers by citing the journal's own systematic review against them, and it volunteered limitations. It now leads with the contribution and fits one page on the Gachon letterhead.

**One known imprecision in my own work:** the formula behind `instability_median` is not recoverable — the producing script is not in the repo and the artifact records only the value and a note. The Methods describe it at the level the artifact supports.

## 8. Open

- **LLM-disclosure statement** in Methods, which npj requires outright.

Two items previously listed here as blockers are **not** blockers, corrected 2026-08-12 after checking the guideline wording verbatim rather than a summary of it:

- **Suggested referees are optional at npj PD.** The page says *"It is also appropriate to include suggested or excluded referees in the cover letter."* The cover letter's five suggestions can stay or go; the `[insert]` email fields do not hold up submission. The mandatory-referee rule belongs to Scientific Reports, and assuming it carried across is how the error happened.
- **The Reporting Summary is due at revision.** *"You will be asked to complete and submit the Nature Portfolio Reporting Summary together with the revised version of your manuscript after peer-review."* Including it at submission is encouraged, not required, so `Reporting_Summary_answers_npjPD.docx` is useful preparation rather than a gate.
- **Shivani Devi's funder and grant number**, a marked field in Acknowledgements.
- **Repository DOI** on acceptance — archive the GitHub repo to get a persistent identifier.
- Confirm S.Y. and Y.S.'s contribution statements with them.
- At acceptance only: Nature style lists all authors unless more than five; the references currently use *et al.* after three.
- Unchanged: `data/` is gitignored in the working repo and the local tree still has everything after `b80a701` uncommitted. PRS control and B5 remain blocked on external access.
