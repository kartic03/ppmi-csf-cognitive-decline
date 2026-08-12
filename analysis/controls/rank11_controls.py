"""RANK 11 — NEGATIVE AND POSITIVE CONTROLS FOR THE CSF INCREMENT.

WHY THIS EXISTS. The existing noise floor (clinical + 5 random Gaussians,
dR2 = -0.0093) shows the harness does not manufacture signal from pure noise.
That is a weak test: random columns have no batch structure, no missingness
pattern, and no correlation with cohort composition. A REAL measured variable
that should nonetheless be null is a much harder test, because it carries all
the structure that could confound the CSF result.

AND -- the point that decides whether any of this means anything -- a negative
control is uninterpretable on its own. A pipeline that finds nothing anywhere
is not conservative, it is uninformative. So every negative arm below is paired
with a POSITIVE control: a variable with an established effect on cognitive
decline in PD, run through the identical harness. The negative arms only carry
weight if the positive arms fire.

ARMS
  anchor    CSF block                 must reproduce the committed +0.0581
  neg 1     SAA positivity            diagnostic marker, near-saturated within
                                      diagnosed PD, no established relation to
                                      decline RATE -> expect ~0
  neg 2     SAA assay version         PURELY TECHNICAL (which platform ran the
                                      sample). Expect ~0. If this is non-trivial
                                      the cohort has batch structure the CSF
                                      increment could be riding on -- this arm
                                      is a genuine threat test, not a formality.
  neg 3     5 random Gaussians        re-run of the known noise floor, as a
                                      harness sanity check
  pos 1     APOE e4                   established predictor of cognitive decline
                                      in PD -> expect > 0
  pos 2     GBA carrier status        established faster cognitive decline
                                      -> expect > 0

For the positive arms APOE_e4 and gba_status must be REMOVED from the base
block first -- both are already in CLINICAL_COLS, so leaving them in would test
adding a variable to itself and return ~0 for a trivial reason.

PRS IS NOT RUN AND CANNOT BE. See the note printed at the end.
"""
import os, sys, json, importlib.util
import numpy as np
import pandas as pd

ROOT = os.path.expanduser("~/pd_repro")
spec = importlib.util.spec_from_file_location(
    "inc", os.path.join(ROOT, "src/phase2/03_increment.py"))
inc = importlib.util.module_from_spec(spec)
sys.modules["inc"] = inc
spec.loader.exec_module(inc)

CV = {"n_outer": 5, "n_inner": 5, "seeds": list(range(10))}
GRID = {"alpha": inc.RIDGE_ALPHA_GRID}   # NEVER leave this None -- see the
                                         # param_grid=None incident; an untuned
                                         # run silently returns wrong numbers


def run(base_builder, aug, moca_long, ids, label):
    r = inc.paired_increment_cv(
        base_builder=base_builder, aug_df=aug, moca_long=moca_long,
        model_factory=inc._default_model, param_grid=GRID, cv_config=CV,
        analytic_ids=ids, outcome_col="eb_slope")
    print(f"  {label:<34s} n={r['n_subjects']:4d}  "
          f"R2 {r['r2_base']:.4f} -> {r['r2_aug']:.4f}   "
          f"dR2 {r['dr2']:+.4f}  [seed range {r['dr2_repeat_lo']:+.4f}, "
          f"{r['dr2_repeat_hi']:+.4f}]")
    return {"label": label, "n": int(r["n_subjects"]), "r2_base": r["r2_base"],
            "r2_aug": r["r2_aug"], "dr2": r["dr2"],
            "repeat_lo": r["dr2_repeat_lo"], "repeat_hi": r["dr2_repeat_hi"]}


def main():
    moca_long = inc.load_moca(inc._CURATED)
    slope_ids = set(pd.read_parquet(inc._OUTCOME)["subject_id"].astype(int))
    clin = inc._make_clinical_builder(inc._CURATED, slope_ids)

    csf = inc._load_csf_aug_df(inc._CURATED, inc._NULISA, slope_ids)
    csf.index = csf.index.astype(int)
    csf_ids = clin(list(csf.index)).index.intersection(csf.index).tolist()

    out = {}

    # ---- ANCHOR ---------------------------------------------------------
    # Reproduce a known committed number BEFORE trusting anything else this
    # harness produces. If this does not land on +0.0581, everything below is
    # suspect and the run stops.
    print("=" * 92)
    print("ANCHOR — must reproduce the committed CSF increment (+0.0581)")
    print("=" * 92)
    out["anchor_csf"] = run(clin, csf, moca_long, csf_ids, "CSF block (committed)")
    if abs(out["anchor_csf"]["dr2"] - 0.0581) > 0.002:
        print(f"\n!! ANCHOR FAILED: got {out['anchor_csf']['dr2']:+.4f}, "
              f"expected +0.0581. Harness is not trustworthy. STOPPING.")
        sys.exit(1)
    print("  -> anchor OK\n")

    # ---- build control blocks ------------------------------------------
    cur = pd.read_parquet(inc._CURATED)
    bl = cur[(cur["COHORT"] == 1) & (cur["EVENT_ID"] == "BL")].copy()
    bl["gba_status"] = bl["subgroup"].str.contains("GBA", na=False).astype(float)
    bl = bl.set_index("PATNO")
    bl.index = bl.index.astype(int)

    # SAA: 1=positive, 0=negative in this cut. Codes 2 and 3 (10 and 18
    # subjects) are NOT documented in the dictionary available here, so they
    # are dropped rather than guessed at.
    saa_raw = bl["CSFSAA"]
    saa = saa_raw[saa_raw.isin([0.0, 1.0])].rename("saa_positive").to_frame()
    assay = bl["CSFSAA_assay"].dropna().rename("saa_assay_version").to_frame()

    rng = np.random.default_rng(0)
    noise = pd.DataFrame(
        rng.standard_normal((len(bl), 5)),
        index=bl.index, columns=[f"noise{i}" for i in range(5)])

    print("=" * 92)
    print(f"NEGATIVE CONTROLS — on the CSF-complete index (n={len(csf_ids)}), "
          "so dR2 is directly comparable to the anchor")
    print("=" * 92)
    out["neg_saa_csfset"] = run(clin, saa, moca_long,
                                [i for i in csf_ids if i in saa.index],
                                "SAA positivity")
    out["neg_assay_csfset"] = run(clin, assay, moca_long,
                                  [i for i in csf_ids if i in assay.index],
                                  "SAA assay version (technical)")
    out["neg_noise_csfset"] = run(clin, noise, moca_long, csf_ids,
                                  "5 random Gaussians")

    print("\n" + "=" * 92)
    print("NEGATIVE CONTROLS — each on its own maximal index (more power, "
          "not comparable to the anchor)")
    print("=" * 92)
    saa_ids = clin(list(saa.index)).index.intersection(saa.index).tolist()
    assay_ids = clin(list(assay.index)).index.intersection(assay.index).tolist()
    out["neg_saa_max"] = run(clin, saa, moca_long, saa_ids, "SAA positivity")
    out["neg_assay_max"] = run(clin, assay, moca_long, assay_ids,
                               "SAA assay version (technical)")

    # ---- POSITIVE CONTROLS ---------------------------------------------
    # APOE_e4 and gba_status live in CLINICAL_COLS, so they must come OUT of
    # the base before being tested as an augmentation.
    print("\n" + "=" * 92)
    print("POSITIVE CONTROLS — base block with the variable REMOVED, then "
          "added back as the augmentation")
    print("=" * 92)

    def builder_without(drop):
        cols = [c for c in inc.CLINICAL_COLS if c not in drop]
        sub = bl[bl.index.isin(slope_ids)]

        def b(subject_ids):
            ids = set(int(i) for i in np.asarray(list(subject_ids)))
            valid = ids & set(sub.index)
            return sub.loc[sorted(valid), cols].copy()
        return b

    for var, lbl in [("APOE_e4", "APOE e4 (established)"),
                     ("gba_status", "GBA carrier (established)")]:
        base = builder_without({var})
        aug = bl[[var]].dropna()
        ids = base(list(aug.index)).index.intersection(aug.index).tolist()
        out[f"pos_{var}"] = run(base, aug, moca_long, ids, lbl)

    # ---- verdict --------------------------------------------------------
    print("\n" + "=" * 92)
    print("READING")
    print("=" * 92)
    pos_fires = [out[k]["dr2"] for k in out if k.startswith("pos_")]
    neg_flat = [out[k]["dr2"] for k in out if k.startswith("neg_")]
    print(f"  positive-control dR2 range : {min(pos_fires):+.4f} .. {max(pos_fires):+.4f}")
    print(f"  negative-control dR2 range : {min(neg_flat):+.4f} .. {max(neg_flat):+.4f}")
    print(f"  CSF anchor                 : {out['anchor_csf']['dr2']:+.4f}")
    if max(pos_fires) < 0.005:
        print("\n  !! POSITIVE CONTROLS DID NOT FIRE. The negative controls below"
              "\n     are then UNINTERPRETABLE -- an outcome this harness cannot"
              "\n     detect a known effect on cannot be used to argue that a null"
              "\n     is meaningful. Report this, do not report the nulls alone.")
    batch = max(abs(out["neg_assay_csfset"]["dr2"]), abs(out["neg_assay_max"]["dr2"]))
    if batch > 0.02:
        print(f"\n  !! ASSAY VERSION CARRIES dR2 {batch:+.4f} -- the cohort has "
              "batch structure\n     the CSF increment may be riding on. This must be "
              "investigated, not\n     reported as a passed control.")

    print("\n" + "=" * 92)
    print("PRS ARM: NOT RUN — BLOCKED, NOT NULL")
    print("=" * 92)
    print("""  A PD polygenic risk score would be the strongest negative control
  available: PD SUSCEPTIBILITY risk is well documented not to predict
  PROGRESSION, so a non-trivial PRS increment would indicate the harness is
  picking up cohort structure rather than biology.

  It cannot be computed here. The project holds no PPMI genotypes -- the only
  genetic files on disk are 1000G LD reference panels (EUR/AFR/EAS/SAS/AMR)
  downloaded for the MR, which are a reference panel, not participant data.
  Building the score needs the PPMI imputed-genotype release from LONI, plus
  Nalls 2019 (PMID 31701892) 90-variant weights.

  APOE e4 and GBA carrier status are NOT substitutes: both have established
  effects on cognitive decline, which makes them positive controls (as used
  above), not negative ones.

  Do not describe the PRS control as null. It is unrun.""")

    p = os.path.join(ROOT, "data/processed/phase2/rank11_controls.json")
    json.dump({"cv_config": {**CV, "seeds": list(CV["seeds"])},
               "param_grid": GRID, "outcome_col": "eb_slope",
               "arms": out,
               "prs_status": "NOT RUN — no PPMI genotypes in project; needs "
                             "LONI imputed-genotype release + Nalls 2019 weights",
               "saa_coding": "1=positive, 0=negative; codes 2 (n=10) and 3 "
                             "(n=18) undocumented in the available dictionary "
                             "and therefore dropped, not guessed",
               "caveat": "brackets are cross-seed repeat ranges, NOT confidence "
                         "intervals"},
              open(p, "w"), indent=2)
    print(f"\nWrote {p}")


if __name__ == "__main__":
    main()
