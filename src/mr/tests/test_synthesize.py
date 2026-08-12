"""
test_synthesize.py - Contract tests for the cross-platform synthesis (Task A9).

Verbatim contract from task-A9-brief.md, strengthened with per-protein assertions.
"""
import pandas as pd, subprocess, json


def test_synthesis_requires_poscontrol_and_concordance():
    subprocess.run(["python", "src/mr/17_synthesize.py"], check=True)
    t = pd.read_csv("data/processed/mr/mr_final_table.csv", keep_default_na=False)
    # no protein gets a "causal null" call unless the positive control passed
    pc = json.load(open("data/processed/mr/poscontrol_result.json"))
    if not pc["recovers_expected_direction"]:
        assert (t["causal_call"] != "null").all()
    else:
        # The branch above is DEAD on the committed data (the control passes),
        # so it has never actually executed and the gate it claims to enforce
        # was untested. Exercise it directly on a forced-failure copy so the
        # "no null without a passing positive control" rule is really checked.
        # See test_poscontrol_gate_blocks_null_calls below.
        assert pc["recovers_expected_direction"] is True
    # a "null" call requires concordant null on both platforms (where both exist)
    for _, r in t.iterrows():
        if r["causal_call"] == "null" and int(r["platforms_tested"]) == 2:
            assert r["decode_verdict"] == "null" and r["ukbppp_verdict"] == "null"
    # PDGFRB is labeled the primary BBB-integrity instrument
    assert (t.loc[t["protein"] == "PDGFRB", "role"] == "primary").all()

    # Per-protein assertions (keep_default_na=False ensures "null" stays a string)
    pdgfrb = t.loc[t["protein"] == "PDGFRB"].iloc[0]
    assert pdgfrb["role"] == "primary"
    assert pdgfrb["causal_call"] == "null", (
        f"PDGFRB causal_call expected 'null' (string), got {pdgfrb['causal_call']!r}"
    )
    assert pdgfrb["decode_verdict"] == "null", (
        f"PDGFRB decode_verdict expected 'null' (string), got {pdgfrb['decode_verdict']!r}"
    )
    assert pdgfrb["ukbppp_verdict"] == "null", (
        f"PDGFRB ukbppp_verdict expected 'null' (string), got {pdgfrb['ukbppp_verdict']!r}"
    )

    icam1 = t.loc[t["protein"] == "ICAM1"].iloc[0]
    assert icam1["causal_call"] == "discordant", (
        f"ICAM1 causal_call expected 'discordant', got {icam1['causal_call']!r}"
    )

    vcam1 = t.loc[t["protein"] == "VCAM1"].iloc[0]
    assert vcam1["causal_call"] == "inconclusive", (
        f"VCAM1 causal_call expected 'inconclusive' (not 'positive' despite wide-CI UKB OR), "
        f"got {vcam1['causal_call']!r}"
    )

    timp1 = t.loc[t["protein"] == "TIMP1"].iloc[0]
    assert timp1["causal_call"] == "not_testable", (
        f"TIMP1 causal_call expected 'not_testable', got {timp1['causal_call']!r}"
    )


def test_poscontrol_gate_blocks_null_calls(tmp_path, monkeypatch):
    """The positive-control gate must actually block 'null' calls when it fails.

    Added 2026-08-04. The gate assertion in the test above sits behind
    `if not pc["recovers_expected_direction"]`, which is False on the committed
    data -- so that branch had never executed and the single rule protecting
    every causal-null claim in this paper was, in practice, untested.

    Here we run 17_synthesize.py against a copy of the real inputs in which the
    positive control is forced to FAIL, and assert that no protein survives with
    a 'null' call. If someone weakens the gate, this test fails loudly.
    """
    import json
    import shutil
    import subprocess
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    src_dir = repo / "data" / "processed" / "mr"

    work = tmp_path / "repo"
    (work / "data" / "processed" / "mr").mkdir(parents=True)
    (work / "data" / "raw" / "mr").mkdir(parents=True)
    (work / "src" / "mr").mkdir(parents=True)

    for f in src_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, work / "data" / "processed" / "mr" / f.name)
    for f in (repo / "src" / "mr").glob("*.py"):
        shutil.copy2(f, work / "src" / "mr" / f.name)
    # 17_synthesize also reads the phewas pleiotropy flags from data/raw.
    phewas = repo / "data" / "raw" / "mr" / "mr_phewas.json"
    if phewas.exists():
        shutil.copy2(phewas, work / "data" / "raw" / "mr" / phewas.name)

    pc_path = work / "data" / "processed" / "mr" / "poscontrol_result.json"
    pc = json.loads(pc_path.read_text())
    assert pc["recovers_expected_direction"] is True, "fixture assumes a passing control"
    pc["recovers_expected_direction"] = False          # force the failure
    pc_path.write_text(json.dumps(pc, indent=2))

    subprocess.run(["python", "src/mr/17_synthesize.py"], check=True, cwd=work)

    import pandas as pd
    t = pd.read_csv(
        work / "data" / "processed" / "mr" / "mr_final_table.csv",
        keep_default_na=False,
    )
    assert (t["causal_call"] != "null").all(), (
        "GATE BROKEN: a 'null' causal_call survived even though the positive "
        "control failed. No causal-null claim is licensed without a validated "
        "pipeline.\n" + t[["protein", "causal_call"]].to_string(index=False)
    )
