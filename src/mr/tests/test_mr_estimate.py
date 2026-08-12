import json, subprocess
def test_null_gate_and_pdgfrb_recovers_prior():
    subprocess.run(["python","src/mr/14_mr_estimate.py"], check=True, timeout=120)
    res = {(r["protein"], r["platform"]): r for r in json.load(open("data/processed/mr/mr_results.json"))}
    pdg = res[("PDGFRB","decode")]
    # prior single-platform result was OR ~1.00; expect a tight null on the better-powered file
    assert abs(pdg["or"] - 1.00) < 0.10
    # verdict rule: null requires CI to exclude OR 1.10 AND be above the 0.91 floor
    for r in res.values():
        if r["verdict"] == "null":
            assert r["ci_low"] <= 1.0 and r["ci_high"] < 1.10
            assert r["ci_low"] > 0.91
        if r["ci_high"] >= 1.10 and r["ci_low"] <= 1.0:
            assert r["verdict"] == "inconclusive"
