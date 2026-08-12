import json
import subprocess


def test_positive_control_recovers_signal():
    subprocess.run(["python", "src/mr/15_poscontrol.py"], check=True)
    with open("data/processed/mr/poscontrol_result.json") as f:
        r = json.load(f)
    # the chosen control has a known non-null causal direction on PD
    assert r["recovers_expected_direction"] is True
    assert not (r["ci_low"] <= 1.0 <= r["ci_high"])   # CI excludes the null
