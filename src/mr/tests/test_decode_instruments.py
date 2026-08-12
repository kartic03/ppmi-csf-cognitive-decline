import pandas as pd, subprocess
def test_decode_instruments_hygiene():
    subprocess.run(["python","src/mr/11_decode_instruments.py"], check=True)
    df = pd.read_parquet("data/processed/mr/instruments_decode.parquet")
    assert (df["pval"] < 5e-8).all()                 # genome-wide significant
    assert (df["F"] > 10).all()                       # strong instruments only
    assert df["eaf"].notna().all()                    # EAF from annotation, never missing
    assert (df["otherAllele"] != "!").all()           # multiallelic rows repaired/dropped
    # PDGFRB lead is the known palindrome rs2304058 flagged palindromic
    pdg = df[df["protein"] == "PDGFRB"]
    assert pdg["palindromic"].any()
    assert (df[df["protein"] == "PDGFRB"]["rsid"] == "rs2304058").any()
