# src/mr/tests/test_ukbppp_instruments.py
import pandas as pd, subprocess
def test_ukbppp_instruments():
    subprocess.run(["python","src/mr/12_ukbppp_instruments.py"], check=True)
    df = pd.read_parquet("data/processed/mr/instruments_ukbppp.parquet")
    assert set(df["protein"].unique()) <= {"PDGFRB","ICAM1","VCAM1","MMP9","TIMP1"}  # MMP2 absent
    assert (df["pval"] < 5e-8).all()
    assert (df["F"] > 10).all()
    assert df["pos"].notna().all()   # POS38 mapped
    # Fix 2: guard against column rename/drop/reorder breaking the Task A5 union with deCODE
    assert list(df.columns) == ["protein","Name","rsid","chrom","pos","effectAllele","otherAllele","eaf","beta","se","pval","F","palindromic"]
