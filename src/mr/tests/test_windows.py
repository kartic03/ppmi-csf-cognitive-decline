import json, subprocess, pathlib
def test_windows_grch38_and_chromosomes():
    subprocess.run(["python","src/mr/10_windows.py"], check=True)
    with open("data/processed/mr/windows.json") as f:
        w = json.load(f)
    # Known GRCh38 chromosomes for the six proteins
    assert w["PDGFRB"]["chrom"] in ("5","chr5")
    assert w["ICAM1"]["chrom"] in ("19","chr19")
    assert w["VCAM1"]["chrom"] in ("1","chr1")
    assert w["MMP9"]["chrom"] in ("20","chr20")
    assert w["MMP2"]["chrom"] in ("16","chr16")
    assert w["TIMP1"]["chrom"] in ("X","chrX","23")
    for p in w.values():
        assert p["cis_start"] == max(0, p["gene_start"] - 500_000)
        assert p["cis_end"] == p["gene_end"] + 500_000
