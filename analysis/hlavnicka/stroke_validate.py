"""Validate the free-text stroke coding against the RBDSQ's STRUCTURED field.

The RBDSQ comorbidity block carries a coded STROKE 0/1. That is almost certainly
what Hlavnicka used ("standard clinical questionnaires"), and it beats parsing
investigator free text. Comparing the two tells us:
  (a) whether to use the structured field for stroke  -> yes, if it is well populated
  (b) whether the free-text method is trustworthy at all -> which matters because
      FAINTING has no structured field here and would have to come from MHTERM.
"""
import io, re, zipfile
import pandas as pd

RBDSQ = (r"C:\Users\Kartic Mishra\Downloads"
         r"\REM_Sleep_Behavior_Disorder_Screening_Questionnaire_04Aug2026.csv")
MHZIP = r"C:\Users\Kartic Mishra\Downloads\Medical_History.zip"

STROKE_RX = re.compile(
    r"\bstroke\b|\bcerebrovascular\b|\bCVA\b|\bTIA\b|"
    r"transient isch[a]?emic attack|cerebral (?:infarct|h(?:a)?emorrhage)", re.I)
CARDIAC_RX = re.compile(r"myocard|cardiac infarct|cardic infarct|coronary", re.I)

rb = pd.read_csv(RBDSQ, low_memory=False)
# ever-positive per subject across visits (history question -> ever = 1)
struct = rb.groupby("PATNO")["STROKE"].max()
struct = struct.dropna()
print(f"RBDSQ structured STROKE: {len(struct):,} subjects, "
      f"{int((struct == 1).sum()):,} positive ({(struct == 1).mean():.2%})")

z = zipfile.ZipFile(MHZIP)
mh = pd.read_csv(z.open("Medical_Conditions_Log_04Aug2026.csv"), low_memory=False)
s = mh["MHTERM"].astype(str)
hit = s.str.contains(STROKE_RX, na=False) & ~s.str.contains(CARDIAC_RX, na=False)
ft_pos = set(mh.loc[hit, "PATNO"].unique())
print(f"free-text MHTERM stroke : {len(ft_pos):,} subjects flagged")

common = struct.index.intersection(pd.Index(sorted(set(mh['PATNO'].unique()))))
print(f"\nsubjects present in BOTH sources: {len(common):,}")

sp = set(struct[struct == 1].index) & set(common)
fp = ft_pos & set(common)
both = sp & fp
only_struct = sp - fp
only_ft = fp - sp
print(f"  structured positive      : {len(sp)}")
print(f"  free-text positive       : {len(fp)}")
print(f"  BOTH agree positive      : {len(both)}")
print(f"  structured only          : {len(only_struct)}")
print(f"  free-text only           : {len(only_ft)}")
if sp:
    print(f"\n  sensitivity of free text vs structured = {len(both)/len(sp):.1%}")
if fp:
    print(f"  PPV of free text vs structured         = {len(both)/len(fp):.1%}")

print("\nREADING:")
print("  Poor agreement does NOT necessarily mean the free text is wrong -- the")
print("  two instruments ask different questions (RBDSQ asks the participant a")
print("  screening question; the conditions log records investigator-entered")
print("  diagnoses). But it does mean the two are NOT interchangeable, and the")
print("  choice must be stated rather than assumed.")
