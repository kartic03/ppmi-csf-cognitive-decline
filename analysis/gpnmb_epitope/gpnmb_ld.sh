#!/usr/bin/env bash
# Are the deCODE and UKB-PPP GPNMB instrument sets tagging the SAME signal?
cd "$HOME/pd_repro" || exit 1
export PATH="$HOME/.pixi/bin:$PATH"
T=/tmp/gpnmb_ld; rm -rf $T; mkdir -p $T

pixi run python -c "
import pandas as pd
d=pd.read_csv('data/processed/mr/poscontrol_instruments.csv')
d=d[d.protein=='GPNMB']
open('$T/snps.txt','w').write('\n'.join(d.rsid.dropna().unique())+'\n')
print('submitted:', list(d.rsid.dropna().unique()))
" 2>&1 | grep -v "WARN the lock"

pixi run plink --bfile data/raw/mr/ld_1000g_eur/EUR --extract $T/snps.txt \
  --r2 --ld-window 99999 --ld-window-kb 10000 --ld-window-r2 0 --out $T/ld \
  > $T/plink.log 2>&1
grep -i "variants remaining\|--extract" $T/plink.log | head -3

echo
echo "===== pairwise r2 ====="
pixi run python -c "
import pandas as pd, os
f='$T/ld.ld'
if not os.path.exists(f):
    print('no .ld file — see log'); raise SystemExit
ld=pd.read_csv(f, sep=r'\s+')
inst=pd.read_csv('data/processed/mr/poscontrol_instruments.csv')
plat=dict(zip(inst.rsid,inst.platform))
ld['p_a']=ld.SNP_A.map(plat); ld['p_b']=ld.SNP_B.map(plat)
cross=ld[ld.p_a!=ld.p_b].sort_values('R2',ascending=False)
same=ld[ld.p_a==ld.p_b].sort_values('R2',ascending=False)
print('panel-present instruments:', sorted(set(ld.SNP_A)|set(ld.SNP_B)))
print()
print('CROSS-PLATFORM pairs:', len(cross))
if len(cross):
    print(cross[['SNP_A','p_a','SNP_B','p_b','R2']].to_string(index=False))
    print()
    print('  MAX cross-platform r2 = %.4f' % cross.R2.max())
print()
print('WITHIN-PLATFORM pairs:', len(same))
if len(same):
    print(same[['SNP_A','p_a','SNP_B','p_b','R2']].to_string(index=False))
" 2>&1 | grep -v "WARN the lock"
