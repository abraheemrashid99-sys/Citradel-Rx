# Reference verification (revised manuscript)

Every entry of the bibliography of `paper/main.tex` was checked against its record at an
authoritative source: the publisher or proceedings page (IEEE Xplore, ACM DL, PMLR, CVF Open
Access, NeurIPS proceedings, OpenReview, JMLR, PNAS/PubMed, Project Euclid, JSTOR, Taylor &
Francis, Cell), the arXiv abstract page, the 3GPP/ETSI specification record, with DBLP as a
cross-check of authors, venue and year.

Method and its limits. Direct page fetches from the session were refused by its network policy
for dblp.org, api.crossref.org, proceedings.mlr.press and export.arxiv.org, so the fields were
confirmed from search results restricted to the domains above. Fields that rest only on
secondary records are marked below; nothing was filled from memory.

## Corrections made against the submitted bibliography

| Entry | Correction |
|---|---|
| CRC-8 polynomial | The polynomial used, D^8+D^7+D^4+D^3+D+1 (`citadel_rx/coding/crc.py`, 0x9B), is gCRC8 of 3GPP TS 36.212 (E-UTRA) Sec. 5.1.1; TS 38.212 (NR) defines CRC-24A/B/C, CRC-16, CRC-11 and CRC-6 only. The citation now points to TS 36.212 (seen as ETSI TS 136 212 V19.1.0, 2025-10). |
| Titans | Now published: NeurIPS 2025 (vol. 38); arXiv:2501.00663 kept. |
| Nested learning | Confirmed NeurIPS 2025 (vol. 38), arXiv:2512.24695. |
| Wu et al. 2024 | arXiv only (2409.02430); no journal or conference version found. The arXiv title spells "Receviers"; it is reproduced as in the source. |
| Wu et al. 2023 (TTA risks) | PMLR vol. 202, pp. 37456-37495 added. |
| MedBN | CVPR 2024, pp. 5997-6007 added. |
| Blanchard et al. 2017 | pp. 119-129 (NeurIPS 30 as indexed by DBLP; the ACM edition prints 118-128). |
| Steinhardt et al. 2017 | pp. 3517-3529 (DBLP). |
| Code-verified training | ViterbiNet (IEEE TWC 2020) re-trains on decoded and re-encoded codewords when decoding succeeds; Schibisch et al. (ISWCS 2018), whose training set "contains only the correctly received symbols ... (e.g., due to a cyclic redundancy check (CRC))", is cited with it. |
| McClelland et al. 1995 | Full title with its subtitle. |

## New entries (all verified)

Zhou, Maham, Hjorungnes, IEEE TWC 11(3):903-907, 2012; Kapetanovic, Zheng, Rusek, IEEE Commun.
Mag. 53(6):21-27, 2015; Kapetanovic, Zheng, Wong, Ottersten, PIMRC 2013, pp. 13-18; Tugnait,
IEEE WCL 4(5):525-528, 2015 (DOI from citing records); Shlezinger, Farsad, Eldar, Goldsmith, IEEE
TWC 19(5):3319-3331, 2020; Schibisch et al., ISWCS 2018, pp. 1-5; Niu et al. (EATA), ICML 2022,
PMLR 162:16888-16905; Wang, Fink, Van Gool, Dai (CoTTA), CVPR 2022, pp. 7201-7211; Niu et al.
(SAR), ICLR 2023; Yuan, Xie, Li (RoTTA), CVPR 2023, pp. 15922-15932; Hoang, Vo, Do, NeurIPS 2024
(vol. 37; pages not confirmed); Li, Grandvalet, Davoine (L2-SP), ICML 2018, PMLR 80:2825-2834
(PMLR pagination); Kirkpatrick et al. (EWC), PNAS 114(13):3521-3526, 2017 (all 14 authors);
Tarvainen, Valpola, NeurIPS 30, pp. 1195-1204; Athalye, Carlini, Wagner (BPDA, Sec. 4.1), ICML
2018, PMLR 80:274-283; Athalye, Engstrom, Ilyas, Kwok (EOT), ICML 2018, PMLR 80:284-293; Tramer,
Carlini, Brendel, Madry, NeurIPS 33, pp. 1633-1645; Carlini et al., arXiv:1902.06705 (9 authors in
v2); Ilyas, Engstrom, Athalye, Lin, ICML 2018, PMLR 80:2137-2146; Wierstra et al., JMLR 15(27):
949-980, 2014; Sadeghi, Larsson, IEEE WCL 8(1):213-216, 2019; McClelland, McNaughton, O'Reilly,
Psychol. Rev. 102(3):419-457, 1995; Kumaran, Hassabis, McClelland, Trends Cogn. Sci. 20(7):
512-534, 2016; Kunsch, Ann. Statist. 17(3):1217-1241, 1989; Wilson, JASA 22(158):209-212, 1927;
Huber, Ann. Math. Statist. 35(1):73-101, 1964; 3GPP TR 38.901 V19.5.0 Table 7.4.1-1
(UMi-Street Canyon LOS: PL1 = 32.4 + 21 log10(d3D) + 20 log10(fc) below the breakpoint).

## Added in the second revision campaign

3GPP TS 38.212 "NR; Multiplexing and channel coding", V19.2.0 (2025-12): Sec. 5.1 CRC calculation,
Sec. 5.3.2 low density parity check coding (table of contents of the 3GPP draft record). The code
blocks of the coded scene are built by Sionna PHY 2.2 (`LDPC5GEncoder`, `CRCEncoder("CRC24A")`).

Sionna: cited as the software record its authors ask for, copied from the citation block of the
installed package's metadata (`sionna-no-rt` 2.2.0, `importlib.metadata ... METADATA`): `@software{sionna,
title = {Sionna}, author = {Hoydis, Jakob and Cammerer, Sebastian and {Ait Aoudia}, Fayçal and
Nimier-David, Merlin and Maggi, Lorenzo and Marcus, Guillermo and Vem, Avinash and Keller, Alexander},
note = {https://nvlabs.github.io/sionna/}, year = {2022}, version = {2.2.0}}`. The UMi drops use
`sionna.phy.channel.tr38901.UMi` of the same package.

## Items to check once by eye before submission

* TR 38.901 V19.5.0, Table 7.4.1-1: the extracted PDF text was partly garbled; the coefficients
  (21 log10 d below the breakpoint) match every earlier version seen (V14-V17).
* Mokhtari et al. (CDC 2016, pp. 7195-7201) and Samuel et al. (DOI) rest on secondary records;
  Mokhtari et al. is no longer cited in the revision.
