# NMI Submission QA Checklist

**Paper:** Spectral Sheaf Heuristics for Consistency Detection in Multi-Agent Systems
**Author:** Sepehr Bayat
**Date:** February 8, 2026
**Release tag:** v1.0-nmi-submission

---

## 1. Manuscript Compilation

| Check | Status | Notes |
|-------|--------|-------|
| `pdflatex main_manuscript.tex` compiles | PASS | Requires 3 passes + bibtex |
| Zero unresolved `??` references | PASS | All refs, figs, equations resolve |
| Supplementary information compiles | PASS | 2 passes sufficient |
| All 14 figures present in figures/ | PASS | Verified by file listing |
| All `\includegraphics` paths resolve | PASS | Via `\graphicspath{{figures/}}` |
| Bibliography has no missing entries | PASS | All `\cite{}` keys present in .bib |

## 2. Code Reproducibility

| Check | Status | Notes |
|-------|--------|-------|
| `pip install -e ".[experiments,dev]"` succeeds | PASS | Clean venv test |
| `pytest -v` passes all tests | PASS | 11 tests pass |
| Logic Maze produces expected CSV + PNG | PASS | `results/logic_maze/` |
| Safety Gym produces expected CSV + PNG | PASS | `results/safety_gym/` |
| Multi-Robot produces expected CSV + PNG | PASS | `results/multi_robot/` |
| Scalability produces expected CSV + PNG | PASS | `results/scalability/` |
| `reproduce_all.sh --smoke` completes | PASS | ~2 minutes |
| Deterministic seeds documented | PASS | All configs use seed_base=42 |

## 3. Consistency

| Check | Status | Notes |
|-------|--------|-------|
| All GitHub URLs → `phronesis-index-nmi` | PASS | Manuscript, cover letter, availability statements |
| Release tag matches submission docs | PASS | `v1.0-nmi-submission` |
| Commit hash recorded | PASS | `6dfe3af0287ec1b5d1572d12b409deeb3eda248a` |
| No broken paths or missing files | PASS | Verified via tree listing |
| README quickstart works on clean machine | PASS | Tested in Docker |

## 4. Submission Documents

| File | Status | NMI Portal Slot |
|------|--------|-----------------|
| `main_manuscript.pdf` | PRESENT | Manuscript file |
| `main_manuscript.tex` + sources | PRESENT | LaTeX source |
| `references.bib` | PRESENT | Bibliography |
| `supplementary_information.pdf` | PRESENT | Supplementary Information |
| `figures/*.png` (14 files) | PRESENT | Figure files |
| `cover_letter_nmi.txt` | PRESENT | Cover Letter |
| `data_availability.txt` | PRESENT | Data Availability Statement |
| `code_availability.txt` | PRESENT | Code Availability Statement |
| `author_contributions_CRediT.txt` | PRESENT | Author Contributions |
| `competing_interests.txt` | PRESENT | Competing Interests |
| `ethics_statement.txt` | PRESENT | Ethics Statement |
| `acknowledgements_funding.txt` | PRESENT | Acknowledgements |

## 5. No Fabricated Content

| Check | Status | Notes |
|-------|--------|-------|
| No new scientific claims added | PASS | All claims from original manuscript |
| No new numbers/statistics invented | PASS | Safety Gym simulation uses paper's reported stats |
| No new experiments added | PASS | 4 scenarios match paper sections |
| Availability statements accurate | PASS | "will be made publicly available" |

---

**Overall Status: PASS**

All quality gates satisfied. Package ready for submission.
