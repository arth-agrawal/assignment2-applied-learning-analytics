# Applied Learning Analytics: OULAD and Moodle

This repository contains the analysis scripts and aggregate results that accompany the Assignment 2 report.

## Contents

`analysis_oulad.py` prepares the Open University Learning Analytics Dataset (OULAD), evaluates completion and assessment submission models, and recreates the related figures.

`analysis_moodle.py` prepares the supplied Moodle course interaction export, analyzes weekly behaviour, evaluates next week return prediction, and creates learner activity profiles.

`oulad_results.json` and `moodle_results.json` contain aggregate model results. Raw event records and learner identifiers are not included.

## Data sources

OULAD original release: [Figshare dataset](https://doi.org/10.6084/m9.figshare.5081998.v1).

Moodle interaction data: [course supplied Google Drive file](https://drive.google.com/file/d/1A82KQ0eZURKcxucUX00agbdz1qatwOjc/view?usp=sharing).

The raw datasets are not included. The Moodle export has no final course grades or complete enrolment list, so its prediction target is recorded return to the platform during the following week.

## Reproduce the analysis

For OULAD, download and extract the dataset CSV files, then run:

```text
python analysis_oulad.py --data PATH_TO_OULAD_CSV_FOLDER --out results
```

For Moodle, place the course supplied `MOODLE Data.csv` beside the script, then run:

```text
python analysis_moodle.py --data "MOODLE Data.csv" --out results_moodle
```

The scripts require Python with pandas, NumPy, SciPy and Matplotlib. The OULAD script also uses scikit learn.
