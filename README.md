# Deep Learning Projects

A collection of end-to-end deep learning projects built with **Python, TensorFlow and Keras**.
Each project is self-contained: it has its own dataset, code, tests, notebook and README, and can
be set up and run on its own.

The projects follow the same structure. Reusable logic lives in a tested Python package, a
Jupyter notebook walks through the analysis step by step, and a command-line interface runs the
complete pipeline.

## Projects

| # | Project | Domain | Problem type | Techniques | Key result |
|---|---------|--------|--------------|------------|------------|
| 1 | [Lending Club Loan Default Prediction](LendingClubLoanDefault/) | Finance | Binary classification (imbalanced) | Keras MLP, class weights, BatchNorm/Dropout, threshold tuning | Test ROC-AUC 0.645, 57% of defaults detected |

<!-- Add new projects as rows above, linking to the project folder. -->

## Repository layout

```
DeepLearningProjects/
├── README.md                    <- this file: overview and index of all projects
├── .gitignore                   <- shared ignore rules for every project
└── <ProjectName>/               <- one folder per project, self-contained
    ├── README.md                <- problem, approach, results, how to run
    ├── DataSet/                 <- input data (or instructions to download it)
    ├── <Project>.ipynb          <- guided walkthrough of the analysis
    ├── src/<package>/           <- reusable, tested project code
    ├── tests/                   <- pytest unit tests
    ├── pyproject.toml           <- package metadata and dependencies
    └── requirements.txt         <- dependency list for a quick `pip install -r`
```

Files generated when a project runs are git-ignored and recreated on every run: `models/` holds
trained models, `reports/` holds metrics and figures, and `logs/` holds run logs.

## Getting started

**Prerequisites:** Python **3.10–3.13** (TensorFlow does not yet support Python 3.14) and Git.

```bash
git clone <repository-url>
cd DeepLearningProjects/<ProjectName>

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -e ".[dev]"
```

Then follow the project's own README. It covers running the notebook, the CLI and the tests.

> Use a separate virtual environment per project. Projects may need different library versions.

## Adding a new project

1. Create a folder named after the project in PascalCase, e.g. `CustomerChurnPrediction/`.
2. Use the structure shown above. Put the logic in `src/<package_name>/`, add tests in `tests/`,
   and make the notebook import from the package rather than duplicating code.
3. Write a project `README.md` covering: objective, dataset, approach, results, project structure,
   setup and usage. [LendingClubLoanDefault/README.md](LendingClubLoanDefault/README.md) can serve
   as a template.
4. Add a row for the project to the **Projects** table above.
5. Large datasets (over ~50 MB) should not be committed. Document where to download them instead.

## Tech stack

Python · TensorFlow / Keras · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · pytest · Jupyter
