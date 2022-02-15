# triage-bandit-sandbox

ML Project on Babylon AI Platform: triage-bandit-sandbox

## Project Organization

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── features       <- The training and testing data sets.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── pyproject.toml     <- The requirements file for reproducing the analysis environment.
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    │
    └── triage_bandit_sandbox
        │   Source code for use in this project.
        │
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data
        │   └── download_data.py
        │
        ├── features       <- Scripts to turn raw data into features for modeling
        │   ├── create_train_test_split.py
        │   └── preprocess.py
        │
        ├── evaluate.py    <- Script to evaluate model performance from predict.py
        │
        ├── predict.py     <- Script to predict from samples, used as entrypoint when serving with pottery
        │
        └── train.py       <- Script to train
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
            └── visualize.py

## Installation

Clone and install the repository locally:

```
git clone 
cd  # TODO strip name only?
python -m venv .venv
source .venv/bin/activate
pottery install # TODO: This won't work!
pottery train # TODO: This won't work!
```

## Required access

Access to sso `sandbox` account is required.
Ask IT to add you to the sso account: `selfcare_sandbox` with the role `sso-admin`.

Then add

```
[profile sandbox]
sso_start_url = https://babylonhealth.awsapps.com/start#/
sso_region = eu-west-2
sso_account_id = 102743306802
sso_role_name = sso-admin
sse = AES256
output = json
```

to your ~/.aws/config file

## How to collaborate:

Create a new branch replacing `<experiment name>` with a name describing your change.

```
git checkout -b <experiment name>
```

Then make your changes to the codebase as required

To reproduce the pipeline locally, run

```
pottery train
```

Once you're happy with your changes locally, run a final

```
pottery train
dvc push

git add -u
git commmit -m <description of change>
git push origin <experiment name>
```
