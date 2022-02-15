rm -f report.md
echo "# triage-bandit-sandbox" >> report.md

echo "> **NOTE:**" >> report.md
echo "> Example Report training a simple MLP classifier." >> report.md
echo "> Replace code as needed for your project." >> report.md

echo "## Metrics: " >> report.md
python -c "import pandas as pd; print(pd.read_json('test.json').to_markdown())" >> report.md


echo "## Training Metrics: " >> report.md
python -c "import pandas as pd; print(pd.read_json('train.json').to_markdown())" >> report.md
