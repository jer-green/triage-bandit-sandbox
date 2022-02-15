# Example pure python pipeline chaining together the steps manually
set -e # Ensure we exit if one of the steps fail
# Replace by any pipeline tooling of your choice
echo "Creating Data..."
python -m triage_bandit_sandbox.data.download_data

echo "Splitting test/train set..."
python -m triage_bandit_sandbox.features.create_train_test_split

echo "Preparing preprocessor..."
python -m triage_bandit_sandbox.features.preprocess

echo "Training model..."
python -m triage_bandit_sandbox.train

echo "Evaluating model..."
python -m triage_bandit_sandbox.evaluate
