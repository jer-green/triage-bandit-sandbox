CREATE_VENV?="TRUE"

.venv:
	@ if [ $(CREATE_VENV) == "TRUE" ]; then python3 -m venv .venv; fi;

.requirements_installed: .venv
	@ if [ $(CREATE_VENV) == "TRUE" ]; then . .venv/bin/activate; fi; \
	pip install --upgrade pip; \
	pip install bh-pottery==0.0.19.dev0; \
	poetry install; \
	touch .requirements_installed

patch:
	./scripts/bump_model_version.sh patch

minor:
	./scripts/bump_model_version.sh minor

major:
	./scripts/bump_model_version.sh major

install: .requirements_installed
	@ echo "Installation Complete"

clean:
	@ rm -rf .requirements_installed

lint: .venv
	@ if [ $(CREATE_VENV) == "TRUE" ]; then . .venv/bin/activate; fi; \
	git diff --quiet || echo 'You have uncommitted changes, commit your changes first and rerun this command' && exit ;\
	isort triage_bandit_sandbox; \
	black triage_bandit_sandbox --exclude .venv; \
	git add -u; \

lint-check: .venv
	@ if [ $(CREATE_VENV) == "TRUE" ]; then . .venv/bin/activate; fi; \
	isort triage_bandit_sandbox --check --diff; \
	black triage_bandit_sandbox --check --exclude .venv; \
	flake8 triage_bandit_sandbox

pull: .venv
	@ if [ $(CREATE_VENV) == "TRUE" ]; then . .venv/bin/activate; fi; \
	aws-vault exec sandbox -- dvc pull; \

push: .venv
	@ if [ $(CREATE_VENV) == "TRUE" ]; then . .venv/bin/activate; fi; \
	aws-vault exec sandbox -- dvc push; \
	git push origin HEAD; \

.PHONY: install clean train pull push lint lint-check