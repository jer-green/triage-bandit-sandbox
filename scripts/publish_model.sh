#!/bin/bash
GIT_BRANCH="$(git rev-parse --symbolic-full-name --abbrev-ref HEAD)"
VERSION=$(poetry version -s)

# NOTE: Ideally, we can move this config to a base image,
# so it does not have to be run on every build.
poetry config repositories.babylonhealth $ARTIFACTORY_PYPI_URL
poetry build
poetry publish -r babylonhealth -u $ARTIFACTORY_PYPI_USER -p $ARTIFACTORY_PYPI_API_KEY

