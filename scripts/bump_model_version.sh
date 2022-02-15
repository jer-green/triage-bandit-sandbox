#!/bin/bash
BUMP_TYPE=$1

if [ -z "$BUMP_TYPE" ]
then
    echo 'Please specify a d "bump type" (patch, minor, major)'
    echo "e.g. bump_model_version.sh patch"
    exit 0
fi

if ! [[ "$BUMP_TYPE" == "patch" ||  "$BUMP_TYPE" == "minor" ||  "$BUMP_TYPE" == "major" ]]
then
    echo "Version bump argument must be either 'patch', 'minor', or 'major'. You specified '$BUMP_TYPE'."
    exit 0
fi

# Now, we bump the version and git tag it.
GIT_BRANCH="$(git rev-parse --symbolic-full-name --abbrev-ref HEAD)"
EXISTING_VERSION=$(poetry version -s)

# Bump poetry version.
poetry version $BUMP_TYPE
VERSION=$(poetry version -s)

# Add bumped version to github, and push.
git add pyproject.toml
git commit -m "Bump version from v$EXISTING_VERSION -> v$VERSION"

# Tag version
git tag v$VERSION

echo "Don't forget to push your commit and tags by running"
echo "git push --set-upstream origin $GIT_BRANCH --tags"

