#!/bin/bash
#
# Deploys the Sphinx website locally.
#
# Usage:
#
#   ./scripts/deploy_sphinx_website_locally.sh

if [[ "$PWD" =~ scripts$ ]]; then
    echo "FATAL ERROR."
    echo "Please run the script from the project root. "
    echo "Present working director: $PWD"
    echo " "
    echo "Tip: like this"
    echo " "
    echo "  ./scripts/deploy_sphinx_website_locally.sh"
    echo " "
    exit 42
fi

sphinx-build docs docs/_build/ --fail-on-warning

# cd docs
# make html
xdg-open docs/_build/html/index.html
