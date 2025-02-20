#!/bin/bash
#
# Deploys the MkDocs website locally.
#
# Usage:
#
#   ./scripts/deploy_mkdocs_website_locally.sh

if [[ "$PWD" =~ scripts$ ]]; then
    echo "FATAL ERROR."
    echo "Please run the script from the project root. "
    echo "Present working director: $PWD"
    echo " "
    echo "Tip: like this"
    echo " "
    echo "  ./scripts/deploy_mkdocs_website_locally.sh"
    echo " "
    exit 42
fi

python3 -m venv hpc_python_venv
source hpc_python_venv/bin/activate
pip install -r requirements.txt
mkdocs serve
