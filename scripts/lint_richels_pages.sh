#!/bin/bash
#
# Lint the pages that Richel
# is responsible for
#
# Usage:
#
#   ./scripts/lint_richels_pages.sh

if [[ "$PWD" =~ scripts$ ]]; then
    echo "FATAL ERROR."
    echo "Please run the script from the project root. "
    echo "Present working director: $PWD"
    echo " "
    echo "Tip: like this"
    echo " "
    echo "  ./scripts/lint_richels_pages.sh"
    echo " "
    exit 42
fi

sphinx-lint docs/prereqs.rst

sphinx-lint docs/common/naiss_projects_overview.rst
sphinx-lint docs/common/use_tarball.rst
sphinx-lint docs/schedule.rst
sphinx-lint docs/day2/use_packages.rst
sphinx-lint docs/day2/intro.rst
sphinx-lint docs/index.rst
