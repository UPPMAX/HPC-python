#!/bin/bash
#
# Converts the RST files to Markdown.
#
# Do not run this locally: it is only used to have a Markdown spell checker
#
# Usage:
#
#   ./scripts/convert_rst_to_mk.sh

if [[ "$PWD" =~ scripts$ ]]; then
    echo "FATAL ERROR."
    echo "Please run the script from the project root. "
    echo "Present working director: $PWD"
    echo " "
    echo "Tip: like this"
    echo " "
    echo "  ./scripts/convert_rst_to_mk.sh"
    echo " "
    exit 42
fi

find ./ -iname "*.rst" -type f -exec sh -c 'pandoc "${0}" -o "${0%.rst}.md"' {} \;

