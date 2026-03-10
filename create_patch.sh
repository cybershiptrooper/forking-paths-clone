#!/bin/bash
# Creates a combined patch file including both tracked and untracked files

PATCH_FILE="all_changes.patch"

echo "Creating patch for tracked changes..."
git diff HEAD > "$PATCH_FILE"

echo "Adding untracked files..."
git ls-files --others --exclude-standard | while read -r file; do
    git diff --no-index /dev/null "$file" >> "$PATCH_FILE" 2>/dev/null
done

echo "Patch saved to $PATCH_FILE"
echo "To apply: git apply --allow-empty $PATCH_FILE"