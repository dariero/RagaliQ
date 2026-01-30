#!/bin/bash
# Cleanup legacy priority-* and size-* labels from all issues

echo "Searching for issues with priority-* or size-* labels..."

# Find all issues with these labels
ISSUES=$(gh issue list --repo dariero/RagaliQ --state all --limit 1000 --json number,labels --jq '.[] | select(.labels[].name | test("^(priority-|size-)")) | .number')

if [ -z "$ISSUES" ]; then
  echo "✓ No issues found with legacy labels"
  exit 0
fi

echo "Found issues to clean: $ISSUES"
echo ""

for ISSUE_NUM in $ISSUES; do
  echo "Processing issue #$ISSUE_NUM..."

  # Get current labels
  LABELS=$(gh issue view $ISSUE_NUM --json labels --jq '.labels[].name')

  # Build remove list
  REMOVE_LABELS=""
  for LABEL in $LABELS; do
    if [[ "$LABEL" =~ ^priority- ]] || [[ "$LABEL" =~ ^size- ]]; then
      if [ -z "$REMOVE_LABELS" ]; then
        REMOVE_LABELS="$LABEL"
      else
        REMOVE_LABELS="$REMOVE_LABELS,$LABEL"
      fi
    fi
  done

  if [ -n "$REMOVE_LABELS" ]; then
    echo "  Removing labels: $REMOVE_LABELS"
    gh issue edit $ISSUE_NUM --remove-label "$REMOVE_LABELS"
    echo "  ✓ Cleaned issue #$ISSUE_NUM"
  fi

  echo ""
done

echo "✓ Cleanup complete!"
