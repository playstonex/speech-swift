#!/usr/bin/env bash
# Run one filtered test shard with a fresh xctest process per selected class.
#
# SwiftPM normally executes every class selected by --filter in one process.
# Model-backed classes can retain multi-GB MLX/CoreML state until that process
# exits, so a matrix shard containing several classes can accumulate enough
# memory to fail later, otherwise healthy tests. This runner preserves the
# shard's include/skip selection while splitting it at class boundaries.
#
# Usage:
#   scripts/test_shard_isolated.sh '<include-regex>' ['<skip-regex>']
#
# Environment:
#   SHARD_TEST_LOG_DIR=.e2e-logs/shard  Override the log directory.
#   SHARD_TEST_LIST_ONLY=1              Discover and summarize without running.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

INCLUDE_FILTER="${1:-}"
SKIP_FILTER="${2:-}"
LOG_DIR="${SHARD_TEST_LOG_DIR:-.e2e-logs/shard}"
LIST_ONLY="${SHARD_TEST_LIST_ONLY:-0}"

if [[ -z "$INCLUDE_FILTER" ]]; then
  echo "usage: $0 '<include-regex>' ['<skip-regex>']" >&2
  exit 64
fi

mkdir -p "$LOG_DIR/classes"
RAW_TESTS="$LOG_DIR/all-tests.raw.txt"
ALL_TESTS="$LOG_DIR/all-tests.txt"
INCLUDED_TESTS="$LOG_DIR/included-tests.txt"
SELECTED_TESTS="$LOG_DIR/selected-tests.txt"
SELECTED_CLASSES="$LOG_DIR/selected-classes.txt"
SUMMARY="$LOG_DIR/summary.txt"
: > "$SUMMARY"
OVERALL=0

record() { # status class detail
  printf '%-6s %-65s %s\n' "$1" "$2" "$3" | tee -a "$SUMMARY"
}

escape_regex() {
  local input="$1"
  local escaped=""
  local char
  local i

  for ((i = 0; i < ${#input}; i++)); do
    char="${input:$i:1}"
    case "$char" in
      '.'|'^'|'$'|'*'|'+'|'?'|'('|')'|'['|']'|'{'|'}'|'|'|\\)
        escaped="${escaped}\\${char}"
        ;;
      *)
        escaped="${escaped}${char}"
        ;;
    esac
  done
  printf '%s' "$escaped"
}

exact_filter_for_file() {
  local tests_file="$1"
  local filter=""
  local test_id
  local escaped

  while IFS= read -r test_id || [[ -n "$test_id" ]]; do
    escaped="$(escape_regex "$test_id")"
    if [[ -n "$filter" ]]; then
      filter="${filter}|"
    fi
    filter="${filter}${escaped}"
  done < "$tests_file"
  printf '^(%s)$' "$filter"
}

echo "== Listing tests =="
if ! swift test list --skip-build --disable-sandbox > "$RAW_TESTS" 2> "$LOG_DIR/list.err"; then
  echo "swift test list failed (build tests first with swift build --build-tests --disable-sandbox)" >&2
  cat "$LOG_DIR/list.err" >&2
  exit 2
fi

# Test specifiers have the form Target.Class/testMethod. Keep discovery output
# separate so incidental SwiftPM status lines can never become filter entries.
awk 'index($0, "/") > 0' "$RAW_TESTS" | LC_ALL=C sort -u > "$ALL_TESTS"

grep -E -- "$INCLUDE_FILTER" "$ALL_TESTS" > "$INCLUDED_TESTS"
GREP_STATUS=$?
if [[ $GREP_STATUS -gt 1 ]]; then
  echo "invalid include regex: $INCLUDE_FILTER" >&2
  exit 65
fi

if [[ -n "$SKIP_FILTER" ]]; then
  grep -Ev -- "$SKIP_FILTER" "$INCLUDED_TESTS" > "$SELECTED_TESTS"
  GREP_STATUS=$?
  if [[ $GREP_STATUS -gt 1 ]]; then
    echo "invalid skip regex: $SKIP_FILTER" >&2
    exit 65
  fi
else
  cp "$INCLUDED_TESTS" "$SELECTED_TESTS"
fi

if [[ ! -s "$SELECTED_TESTS" ]]; then
  echo "no tests matched include '$INCLUDE_FILTER' after skip '$SKIP_FILTER'" >&2
  exit 3
fi

sed 's|/.*||' "$SELECTED_TESTS" | LC_ALL=C sort -u > "$SELECTED_CLASSES"
SELECTED_COUNT="$(wc -l < "$SELECTED_TESTS" | tr -d ' ')"
CLASS_COUNT="$(wc -l < "$SELECTED_CLASSES" | tr -d ' ')"
echo "Selected $SELECTED_COUNT tests across $CLASS_COUNT classes"

CLASS_INDEX=0
while IFS= read -r test_class || [[ -n "$test_class" ]]; do
  CLASS_INDEX=$((CLASS_INDEX + 1))
  SAFE_CLASS="$(printf '%s' "$test_class" | tr -c '[:alnum:]_-' '_')"
  CLASS_TESTS="$LOG_DIR/classes/$SAFE_CLASS.tests.txt"
  CLASS_LOG="$LOG_DIR/classes/$SAFE_CLASS.log"
  awk -v prefix="$test_class/" 'index($0, prefix) == 1' "$SELECTED_TESTS" > "$CLASS_TESTS"
  TEST_COUNT="$(wc -l < "$CLASS_TESTS" | tr -d ' ')"

  if [[ "$LIST_ONLY" == "1" ]]; then
    record READY "$test_class" "$TEST_COUNT tests"
    continue
  fi

  EXACT_FILTER="$(exact_filter_for_file "$CLASS_TESTS")"
  echo "== [$CLASS_INDEX/$CLASS_COUNT] $test_class ($TEST_COUNT tests) =="
  swift test --skip-build --disable-sandbox --no-parallel --filter "$EXACT_FILTER" 2>&1 | tee "$CLASS_LOG"
  TEST_STATUS=${PIPESTATUS[0]}
  if [[ $TEST_STATUS -eq 0 ]]; then
    record PASS "$test_class" "$TEST_COUNT selected tests"
  else
    OVERALL=1
    record FAIL "$test_class" "exit $TEST_STATUS; see $CLASS_LOG"
  fi
done < "$SELECTED_CLASSES"

echo "== Summary =="
cat "$SUMMARY"
if [[ "$LIST_ONLY" == "1" ]]; then
  echo "Selection only: $SELECTED_COUNT tests across $CLASS_COUNT classes"
elif [[ $OVERALL -eq 0 ]]; then
  echo "Overall: GREEN ($SELECTED_COUNT tests across $CLASS_COUNT isolated classes)"
else
  FAILURES="$(grep -c '^FAIL' "$SUMMARY" || true)"
  echo "Overall: RED ($FAILURES failing classes)"
fi
exit $OVERALL
