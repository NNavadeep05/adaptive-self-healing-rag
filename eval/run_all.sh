#!/usr/bin/env bash
# Run every evaluation in priority order. Safe to re-run: finished cells are
# skipped, and a system that hits the Groq daily token limit is paused until
# the next invocation.
#
#   bash eval/run_all.sh            (from the repository root)
set -u
PY="${PYTHON:-python}"
ORIGINAL="Q01 Q02 Q03 Q04 Q05 Q06 Q07 Q08 Q09 Q10"
ADDED="Q21 Q22 Q23 Q24 Q25 Q26 Q27 Q28 Q29 Q30 Q31 Q32 Q33 Q34 Q35 Q36 Q37 Q38 Q39 Q40"

# 1. v2 on the original 10 questions (directly comparable to the published v1 runs)
$PY eval/compare_systems.py --bucket straightforward --systems hybrid_single_pass self_healing_v2 --ids $ORIGINAL
# 2. single-pass systems on the 20 added questions (1 LLM call each)
$PY eval/compare_systems.py --bucket straightforward --systems plain_rag_k5 wide_rerank_k11 hybrid_single_pass --ids $ADDED
# 3. single-pass systems on the hard bucket
$PY eval/compare_systems.py --bucket hard --systems wide_rerank_k11 hybrid_single_pass
# 4. loops on the remaining questions (most expensive: 3+ LLM calls per attempt)
$PY eval/compare_systems.py --bucket straightforward --systems self_healing_v2 self_healing --ids $ADDED
$PY eval/compare_systems.py --bucket hard --systems self_healing_v2 self_healing
# 5. statistics over whatever has been collected
$PY eval/analysis.py
