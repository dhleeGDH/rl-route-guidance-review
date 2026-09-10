# -*- coding: utf-8 -*-
"""The screening stage counts, held once and checked against their own arithmetic.

WHY THIS EXISTS. Three cold-review rounds asked for a flow figure in the body, and the counts
existed only as a literal table in build_supplementary.py. A figure drawn from a second copy of
those numbers is the drift this project has already paid for twice. Both the supplement's table
and the body's figure now read this file, and the arithmetic linking the stages is asserted here
rather than trusted.

    python3 assembly_stages.py
"""
import io
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# Route 1 and route 3 (database searching and citation searching) after the scope screening they
# passed before pooling; route 2 is the IEEE Xplore export, unique after de-duplication.
ROUTE_13 = 86
ROUTE_2 = 1084
BOTH = 15                 # records returned by both, removed once in the merge
EXCLUDED_TA = 1052        # excluded at title and abstract
NOT_OBTAINED = 1          # full texts sought and not obtained
EXCLUDED_FT = 6           # excluded at full-text scope re-confirmation
VERSIONS_MERGED = 3       # version rows of one study merged into it
ABSTRACT_ONLY = 0         # 2026-09-10 (T-1991): none. Two of the three abstract-only records
                          # were read at full text and the third left the corpus, its full text
                          # unobtainable at any institution available to this study.

# 2026-09-10 (T-1984, T-1985). The name Fig. 1 and Supplementary Table S-6 must both give
# route_13. check_flow_stages.py reads it from the record beside the counts rather than out of
# the plotting script, where it was one English phrase this file could silently drop.
ARM_13_LABEL = "Database and citation searching"


def stages():
    pooled = ROUTE_13 + ROUTE_2 - BOTH
    screened_in = pooled - EXCLUDED_TA
    obtained = screened_in - NOT_OBTAINED
    full_text_recorded = obtained - EXCLUDED_FT - VERSIONS_MERGED
    corpus = full_text_recorded + ABSTRACT_ONLY
    s = {"route_13": ROUTE_13, "route_2": ROUTE_2, "both": BOTH,
         "identified": ROUTE_13 + ROUTE_2, "pooled": pooled,
         "excluded_title_abstract": EXCLUDED_TA, "screened_in": screened_in,
         "not_obtained": NOT_OBTAINED, "obtained": obtained,
         "excluded_full_text": EXCLUDED_FT, "versions_merged": VERSIONS_MERGED,
         "full_text_recorded": full_text_recorded, "abstract_only": ABSTRACT_ONLY,
         "corpus": corpus, "arm_13_label": ARM_13_LABEL}
    # every stage is a difference of the stage above it; a count changed in isolation fails here
    assert s["identified"] - s["both"] == s["pooled"], s
    assert s["pooled"] - s["excluded_title_abstract"] == s["screened_in"], s
    assert s["screened_in"] - s["not_obtained"] == s["obtained"], s
    assert s["obtained"] - s["excluded_full_text"] - s["versions_merged"] == s["full_text_recorded"], s
    assert s["full_text_recorded"] + s["abstract_only"] == s["corpus"], s
    assert s["corpus"] == 93, s
    assert s["arm_13_label"], s
    return s


def main():
    s = stages()
    for k, v in s.items():
        print("  %-26s %6s" % (k, v))
    with io.open(os.path.join(HERE, "assembly_stages.json"), "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("  wrote assembly_stages.json")


if __name__ == "__main__":
    main()
