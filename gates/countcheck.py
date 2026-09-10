#!/usr/bin/env python3
"""countcheck.py : tallies Table A-1 (Appendix A) from its markdown source and compares the tallies
with the counts the manuscript and the Supplementary print. Prints one line per check; exits 1 on
any FAIL.

Two kinds of check run. The first tallies a column of Table A-1 and compares the tally with a fixed
expected value, which is the v925 census. The second reads the printed text: every "N of M", every
"N studies" and every cell of Table II, Table S-14 and Table S-15 is extracted from the sources, is
looked up in CLAIMS, and is recomputed from Table A-1. A figure extracted from the text and absent
from CLAIMS and from NOT_DERIVABLE fails the check, so a new or altered count cannot pass unnoticed.
NOT_DERIVABLE records the figures no tally over Table A-1 can produce, with the reason for each.

If a ticket legitimately changes a count, edit the entry here in the same ticket and say so in the
report.
"""
import os
import re
import sys
from collections import namedtuple

TABLE = "handoff/manuscript/source_md/10_appendix_corpus_list.md"
SUPP = "handoff/manuscript/source_md/supplementary.md"
BUILDER = "handoff/manuscript/builders/build_integrated_docx.py"
SRC = ["draft_abstract_intro.md", "draft_section2.md", "draft_section3.md", "draft_section4.md",
       "draft_section5.md", "draft_section6.md", "draft_section7_8.md", "supplementary.md",
       "11_appendix_search_protocol.md"]

Row = namedtuple("Row", "ref mark fam act st rw bd cm cd enc fs gn gam")

# One study per row of the twelve-column table
ROW_RX = re.compile(
    r"\|\s*\[(\d+)\]([‡†]*)\s*\|\s*(dq|ma|pg|tb|di|mb|un)\s*\|\s*(link|path|oth|unc)\s*\|"
    r"\s*(inst|local|roll|unc)\s*\|\s*(ind|mix|sys|unc)\s*\|\s*(yes|no|unc)\s*\|\s*(yes|no|unc)\s*\|"
    r"\s*(yes|no|unc)\s*\|\s*(raw|graph|other|unc)\s*\|\s*(enf|mask|no|unc)\s*\|\s*(yes|no|unc)\s*\|"
    r"\s*([^|]+?)\s*\|")

# 2026-09-10 (T-1992): the census moves to the 93-study record of T-1991, which read [1098] and
# [1109] at full text and removed [1088], whose full text no institution available to this study
# can reach. Every predicate below was re-read against what it counts rather than only its value:
# the three that did not move, the forecast tally and the completion, code and generalization
# numerators, are unchanged because their studies did not move, not because the check is stale.
EXPECTED = {
    "rows": 93,
    "fam": {"tb": 12, "ma": 29, "dq": 28, "pg": 12, "mb": 6, "di": 3, "un": 3},
    "act": {"link": 68, "path": 15},                 # other + unc = 10
    "state": {"inst": 71, "unc": 5},                 # local + roll = 17
    "reward": {"ind": 60, "mix": 21, "sys": 10, "unc": 2},
    # the unclear entry is kept at 0 rather than dropped: it asserts that no reviewed study is
    # left undetermined on the boundary, which is the fact the recomposition produced.
    "boundary": {"yes": 10, "no": 83, "unc": 0},
    "compl_yes": 16,
    "code_yes": 10,
    "gen_yes": 14,
    "enc": {"raw": 64, "graph": 16, "other": 5, "unc": 8},
    "feas": {"enf": 61, "mask": 15, "no": 5, "unc": 12},
    "link_ind_bno": 45,
    "link_inst_ind": 40,
}


# ---------------------------------------------------------------- predicates over one row
def P(**kw):
    """A row test on field values, each given as a value or a tuple of accepted values."""
    def f(r):
        for k, v in kw.items():
            val = getattr(r, k)
            if isinstance(v, tuple):
                if val not in v:
                    return False
            elif val != v:
                return False
        return True
    return f


def AND(*ps):
    return lambda r: all(p(r) for p in ps)


class ROWS(object):
    """A figure computed over the whole table rather than by counting matching rows."""
    def __init__(self, fn):
        self.fn = fn

ALL = lambda r: True
# 2026-09-10 (T-1992): FULL and ALL now select the same 93 rows. Table A-1 carries no dagger
# since T-1991, so every reviewed study is recorded at full text and the two denominators the
# manuscript distinguished have become one. The predicate is kept rather than replaced by ALL:
# it still means "recorded at full text", it returns to selecting a subset the moment a study is
# admitted on an abstract again, and the CLAIMS keys that name it say which denominator they
# intend. A reader of a failure here should know the two coincide.
FULL = lambda r: "†" not in r.mark                    # the full texts, all 93 of them
THREE = P(act=("link", "path", "oth"), st=("inst", "local", "roll"), rw=("ind", "mix", "sys"))
GEN_ASSESS = lambda r: r.gn != "unc"                  # the 89 assessable on transfer
FORECAST = P(st=("local", "roll"))
GRP = P(act="link", rw="ind", bd="no")                # the 44 studies of Section IV
OUTSIDE = lambda r: r.act == "link" and not GRP(r)    # the other 23 link-level studies
GAM = lambda r: re.match(r"^[0-9.]+$", r.gam) is not None
GAM_THRESHOLD = 0.9714                                # Eq. (7) on the bespoke grid, Section V-B

TRIPLE_FIELDS = ("act", "st", "rw")
TRIPLE_VALUES = (("link", "path", "oth"), ("inst", "local", "roll"), ("ind", "mix", "sys"))


def triples(rows):
    """The action, state-form and reward triple of every study settled on all three fields."""
    out = {}
    for r in rows:
        k = tuple(getattr(r, f) for f in TRIPLE_FIELDS)
        if all(v in vals for v, vals in zip(k, TRIPLE_VALUES)):
            out[k] = out.get(k, 0) + 1
    return out


def absent_triples(rows):
    """The combinations the three fields allow and the corpus does not use."""
    import itertools
    return sorted(set(itertools.product(*TRIPLE_VALUES)) - set(triples(rows)))


# ---------------------------------------------------------------- the printed figures
# Key: source file, the figure as printed, and its occurrence index in that file.
# Value: the numerator test, the denominator set for an "N of M" figure (None for "N studies"),
# and the wording the figure carries.
CLAIMS = {
    # ---- abstract and introduction
    "draft_abstract_intro.md|83 of the 93|0": (P(bd="no"), ALL, "boundary unaddressed"),
    "draft_abstract_intro.md|93 studies|0": (ALL, None, "the corpus"),
    "draft_abstract_intro.md|93 studies|1": (ALL, None, "the corpus"),
    "draft_abstract_intro.md|10 of the 93|0": (P(bd="yes"), ALL, "boundary addressed"),
    "draft_abstract_intro.md|93 studies|2": (ALL, None, "the corpus"),
    "draft_abstract_intro.md|93 studies|3": (ALL, None, "the corpus"),
    "draft_abstract_intro.md|40 studies|0": (P(act="link", st="inst", rw="ind"), None, "modal design"),
    # T-1933: the contribution paragraph now names the census denominator, and the opening of
    # Section V names the group the census leaves undetermined. Both are tallies of Table A-1.
    "draft_abstract_intro.md|94 studies|4": (ALL, None, "the corpus"),
    "draft_section5.md|45 studies|0": (GRP, None, "the group of Section IV"),
    "draft_section5.md|45 studies|1": (GRP, None, "the group of Section IV"),
    "draft_abstract_intro.md|37 studies|0": (P(act="link", st="inst", rw="ind", bd="no"), None,
                                             "modal design leaving the boundary unaddressed"),
    # ---- Section II
    "draft_section2.md|1 of these 15|0": (P(act="path", bd="yes"), P(act="path"), "candidate-path with a boundary"),
    "draft_section2.md|8 of the 68|0": (P(act="link", bd="yes"), P(act="link"), "link-level with a boundary"),
    "draft_section2.md|68 studies|0": (P(act="link"), None, "link-level action"),
    "draft_section2.md|15 studies|0": (P(act="path"), None, "candidate-path action"),
    "draft_section2.md|15 studies|1": (P(act="path"), None, "candidate-path action"),
    "draft_section2.md|68 studies|1": (P(act="link"), None, "link-level action"),
    "draft_section2.md|93 studies|0": (ALL, None, "the corpus"),
    "draft_section2.md|93 full texts|0": (FULL, None, "the full texts"),
    # T-1993: II-B named the full-text denominator here until every reviewed study became a full
    # text. The surviving variation is the transfer axis, whose denominator the sentence now gives.
    "draft_section2.md|91 studies|0": (GEN_ASSESS, None, "assessable on transfer"),
    # ---- Section IV
    "draft_section4.md|12 of the 93|0": (P(fam="tb"), ALL, "tabular family"),
    # T-1937: the family profiles added to IV-E and the Table S-22 rows recovered into IV-G.
    # The IV-E figures are tallies of two Appendix A columns; the IV-G figures are the full-text
    # scan, which no column of Appendix A produces, and are excused with the standing S-22 reason.
    "draft_section4.md|11 of the 12|1": (P(fam="pg", rw="ind"), P(fam="pg"), "policy-gradient, individual reward"),
    "draft_section4.md|93 studies|5": (ALL, None, "the corpus"),
    # T-1945: the node-count sentence of IV-G names the denominator a third time, since the
    # statistics beside it are over the 63 values those texts state and not over the texts.
    "draft_section4.md|52 of the 64|0": (lambda r: r.fam != "ma" and r.rw == "ind",
                                         lambda r: r.fam != "ma", "single-agent with an individual reward"),
    "draft_section4.md|8 of the 29|0": (P(fam="ma", rw="ind"), P(fam="ma"), "multi-agent with an individual reward"),
    "draft_section4.md|93 studies|0": (ALL, None, "the corpus"),
    "draft_section4.md|93 studies|1": (ALL, None, "the corpus"),
    "draft_section4.md|93 full texts|0": (FULL, None, "the full texts"),
    # T-1994: IV-G names the scan population. The scan reads every reviewed study since the
    # rescan, so the phrase is the corpus denominator and no longer a subset of it.
    "draft_section4.md|93 full texts|1": (FULL, None, "the full texts"),
    "draft_section4.md|93 full texts|2": (FULL, None, "the full texts"),
    "draft_section4.md|3 of the 40|0": (P(act="link", st="inst", rw="ind", bd="yes"),
                                        P(act="link", st="inst", rw="ind"), "modal design with a boundary"),
    "draft_section4.md|40 studies|0": (P(act="link", st="inst", rw="ind"), None, "modal design"),
    "draft_section4.md|15 studies|0": (P(act="link", st="inst", rw="mix"), None, "second combination"),
    "draft_section4.md|86 studies|0": (THREE, None, "assessable on the three fields"),
    "draft_section4.md|7 studies|0": (lambda r: not THREE(r), None, "unclear on one of the three fields"),
    # T-2001: IV-B splits the ten addressed boundaries by evidence class; the split is recorded in
    # the study record (boundary_split) since Table A-1 holds one value for both.
    "draft_section4.md|7 studies|1": (ROWS(lambda rows: boundary_split()[0]), None, "boundary: treatment stated"),
    "draft_section4.md|3 studies|0": (ROWS(lambda rows: boundary_split()[1]), None, "boundary: periphery named"),
    "draft_section4.md|3 of these 40|0": (P(act="link", st="inst", rw="ind", bd="yes"),
                                          P(act="link", st="inst", rw="ind"), "modal design with a boundary"),
    "draft_section4.md|10 of the 86|0": (AND(THREE, P(bd="yes")), THREE, "boundary addressed among the placed"),
    "draft_section4.md|40 studies|1": (P(act="link", st="inst", rw="ind"), None, "modal design"),
    "draft_section4.md|86 studies|1": (THREE, None, "assessable on the three fields"),
    "draft_section4.md|45 studies|0": (GRP, None, "the group"),
    "draft_section4.md|45 studies|1": (GRP, None, "the group"),
    "draft_section4.md|10 of the 93|0": (P(bd="yes"), ALL, "boundary addressed"),
    "draft_section4.md|93 studies|2": (ALL, None, "the corpus"),
    "draft_section4.md|10 studies|0": (P(bd="yes"), None, "boundary addressed"),
    "draft_section4.md|83 studies|0": (lambda r: r.bd != "yes", None, "boundary not addressed or unclear"),
    "draft_section4.md|14 studies|0": (AND(FORECAST, P(bd="no")), None, "forecast state alone"),
    "draft_section4.md|7 studies|4": (P(st="inst", bd="yes"), None, "boundary alone"),
    "draft_section4.md|64 studies|0": (P(st="inst", bd="no"), None, "neither field"),
    "draft_section4.md|5 studies|0": (lambda r: r.st == "unc" or r.bd == "unc", None, "unclear on one field"),
    "draft_section4.md|5 studies|1": (lambda r: r.st == "unc" or r.bd == "unc", None, "unclear on one field"),
    "draft_section4.md|10 studies|1": (P(bd="yes"), None, "boundary addressed"),
    "draft_section4.md|3 studies|2": (AND(FORECAST, P(bd="yes")), None, "forecast state with a boundary"),
    "draft_section4.md|3 studies|3": (AND(FORECAST, P(bd="yes")), None, "forecast state with a boundary"),
    "draft_section4.md|3 studies|4": (AND(FORECAST, P(bd="yes")), None, "forecast state with a boundary"),
    "draft_section4.md|45 studies|2": (GRP, None, "the group"),
    "draft_section4.md|8 of the 10|0": (AND(OUTSIDE, P(bd="yes")), P(bd="yes"),
                                        "addressed boundaries outside the group"),
    "draft_section4.md|15 of these 23|0": (AND(OUTSIDE, lambda r: r.bd != "yes", P(rw=("mix", "sys"))), OUTSIDE,
                                           "outside the group, unaddressed with an aggregate reward"),
    "draft_section4.md|2 studies|0": (lambda r: r.bd == "yes" and r.act != "link", None,
                                      "addressed boundary at another action"),
    "draft_section4.md|8 studies|0": (AND(OUTSIDE, P(bd="yes")), None, "addressed boundaries outside the group"),
    # T-1954: the paragraph on the boundary field against the state form moved behind the two
    # paragraphs below it, so every phrase they share renumbers. The order here follows the source.
    "draft_section4.md|15 studies|1": (AND(OUTSIDE, lambda r: r.bd != "yes", P(rw=("mix", "sys"))),
                                       None, "outside the group, unaddressed with an aggregate reward"),
    "draft_section4.md|4 studies|1": (AND(OUTSIDE, P(bd="yes", rw="ind")), None, "of those, an individual reward"),
    "draft_section4.md|23 studies|0": (OUTSIDE, None, "link-level outside the group"),
    "draft_section4.md|60 studies|0": (P(rw="ind"), None, "individual reward"),
    "draft_section4.md|21 studies|0": (P(rw="mix"), None, "mixed reward"),
    "draft_section4.md|10 studies|2": (P(rw="sys"), None, "system-level reward"),
    "draft_section4.md|2 studies|1": (P(rw="unc"), None, "unclear reward"),
    "draft_section4.md|45 studies|3": (GRP, None, "the group"),
    "draft_section4.md|83 studies|1": (P(bd="no"), None, "boundary unaddressed"),
    "draft_section4.md|45 studies|4": (GRP, None, "the group"),
    "draft_section4.md|45 reports|0": (GRP, None, "the group"),
    "draft_section4.md|45 studies|5": (GRP, None, "the group"),
    "draft_section4.md|6 studies|3": (P(fam="mb"), None, "model-based hybrids"),
    "draft_section4.md|76 of the 93|0": (P(fs=("enf", "mask")), ALL, "feasibility restricted"),
    "draft_section4.md|93 studies|3": (ALL, None, "the corpus"),
    "draft_section4.md|61 studies|0": (P(fs="enf"), None, "enforced by construction"),
    "draft_section4.md|15 studies|3": (P(fs="mask"), None, "explicit mask"),
    "draft_section4.md|5 studies|2": (P(fs="no"), None, "not enforced"),
    "draft_section4.md|12 studies|0": (P(fs="unc"), None, "feasibility unclear"),
    "draft_section4.md|45 studies|6": (GRP, None, "the group"),
    "draft_section4.md|64 studies|1": (P(enc="raw"), None, "raw feature vector"),
    "draft_section4.md|16 studies|0": (P(enc="graph"), None, "graph encoder"),
    "draft_section4.md|5 studies|3": (P(enc="other"), None, "another learned encoder"),
    "draft_section4.md|8 studies|1": (P(enc="unc"), None, "encoder unstated"),
    "draft_section4.md|15 studies|4": (P(act="path"), None, "candidate-path action"),
    "draft_section4.md|9 of the 16|0": (P(fam="dq", enc="graph"), P(enc="graph"), "value-based deep graph encoders"),
    "draft_section4.md|22 of the 29|0": (P(fam="ma", enc="raw"), P(fam="ma"), "multi-agent raw vectors"),
    "draft_section4.md|29 studies|0": (P(fam="ma"), None, "multi-agent family"),
    # T-1954: the sentence carrying this occurrence is the model-based split, since T-1945
    # deleted IV-D's encoder-by-family paragraph and the multi-agent graph-encoder figure with it.
    "draft_section4.md|3 studies|6": (P(fam="mb", rw="ind"), None, "model-based hybrid, individual reward"),
    "draft_section4.md|7 of the 16|0": (P(enc="graph", gn="yes"), P(enc="graph"), "graph encoders with transfer"),
    "draft_section4.md|6 of the 62|0": (P(enc="raw", gn="yes"), AND(P(enc="raw"), GEN_ASSESS),
                                        "raw vectors with transfer"),
    "draft_section4.md|12 of the 16|0": (P(enc="graph", rw="ind"), P(enc="graph"), "graph encoders, individual reward"),
    "draft_section4.md|41 of the 64|0": (P(enc="raw", rw="ind"), P(enc="raw"), "raw vectors, individual reward"),
    "draft_section4.md|7 studies|5": (P(enc="graph", gn="yes"), None, "graph encoders with transfer"),
    "draft_section4.md|20 of the 29|0": (P(fam="ma", rw=("sys", "mix")), P(fam="ma"), "multi-agent aggregate reward"),
    "draft_section4.md|29 studies|1": (P(fam="ma"), None, "multi-agent family"),
    "draft_section4.md|4 studies|2": (P(fam="ma", bd="yes"), None, "multi-agent with a boundary"),
    "draft_section4.md|2 of the 12|0": (P(fam="tb", bd="yes"), P(fam="tb"), "tabular with a boundary"),
    "draft_section4.md|12 studies|1": (P(fam="tb"), None, "tabular family"),
    "draft_section4.md|12 studies|2": (P(fam="tb"), None, "tabular family"),
    "draft_section4.md|12 studies|3": (P(fam="tb"), None, "tabular family"),
    "draft_section4.md|11 of the 12|0": (P(fam="pg", rw="ind"), P(fam="pg"), "policy-gradient, individual reward"),
    "draft_section4.md|1 of the 12|0": (P(fam="pg", bd="yes"), P(fam="pg"), "policy-gradient with a boundary"),
    "draft_section4.md|12 studies|4": (P(fam="pg"), None, "policy-gradient family"),
    "draft_section4.md|12 studies|5": (P(fam="pg"), None, "policy-gradient family"),
    "draft_section4.md|3 studies|7": (P(fam="un"), None, "unspecified family"),
    "draft_section4.md|3 studies|8": (P(fam="un"), None, "unspecified family"),
    "draft_section4.md|2 of these 29|0": (P(fam="ma", gn="yes"), P(fam="ma"), "multi-agent with transfer"),
    "draft_section4.md|29 studies|2": (P(fam="ma"), None, "multi-agent family"),
    "draft_section4.md|29 studies|3": (P(fam="ma"), None, "multi-agent family"),
    "draft_section4.md|5 of the 31|0": (P(rw=("sys", "mix"), bd="yes"), P(rw=("sys", "mix")),
                                        "aggregate objective with a boundary"),
    "draft_section4.md|5 of the 60|0": (P(rw="ind", bd="yes"), P(rw="ind"), "individual objective with a boundary"),
    "draft_section4.md|1 of 3|0": (P(fam="un", bd="yes"), P(fam="un"), "unspecified family with a boundary"),
    "draft_section4.md|2 of 12|0": (P(fam="tb", bd="yes"), P(fam="tb"), "tabular family with a boundary"),
    "draft_section4.md|14 of the 91|0": (P(gn="yes"), GEN_ASSESS, "transfer stated"),
    "draft_section4.md|31 studies|0": (P(rw=("sys", "mix")), None, "aggregate objective"),
    "draft_section4.md|60 studies|1": (P(rw="ind"), None, "individual objective"),
    "draft_section4.md|91 studies|0": (GEN_ASSESS, None, "assessable on transfer"),
    "draft_section4.md|91 studies|1": (GEN_ASSESS, None, "assessable on transfer"),
    "draft_section4.md|16 studies|1": (P(enc="graph"), None, "graph encoder"),
    "draft_section4.md|64 studies|2": (P(enc="raw"), None, "raw feature vector"),
    "draft_section4.md|70 of the 93|0": (AND(FULL, lambda r: r.bd != "yes" and r.cm != "yes"), FULL,
                                         "neither record present"),
    "draft_section4.md|13 of these 16|0": (P(cm="yes", bd="no"), P(cm="yes"), "a rate with no boundary"),
    "draft_section4.md|93 studies|6": (FULL, None, "the full texts"),
    "draft_section4.md|16 studies|2": (P(cm="yes"), None, "a trip completion rate"),
    "draft_section4.md|16 studies|3": (P(cm="yes"), None, "a trip completion rate"),
    "draft_section4.md|16 of 93|0": (AND(FULL, P(cm="yes")), FULL, "a trip completion rate"),
    "draft_section4.md|93 studies|4": (ALL, None, "the corpus"),
    # ---- Section IV-F, the design regularities
    "draft_section4.md|49 of the 68|0": (P(act="link", rw="ind"), P(act="link"),
                                         "link-level with an individual reward"),
    "draft_section4.md|10 of the 15|0": (P(act="path", rw="ind"), P(act="path"),
                                        "candidate-path with an individual reward"),
    "draft_section4.md|68 studies|0": (P(act="link"), None, "link-level action"),
    "draft_section4.md|6 studies|4": (P(act="oth"), None, "guidance-parameter action"),
    "draft_section4.md|15 studies|5": (P(act="path"), None, "candidate-path action"),
    # T-1996: the widened noun list reaches these four; each is the same tally Section IV-F prints
    # as "16 of the 27" and "3 of the 11", registered there since T-1801.
    "draft_section4.md|16 combinations|0": (ROWS(lambda rows: len(triples(rows))), None, "IV-F combinations present"),
    "draft_section4.md|16 combinations|1": (ROWS(lambda rows: len(triples(rows))), None, "IV-F combinations present"),
    "draft_section4.md|11 combinations|0": (ROWS(lambda rows: len(absent_triples(rows))), None, "IV-F combinations absent"),
    "draft_section4.md|27 combinations|0": (ROWS(lambda rows: len(triples(rows)) + len(absent_triples(rows))), None,
                                           "the combinations the three fields allow"),
    "draft_section4.md|10 statements|0": (P(bd="yes"), None, "boundary addressed"),
    "draft_section4.md|16 of the 27|0": (ROWS(lambda rows: len(triples(rows))),
                                         ROWS(lambda rows: len(triples(rows)) + len(absent_triples(rows))),
                                         "combinations occurring of those allowed"),
    "draft_section4.md|3 of the 11|0": (ROWS(lambda rows: len([k for k in absent_triples(rows)
                                                               if k[0] == "oth" and k[2] == "ind"])),
                                        ROWS(lambda rows: len(absent_triples(rows))),
                                        "absences at a guidance-parameter action with an individual reward"),
    "draft_section4.md|3 of the 11|1": (ROWS(lambda rows: len([k for k in absent_triples(rows)
                                                               if k[1] == "roll" and k[2] == "mix"])),
                                        ROWS(lambda rows: len(absent_triples(rows))),
                                        "absences at a rollout state with a mixed reward"),
    "draft_section4.md|86 studies|2": (THREE, None, "assessable on the three fields"),
    "draft_section4.md|40 studies|2": (P(act="link", st="inst", rw="ind"), None, "the modal combination"),
    "draft_section4.md|8 of the 16|0": (AND(P(enc="graph"), FORECAST), P(enc="graph"),
                                        "graph encoders at a forecast-conditioned state"),
    "draft_section4.md|7 of the 64|0": (AND(P(enc="raw"), FORECAST), P(enc="raw"),
                                        "raw vectors at a forecast-conditioned state"),
    "draft_section4.md|16 studies|5": (P(enc="graph"), None, "graph encoder"),
    "draft_section4.md|64 studies|3": (P(enc="raw"), None, "raw feature vector"),
    "draft_section4.md|16 studies|6": (P(enc="graph"), None, "graph encoder"),
    "draft_section4.md|5 studies|4": (P(enc="other"), None, "another learned encoder"),
    "draft_section4.md|22 of the 29|1": (P(fam="ma", enc="raw"), P(fam="ma"), "multi-agent raw vectors"),
    "draft_section4.md|9 of the 28|0": (P(fam="dq", enc="graph"), P(fam="dq"),
                                        "value-based deep graph encoders"),
    "draft_section4.md|10 studies|3": (P(bd="yes"), None, "boundary addressed"),
    "draft_section4.md|10 studies|5": (P(bd="yes"), None, "boundary addressed"),
    # ---- Section V
    "draft_section5.md|60 of the 93|0": (P(rw="ind"), ALL, "individual reward"),
    "draft_section5.md|68 of the 93|0": (P(act="link"), ALL, "link-level action"),
    "draft_section5.md|83 of the 93|0": (P(bd="no"), ALL, "boundary unaddressed"),
    "draft_section5.md|93 studies|0": (ALL, None, "the corpus"),
    "draft_section5.md|93 studies|1": (ALL, None, "the corpus"),
    "draft_section5.md|93 studies|2": (ALL, None, "the corpus"),
    "draft_section5.md|38 of the 93|0": (AND(FULL, GAM), FULL, "a stated discount factor"),
    "draft_section5.md|37 of the 38|0": (lambda r: GAM(r) and float(r.gam) < 1.0, GAM, "a discount below one"),
    "draft_section5.md|22 of the 38|0": (lambda r: GAM(r) and float(r.gam) >= GAM_THRESHOLD, GAM,
                                         "a discount at or above the Eq. (7) threshold"),
    "draft_section5.md|93 full texts|0": (FULL, None, "the full texts"),
    # ---- Section VI
    "draft_section6.md|60 studies|0": (P(rw="ind"), None, "individual reward"),
    "draft_section6.md|31 studies|0": (P(rw=("mix", "sys")), None, "aggregate reward"),
    "draft_section6.md|2 studies|0": (P(rw="unc"), None, "no stated reward value"),
    "draft_section6.md|1 of the 93|0": (AND(FULL, P(bd="yes", cm="yes", cd="yes")), FULL, "all three records"),
    "draft_section6.md|93 studies|0": (FULL, None, "the full texts"),
    "draft_section6.md|17 of the 93|0": (FORECAST, ALL, "forecast-conditioned state"),
    "draft_section6.md|14 of the 91|0": (P(gn="yes"), GEN_ASSESS, "transfer stated"),
    "draft_section6.md|93 studies|1": (ALL, None, "the corpus"),
    "draft_section6.md|93 studies|2": (ALL, None, "the corpus"),
    "draft_section6.md|91 studies|0": (GEN_ASSESS, None, "assessable on transfer"),
    # ---- Section VII
    "draft_section7_8.md|10 of the 93|0": (P(bd="yes"), ALL, "boundary addressed"),
    "draft_section7_8.md|45 studies|0": (GRP, None, "the group"),
    "draft_section7_8.md|93 studies|0": (ALL, None, "the corpus"),
    # ---- Supplementary
    "supplementary.md|8 of the 45|0": (AND(GRP, P(cd="yes")), GRP, "released code inside the group"),
    "supplementary.md|45 studies|0": (GRP, None, "the group"),
    "supplementary.md|93 studies|0": (FULL, None, "the full texts"),
    "supplementary.md|10 of 93|0": (AND(FULL, P(cd="yes")), FULL, "released code"),
    "supplementary.md|93 studies|1": (ALL, None, "the corpus"),
    "supplementary.md|10 of 93|1": (P(bd="yes"), ALL, "boundary addressed"),
    "supplementary.md|16 of 93|0": (AND(FULL, P(cm="yes")), FULL, "a trip completion rate"),
    "supplementary.md|10 of 93|2": (AND(FULL, P(cd="yes")), FULL, "released code"),
    "supplementary.md|14 of 91|0": (P(gn="yes"), GEN_ASSESS, "transfer stated"),
    "supplementary.md|45 of 93|0": (GRP, ALL, "the group"),
    "supplementary.md|14 of the 91|0": (P(gn="yes"), GEN_ASSESS, "transfer stated"),
    "supplementary.md|91 studies|0": (GEN_ASSESS, None, "assessable on transfer"),
    "supplementary.md|86 studies|0": (THREE, None, "assessable on the three fields"),
    "supplementary.md|86 studies|1": (THREE, None, "assessable on the three fields"),
    "supplementary.md|45 studies|1": (GRP, None, "the group"),
    "supplementary.md|45 studies|2": (GRP, None, "the group"),
    "supplementary.md|45 studies|3": (GRP, None, "the group"),
    "supplementary.md|10 studies|0": (P(bd="yes"), None, "boundary addressed"),
    "supplementary.md|93 studies|2": (ALL, None, "the corpus"),
    # ---- Appendix B
    "11_appendix_search_protocol.md|93 studies|0": (ALL, None, "the corpus"),
}


# ---------------------------------------------------------------- the archived simulator audit
# Section IV-F reads the evaluation environment of every study from the audit deposited with the
# archive. Table A-1 carries no such column, so these figures are asserted against that record.
NAMED, CUSTOM, NONE_, UNCLEAR_ = ("named-platform", "custom-described", "not-addressed", "unclear")
ADDRESSED = "boundary-open-addressed"

AUDIT_CLAIMS = {
    "draft_section4.md|53 studies|0": (lambda b, c: c == NAMED, "a named platform"),
    # T-1937: IV-G now prints the same count as "53 of the 94", from the simulator record.
    "draft_section4.md|53 of the 93|0": (lambda b, c: c == NAMED, "a named platform"),
    # T-1999: idx 13 and 40 were re-read at full text; both are described custom environments and
    # no unclear record remains, so the sentence prints 53 / 21 / 19 and no unclear clause.
    "draft_section4.md|21 studies|1": (lambda b, c: c == CUSTOM, "a described custom environment"),
    "draft_section4.md|19 studies|0": (lambda b, c: c == NONE_, "neither"),
    "draft_section4.md|8 studies|2": (lambda b, c: b == ADDRESSED and c == NAMED,
                                      "addressed boundary at a named platform"),
    "draft_section4.md|2 studies|4": (lambda b, c: b == ADDRESSED and c == CUSTOM,
                                      "addressed boundary at a described environment"),
    "draft_section4.md|19 studies|1": (lambda b, c: c == NONE_, "neither"),
}


# ---------------------------------------------------------------- figures no tally can produce
# Table A-1 records twelve columns and nothing else: no year, no venue, no publisher, no simulator,
# no destination-in-state, no optimality claim, no substrate and no geometry reading. A figure
# resting on any of those is listed here with the reason, and is not asserted.
YEAR = "Table A-1 records no publication year"
VENUE = "Table A-1 records no venue or publisher"
SCREEN = "a screening-stage count of the record pool, outside the corpus"
POOL = "a count over an enlarged or re-screened pool, not over the corpus"
S18 = "recorded in Supplementary Table S-18, which is not a column of Table A-1"
S20 = "a reward-detail category of Supplementary Table S-20, asserted against that table instead"
S22 = "read from the full texts for Supplementary Table S-22, not tabulated per study"
GEOM = "a formulation or code reading; the geometry is recorded in no column"
PROSE = "a reading of the source sentences, not a value of any column"
PASS41 = "the studies with a reward detail recorded in Table S-18, which no column records"
EXPT = "a value of the controlled experiment, not of the corpus"

NOT_DERIVABLE = {
    "draft_abstract_intro.md|53 studies|0": S18,
    "draft_abstract_intro.md|47 studies|0": S18,
    "draft_section2.md|4 studies|0": YEAR,
    "draft_section2.md|89 studies|0": YEAR,
    "draft_section2.md|4 studies|1": YEAR,
    "draft_section4.md|3 of 44|0": YEAR,
    "draft_section4.md|7 of 49|0": YEAR,
    "draft_section4.md|4 of 44|0": YEAR,
    "draft_section4.md|12 of 49|0": YEAR,
    "draft_section4.md|7 of 44|0": YEAR,
    "draft_section4.md|3 of 49|0": YEAR,
    "draft_section4.md|4 studies|0": PROSE,
    "draft_section4.md|7 studies|2": GEOM,
    "draft_section4.md|7 studies|3": GEOM,
    "draft_section4.md|3 studies|1": GEOM,
    "draft_section4.md|38 studies|0": GEOM,
    "draft_section4.md|38 studies|1": GEOM,
    "draft_section4.md|39 of the 45|0": S20,
    "draft_section4.md|6 studies|0": S20,
    "draft_section4.md|15 studies|2": S20,
    "draft_section4.md|24 studies|0": S20,
    "draft_section4.md|6 studies|1": GEOM,
    "draft_section4.md|3 studies|5": GEOM,
    "draft_section4.md|2 studies|2": GEOM,
    "draft_section4.md|6 studies|2": GEOM,
    "draft_section4.md|45 studies|7": S18,
    "draft_section4.md|4 studies|3": PROSE,
    "draft_section4.md|2 studies|3": PROSE,
    "draft_section4.md|17 of 29|0": S18,
    "draft_section7_8.md|7 of the 45|0": GEOM,
    "draft_section4.md|79 of the 93|0": S22,
    "draft_section4.md|79 of the 93|1": S22,
    # T-1937: the Table S-22 rows now printed in IV-G. Each is a reading of the full texts.
    # T-1994: the scan was re-executed over the 93 full texts and the denominator is the corpus.
    # The tabular-family claim of IV-B prints the same form and holds occurrence 0.
    "draft_section4.md|12 of the 93|1": S22,
    "draft_section4.md|5 of the 12|0": S22,
    "draft_section4.md|87 of the 93|0": S22,
    "draft_section4.md|63 of the 93|0": S22,
    "draft_section4.md|40 of the 93|0": S22,
    "draft_section4.md|73 of the 93|0": S18,
    "draft_section4.md|16 studies|4": S18,
    "supplementary.md|79 of the 552|0": EXPT,
    "supplementary.md|16 of the 25|0": EXPT,
    "supplementary.md|253 of the 416|0": EXPT,
    "supplementary.md|28 of the 30|0": EXPT,
    "supplementary.md|19 of the 95|0": SCREEN,
    "supplementary.md|19 of the 95|1": SCREEN,
    "supplementary.md|12 of the 95|0": SCREEN,
    "supplementary.md|10 of the 95|0": SCREEN,
    "supplementary.md|95 studies|0": SCREEN,
    "supplementary.md|95 studies|1": SCREEN,
    "supplementary.md|101 studies|0": SCREEN,
    # T-1996: reached by the widened noun list (records, candidates, exclusions, queries, the venue
    # nouns). Every one is a count of the search or screening record, or of a venue class no
    # column of Table A-1 holds; each is named here so the gate cannot pass it in silence.
    "supplementary.md|46 journal articles|0": VENUE,
    "supplementary.md|46 journal articles|1": VENUE,
    "supplementary.md|44 conference papers|0": VENUE,
    "supplementary.md|44 conference papers|1": VENUE,
    "supplementary.md|3 preprints|0": VENUE,
    "supplementary.md|3 preprints|1": VENUE,
    "supplementary.md|7 conference papers|0": VENUE,
    "supplementary.md|8 journal articles|0": VENUE,
    "supplementary.md|3 conference papers|0": VENUE,
    "supplementary.md|7 journal articles|0": VENUE,
    "supplementary.md|8 candidates|0": SCREEN,
    "supplementary.md|518 records|0": SCREEN,
    "supplementary.md|518 records|1": SCREEN,
    "supplementary.md|518 records|2": SCREEN,
    "supplementary.md|505 records|0": SCREEN,
    "supplementary.md|505 records|1": SCREEN,
    "supplementary.md|505 records|2": SCREEN,
    "supplementary.md|505 records|3": SCREEN,
    "supplementary.md|307 records|0": SCREEN,
    "supplementary.md|307 records|1": SCREEN,
    "supplementary.md|7 records|0": SCREEN,
    "supplementary.md|7 records|1": SCREEN,
    "supplementary.md|7 records|2": SCREEN,
    "supplementary.md|7 records|3": SCREEN,
    "supplementary.md|7 records|4": SCREEN,
    "supplementary.md|1084 records|0": SCREEN,
    "supplementary.md|1084 records|1": SCREEN,
    "supplementary.md|55 records|0": SCREEN,
    "supplementary.md|86 candidates|0": SCREEN,
    "supplementary.md|1155 candidates|0": SCREEN,
    "supplementary.md|103 records|0": SCREEN,
    "supplementary.md|103 records|1": SCREEN,
    "supplementary.md|102 records|0": SCREEN,
    "supplementary.md|6 records|0": SCREEN,
    "supplementary.md|3 records|0": SCREEN,
    "supplementary.md|1052 exclusions|0": SCREEN,
    "supplementary.md|33 records|0": SCREEN,
    "supplementary.md|33 records|1": SCREEN,
    "supplementary.md|26 records|0": SCREEN,
    "draft_section2.md|86 records|0": SCREEN,
    "11_appendix_search_protocol.md|514 queries|0": SCREEN,
    # T-2003: the one Scopus pass of the collection, 39 records of which 8 came from Scopus alone.
    "11_appendix_search_protocol.md|39 records|0": SCREEN,
    "11_appendix_search_protocol.md|8 records|0": SCREEN,
    "11_appendix_search_protocol.md|405 records|0": SCREEN,
    "11_appendix_search_protocol.md|1084 records|0": SCREEN,
    "supplementary.md|9 of 46|0": VENUE,
    "supplementary.md|1 of 44|0": VENUE,
    "supplementary.md|5 of 40|0": VENUE,
    "supplementary.md|11 of 53|0": VENUE,
    "supplementary.md|40 studies|0": VENUE,
    "supplementary.md|40 studies|1": VENUE,
    "supplementary.md|53 studies|0": VENUE,
    "supplementary.md|8 of these 40|0": VENUE,
    "supplementary.md|9 of 48|0": VENUE,
    "supplementary.md|3 of 40|0": VENUE,
    "supplementary.md|7 of 53|0": VENUE,
    "supplementary.md|4 of 40|0": VENUE,
    "supplementary.md|6 of 53|0": VENUE,
    "supplementary.md|30 of the 93|0": SCREEN,
    "supplementary.md|30 studies|0": SCREEN,
    "supplementary.md|1069 of the 1084|0": SCREEN,
    "supplementary.md|89 studies|0": YEAR,
    "supplementary.md|1052 of the 1155|0": SCREEN,
    "supplementary.md|102 of the 103|0": SCREEN,
    "supplementary.md|966 of the 1052|0": SCREEN,
    "supplementary.md|13 of 108|0": POOL,
    # T-1938: the three sensitivity checks brought into II-B. Each is a count over the borderline
    # set or the re-screened pool, so no tally of Table A-1 produces it.
    "draft_section2.md|101 studies|0": POOL,
    "draft_section2.md|12 of 95|0": POOL,
    "draft_section2.md|95 studies|0": POOL,
    "supplementary.md|17 of 108|0": POOL,
    "supplementary.md|10 of 108|0": POOL,
    "supplementary.md|15 of 106|0": POOL,
    "supplementary.md|53 of 108|0": POOL,
    "supplementary.md|42 studies|0": PASS41,
    "11_appendix_search_protocol.md|30 of the 93|0": SCREEN,
    "11_appendix_search_protocol.md|4 studies|0": YEAR,
}

# ---------------------------------------------------------------- printed tables
# Table II of the body, printed by the builder, one row per recorded field.
TABLE_II = {
    "State form": [("forecast", FORECAST), ("instantaneous", P(st="inst")), ("unclear", P(st="unc"))],
    # 2026-09-10 (T-1993): the boundary row prints two values. No reviewed study is recorded from
    # an abstract since T-1991 and the unclear count the row carried is 0, so the value left the
    # printed row. The predicate is not deleted: an unclear boundary returns the moment a report
    # states something the rule cannot resolve, and the row prints the count again.
    "Boundary condition": [("addressed", P(bd="yes")), ("not addressed", P(bd="no"))],
    "Reward alignment": [("individual", P(rw="ind")), ("mixed", P(rw="mix")), ("system", P(rw="sys")),
                         ("unclear", P(rw="unc"))],
    "Action feasibility": [("enforced", P(fs="enf")), ("masked", P(fs="mask")),
                           ("not enforced", P(fs="no")), ("unclear", P(fs="unc"))],
}
# Rows carrying one count and its denominator
TABLE_II_SINGLE = {
    "Trip completion rate reported": (AND(FULL, P(cm="yes")), FULL),
    "Released code": (AND(FULL, P(cd="yes")), FULL),
    "Generalization mechanism": (P(gn="yes"), GEN_ASSESS),
}
# The remaining rows of Table II are read from the full texts for Supplementary Tables S-18 and
# S-22: the evaluation environment, travel time as a reported measure, the benchmark network and
# the shortest-path comparison. No column of Table A-1 produces them.

WORD = {"link-level": ("act", "link"), "candidate-path": ("act", "path"), "other": ("act", "oth"),
        "instantaneous": ("st", "inst"), "short-range forecast": ("st", "local"),
        "rollout": ("st", "roll"), "individual": ("rw", "ind"), "mixed": ("rw", "mix"),
        "system-level": ("rw", "sys")}
FAMILY = {"Multi-agent": "ma", "Value-based deep": "dq", "Tabular": "tb", "Policy-gradient": "pg",
          "Model-based hybrid": "mb", "Unspecified": "un", "Distributional": "di"}


# ---------------------------------------------------------------- Table S-22 group closures
# The evaluation-practice table left the body for the Supplementary at the item-6 decision. Its rows are
# read from the full texts and no per-study record backs them, so a group that must sum to a fixed total
# is the only arithmetic a reader can check. The substrate group silently stopped closing when the
# denominator moved from 90 to 91; these checks exist so that cannot recur.
def substrate_artefact():
    """The Table S-22 rows their own run produces, or None where the run output is absent.

    handoff/experiments is a symlink into archive/, so the path is given directly rather than
    walked: pathlib.rglob and os.walk do not descend into a symlinked directory on Python 3.8.
    """
    import json, statistics
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "handoff", "experiments", "eval_substrate", "substrate.json")
    if not os.path.exists(path):
        return None
    d = json.load(open(path, encoding="utf-8"))
    per = d["per_study"]
    both = sum(1 for v in per.values()
               if "synthetic grid or lattice" in v["network"]
               and ("real city or corridor" in v["network"]
                    or "benchmark road network" in v["network"]))
    n = d["n_texts"]
    real = d["network"].get("real city or corridor", 0)
    syn = d["network"].get("synthetic grid or lattice", 0)
    vals = sorted(x for v in per.values() for x in v["nodes_stated"])
    return {"n_texts": n, "network": d["network"], "metrics": d["metrics"],
            "baselines": d["baselines"], "both": both, "neither": n - (real + syn - both),
            "nodes_stated_by": sum(1 for v in per.values() if v["nodes_stated"]),
            "nodes_values": len(vals), "nodes_median": int(statistics.median(vals)),
            "nodes_min": vals[0], "nodes_max": vals[-1]}


def s22_rows():
    """Item -> (count, denominator) for every row of Table S-22."""
    text = open(SUPP, encoding="utf-8").read()
    a = text.find("**Table S-22.")
    if a < 0:
        return {}
    blk = text[a:text.find("\n###", a)]
    out = {}
    for line in blk.splitlines():
        m = re.match(r"\|\s*([^|]+?)\s*\|\s*([0-9]+)\s*\|\s*([0-9]+)\s*\|", line)
        if m:
            out[m.group(1).strip()] = (int(m.group(2)), int(m.group(3)))
    return out


# A table can lose a row without any count changing on the page: a deletion aimed at one table can
# match a row of another, which is how five rows of Table S-18 were removed while Table S-21 kept
# the rows the edit was meant to take. Every supplementary table with a fixed length is asserted.
FIXED_ROWS = {"S-18": 93, "S-19": 45, "S-20": 45, "S-21": 10}


def supp_table_rows(name):
    """Data rows of one supplementary table, counted between its caption and the next caption."""
    text = open(SUPP, encoding="utf-8").read()
    a = text.find("**Table %s." % name)
    if a < 0:
        return None
    nxt = re.search(r"\n\*\*Table S-\d+\.", text[a + 10:])
    blk = text[a:a + 10 + nxt.start()] if nxt else text[a:]
    return len([l for l in blk.splitlines()
                if re.match(r"\|\s*\[\d+\]", l) or re.match(r"\|[^|]*\|\s*\[\d+\]", l)])


def reward_categories():
    """Members of each Table S-20 category, which the body prints as 6 / 15 / 23."""
    text = open(SUPP, encoding="utf-8").read()
    a = text.find("**Table S-20.")
    if a < 0:
        return {}
    nxt = re.search(r"\n\*\*Table S-\d+\.", text[a + 10:])
    blk = text[a:a + 10 + nxt.start()] if nxt else text[a:]
    cur, out = None, {}
    for line in blk.splitlines():
        m = re.match(r"\|\s*([^|]*?)\s*\|\s*\[(\d+)\]", line)
        if not m:
            continue
        d = re.match(r"([123])\.", m.group(1))
        if d:
            cur = int(d.group(1))
        if cur:
            out[cur] = out.get(cur, 0) + 1
    return out


# 2026-09-10 (T-1992): these pointed at runs/zenodo_v1.5.0/, a frozen snapshot two deposits old,
# so the audit join read a record the manuscript had stopped using. They now read the staged
# package, which is the record the next deposit is built from and the one T-1991 recomposed.
AUDIT = "handoff/deposit_next/v1.7.0/study_record_simulator_audit_v6.csv"
RECORD = "handoff/deposit_next/v1.7.0/study_record_corpus_v9_coded.csv"


def boundary_split():
    """(treatment stated, periphery named) among the addressed boundaries, from the study record.

    2026-09-10 (T-2001). The boundary field counts a report addressing the condition, and the ten
    it counts divide by the evidence: seven fix what happens to a vehicle reaching a peripheral
    link, three state a property of the periphery itself (III-A). Table A-1 holds one value for
    both, so the split is recorded in boundary_status_note of the record and read from there.
    """
    import csv
    t = p = 0
    for r in csv.DictReader(open(RECORD, encoding="utf-8-sig")):
        if r["in_reviewed_corpus"].strip().lower() != "yes" or r["boundary_condition"].strip() != "boundary-open-addressed":
            continue
        note = (r.get("boundary_status_note") or "").strip().lower()
        if note.startswith("periphery named"):
            p += 1
        elif note.startswith("treatment stated"):
            t += 1
    return t, p


def audit_rows():
    """The reviewed studies with the evaluation-environment category of the archived audit.

    Section IV-F reads the environment from this record, which no column of Table A-1 carries.
    """
    import csv
    rec = [r for r in csv.DictReader(open(RECORD, encoding="utf-8-sig"))
           if r["in_reviewed_corpus"].strip().lower() == "yes"]
    cat = {r["idx"]: r["category"].strip()
           for r in csv.DictReader(open(AUDIT, encoding="utf-8-sig"))}
    return [(r["boundary_condition"].strip(), cat.get(r["idx"], "?")) for r in rec]


def a1_rows():
    text = open(TABLE, encoding="utf-8").read()
    return [Row(*m) for m in ROW_RX.findall(text)]


def tally(rows, field):
    d = {}
    for r in rows:
        v = getattr(r, field)
        d[v] = d.get(v, 0) + 1
    return d


# ---------------------------------------------------------------- the printed figures
P1 = re.compile(r"\b(\d+) of (?:the |these )?(\d+)\b")
# 2026-09-10 (T-1996). The second pattern matched three nouns, and the T-1995 audit found eleven
# stale corpus figures printed beside nouns it did not match ("40 studies and 54 studies" was
# caught only as "54 studies"; "the two strata", "8 candidates", "1052 exclusions" were not).
# CORPUS_NOUNS names every unit the corpus or its assembly is counted in. A number in front of one
# of them is a corpus figure and must be classified. Experiment units (seeds, episodes, pairs,
# nodes, links) are not corpus units and stay outside on purpose.
CORPUS_NOUNS = ["studies", "reports", "full texts", "records", "candidates", "exclusions",
                "combinations", "statements", "queries", "preprints", "conference papers",
                "journal articles"]
P2 = re.compile(r"\b(\d[\d,]*) (?:%s)\b" % "|".join(re.escape(n) for n in CORPUS_NOUNS))


def figures():
    """Every corpus figure printed in the manuscript and the Supplementary, in source order.

    A figure is keyed by its file, its printed form and its occurrence index in that file, so a
    changed count changes the key and is reported as unclassified rather than passing silently.
    """
    out = []
    for f in SRC:
        text = open("handoff/manuscript/source_md/" + f, encoding="utf-8").read()
        seen = {}
        for line in text.split("\n"):
            if line.startswith(("|", ">", "#")):
                continue
            for rx in (P1, P2):
                for m in rx.finditer(line):
                    if line[:m.start()].endswith(("S-", "Table ", "Fig. ", "Eq. ", "A-")):
                        continue      # a table or figure number ("Table A-1 records"), not a count
                    k = m.group(0)
                    seen[k] = seen.get(k, 0) + 1
                    out.append((f, k, seen[k] - 1, m))
    return out


def s14_rows():
    """Table S-14: one row per design combination, with its study count and boundary coverage."""
    text = open(SUPP, encoding="utf-8").read()
    a = text.find("**Table S-14.")
    if a < 0:
        return []
    nxt = re.search(r"\n\*\*Table S-\d+\.", text[a + 10:])
    blk = text[a:a + 10 + nxt.start()] if nxt else text[a:]
    out = []
    for line in blk.splitlines():
        m = re.match(r"\|\s*([^|]+?)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*(\d+)\s*\|"
                     r"\s*(\d+) of (\d+)\s*\|", line)
        if m:
            out.append((m.group(1), m.group(2), m.group(3), int(m.group(4)),
                        int(m.group(5)), int(m.group(6))))
    return out


def s15_rows():
    """Table S-15: one row per family, with the study count, the object parenthetical and the
    boundary column."""
    text = open(SUPP, encoding="utf-8").read()
    a = text.find("**Table S-15.")
    if a < 0:
        return []
    nxt = re.search(r"\n\*\*Table S-\d+\.", text[a + 10:])
    blk = text[a:a + 10 + nxt.start()] if nxt else text[a:]
    out = []
    for line in blk.splitlines():
        m = re.match(r"\|\s*([A-Z][^|]*?)\s*\|\s*(\d+)\s*\|\s*([^|]*?)\s*\|\s*[^|]*?\s*\|\s*(\d+)\s*\|", line)
        if m:
            out.append((m.group(1), int(m.group(2)), m.group(3), int(m.group(4))))
    return out


def table_ii_rows():
    """Table II of the body, read from the builder that prints it."""
    text = open(BUILDER, encoding="utf-8").read()
    a = text.find("@@TABLE_REPORTING@@")
    blk = text[a:text.find("\n}", a)]
    return re.findall(r'\["([^"]+)", "([^"]+)", "(\d+)"', blk.replace("\n", " "))


def main():
    rows = a1_rows()
    fails = 0
    asserts = 0
    explain = "--figures" in sys.argv

    def check(name, got, exp):
        nonlocal fails, asserts
        ok = got == exp
        fails += 0 if ok else 1
        asserts += 1
        print(f"{'OK  ' if ok else 'FAIL'} {name}: got {got} expected {exp}")

    def compare(name, got, exp):
        """One assertion of the printed-figure pass, reported only where it fails."""
        nonlocal fails, asserts
        asserts += 1
        if got != exp:
            fails += 1
            print(f"FAIL {name}: got {got} expected {exp}")

    check("rows", len(rows), EXPECTED["rows"])
    for k, v in EXPECTED["fam"].items():
        check(f"family {k}", tally(rows, "fam").get(k, 0), v)
    for k, v in EXPECTED["act"].items():
        check(f"action {k}", tally(rows, "act").get(k, 0), v)
    for k, v in EXPECTED["state"].items():
        check(f"state {k}", tally(rows, "st").get(k, 0), v)
    check("state forecast (local+roll)", sum(1 for r in rows if FORECAST(r)), 17)
    for k, v in EXPECTED["reward"].items():
        check(f"reward {k}", tally(rows, "rw").get(k, 0), v)
    for k, v in EXPECTED["boundary"].items():
        check(f"boundary {k}", tally(rows, "bd").get(k, 0), v)
    check("completion yes", tally(rows, "cm").get("yes", 0), EXPECTED["compl_yes"])
    check("code yes", tally(rows, "cd").get("yes", 0), EXPECTED["code_yes"])
    check("generalization yes", tally(rows, "gn").get("yes", 0), EXPECTED["gen_yes"])
    for k, v in EXPECTED["enc"].items():
        check(f"encoder {k}", tally(rows, "enc").get(k, 0), v)
    for k, v in EXPECTED["feas"].items():
        check(f"feasibility {k}", tally(rows, "fs").get(k, 0), v)
    check("link+ind+boundary-no (the 44)", sum(1 for r in rows if GRP(r)), EXPECTED["link_ind_bno"])
    check("link+inst+ind (the 39)", sum(1 for r in rows if P(act="link", st="inst", rw="ind")(r)),
          EXPECTED["link_inst_ind"])
    check("full texts (the 93)", sum(1 for r in rows if FULL(r)), 93)
    check("assessable on action, state and reward (the 86)", sum(1 for r in rows if THREE(r)), 86)
    check("assessable on transfer (the 91)", sum(1 for r in rows if GEN_ASSESS(r)), 91)

    # --- Table S-22 closures -------------------------------------------------
    t = s22_rows()
    if not t:
        check("Table S-22 present", 0, 1)
    else:
        def g(k):
            return t.get(k, (None, None))[0]
        den = t.get("Real city or corridor substrate", (0, 0))[1]
        if None not in (g("Real city or corridor substrate"), g("Synthetic grid or lattice"),
                        g("Both substrates"), g("Neither substrate")):
            closure = (g("Real city or corridor substrate") + g("Synthetic grid or lattice")
                       - g("Both substrates") + g("Neither substrate"))
            check("S-22 substrate closure (real+synthetic-both+neither)", closure, den)
        if None not in (g("Sioux Falls"), g("Braess"), g("Anaheim"), g("Nguyen-Dupuis"),
                        g("Benchmark road network"), g("More than one benchmark network")):
            named = g("Sioux Falls") + g("Braess") + g("Anaheim") + g("Nguyen-Dupuis")
            check("S-22 benchmark closure (named = total + more than one)",
                  named, g("Benchmark road network") + g("More than one benchmark network"))
        if None not in (g("SUMO"), g("Another engine"), g("Evaluation environment named")):
            check("S-22 engine closure (SUMO + another = named)",
                  g("SUMO") + g("Another engine"), g("Evaluation environment named"))
        STATS = ("Median node count", "Smallest node count", "Largest node count")
        for item, (cnt, d) in sorted(t.items()):
            if item.startswith(STATS):
                continue
            if cnt > d:
                check("S-22 count within its denominator: %s" % item, cnt, d)

    # --- Table S-22 against the run that produces it (T-1945) -----------------
    # Every row above is print-versus-print. These rows have one artefact behind them,
    # eval_substrate/substrate.json, and T-1907 found the printed denominator had drifted from it
    # without any gate seeing the drift. The comparison is made here so the same drift cannot
    # recur. One row carries a judgement rather than the lexical count and is named with its
    # reason: the 'random or greedy' pattern matches an epsilon-greedy exploration rule as well as
    # a greedy comparison, and the manuscript prints the count with that one text withheld.
    sub = substrate_artefact()
    if sub is None:
        print("  -  %-58s %s" % ("Table S-22 against substrate.json",
                                 "the artefact is not on disk; rows are print-versus-print alone"))
    elif t:
        check("S-22 texts read by the scan", sub["n_texts"], den)
        ART = {"Real city or corridor substrate": sub["network"].get("real city or corridor"),
               "Synthetic grid or lattice": sub["network"].get("synthetic grid or lattice"),
               "Benchmark road network": sub["network"].get("benchmark road network"),
               "Both substrates": sub["both"],
               "Neither substrate": sub["neither"],
               "Travel time reported": sub["metrics"].get("travel time"),
               "Delay": sub["metrics"].get("delay"),
               "Fuel or emission term": sub["metrics"].get("fuel or emission"),
               "Reward or return": sub["metrics"].get("reward or return"),
               "Distance": sub["metrics"].get("distance"),
               "Throughput or queue term": sub["metrics"].get("throughput or queue"),
               "Trip completion named as a measure": sub["metrics"].get("completion or arrival rate"),
               "Shortest-path baseline": sub["baselines"].get("shortest path or Dijkstra"),
               "User equilibrium": sub["baselines"].get("user equilibrium"),
               "Static or fixed routing rule": sub["baselines"].get("static or fixed routing"),
               "Another learned method": sub["baselines"].get("another RL method"),
               "Node or intersection count stated": sub["nodes_stated_by"],
               "Median node count across the values stated": sub["nodes_median"],
               "Smallest node count across the values stated": sub["nodes_min"],
               "Largest node count across the values stated": sub["nodes_max"]}
        for item, want in sorted(ART.items()):
            if want is None or g(item) is None:
                continue
            check("S-22 row against the run: %s" % item, g(item), want)
        # the one judged row, checked against the lexical count less the withheld text
        lex = sub["baselines"].get("random or greedy")
        if lex is not None and g("Random or greedy rule") is not None:
            check("S-22 judged row: random or greedy is the lexical count less one",
                  g("Random or greedy rule"), lex - 1)
        # the three node statistics are over the values, not over the texts stating them
        for item in ("Median node count across the values stated",
                     "Smallest node count across the values stated",
                     "Largest node count across the values stated"):
            if item in t:
                check("S-22 node statistic denominated by the values: %s" % item.split(" node")[0],
                      t[item][1], sub["nodes_values"])

    for name, n in sorted(FIXED_ROWS.items()):
        got = supp_table_rows(name)
        check("Table %s row count" % name, -1 if got is None else got, n)
    cats = reward_categories()
    check("S-20 category 1", cats.get(1, 0), 6)
    check("S-20 category 2", cats.get(2, 0), 15)
    check("S-20 category 3", cats.get(3, 0), 24)
    check("S-20 categories sum", sum(cats.values()), 45)

    # --- Section IV-F, the figures the extractor cannot see -------------------
    # "in none of the 6", "5 of those 6", "13 of the same 16" and the absence structure are
    # printed in forms no "N of M" pattern matches, so each is asserted here by name.
    check("IV-F guidance-parameter with an individual reward",
          sum(1 for r in rows if P(act="oth", rw="ind")(r)), 0)
    check("IV-F guidance-parameter with a system-level reward",
          sum(1 for r in rows if P(act="oth", rw="sys")(r)), 5)
    check("IV-F guidance-parameter with a mixed reward",
          sum(1 for r in rows if P(act="oth", rw="mix")(r)), 1)
    check("IV-F graph encoders at a link-level action",
          sum(1 for r in rows if P(enc="graph", act="link")(r)), 13)
    check("IV-F another learned encoder with transfer",
          sum(1 for r in rows if P(enc="other", gn="yes")(r)), 0)
    check("IV-F tabular studies with a graph encoder",
          sum(1 for r in rows if P(fam="tb", enc="graph")(r)), 0)
    check("IV-F unspecified family at a link-level action",
          sum(1 for r in rows if P(fam="un", act="link")(r)), 0)
    tri = triples(rows)
    absent = absent_triples(rows)
    check("IV-F combinations absent", len(absent), 11)
    check("IV-F absences at a guidance-parameter action with an individual reward",
          len([k for k in absent if k[0] == "oth" and k[2] == "ind"]), 3)
    check("IV-F absences at a rollout state with a mixed reward",
          len([k for k in absent if k[1] == "roll" and k[2] == "mix"]), 3)
    check("IV-F absences at a forecast-conditioned state with a system-level reward",
          len([k for k in absent if k[1] in ("local", "roll") and k[2] == "sys"]), 4)
    check("IV-F largest combination", max(tri.values()), 40)
    check("IV-F combinations holding two studies or fewer",
          len([k for k, v in tri.items() if v <= 2]), 10)
    aud = audit_rows()
    check("IV-F studies naming no environment that address the boundary",
          sum(1 for b, c in aud if c == NONE_ and b == ADDRESSED), 0)
    check("IV-F reviewed studies in the audit", len(aud), 93)
    ts, pn = boundary_split()
    check("boundary field: treatment stated", ts, 7)
    check("boundary field: periphery named", pn, 3)
    check("boundary field: treatment + periphery = addressed", ts + pn, sum(1 for r in rows if r.bd == "yes"))

    # --- Table III, every cell a tally of two Appendix A columns ---------------
    design = [r for r in table_ii_rows()]  # the builder's row literals, all tables
    ACTS = {"Link-level": "link", "Candidate-path": "path", "Guidance-parameter": "oth",
            "Unclear": "unc"}
    REWARDS = ["ind", "mix", "sys", "unc"]
    blk = open(BUILDER, encoding="utf-8").read()
    a = blk.find('"@@TABLE_DESIGN@@": (')
    rows_txt = re.findall(r'\["([^"]+)", "(\d+)", "(\d+)", "(\d+)", "(\d+)", "(\d+)", "(\d+)"\]',
                          blk[a:a + 1600])
    check("Table III rows", len(rows_txt), 5)
    for cells in rows_txt:
        name, vals = cells[0], [int(x) for x in cells[1:]]
        sel = (lambda r: True) if name == "All" else P(act=ACTS[name])
        for rw, printed in zip(REWARDS, vals[:4]):
            check("Table III %s / %s" % (name, rw),
                  sum(1 for r in rows if sel(r) and r.rw == rw), printed)
        check("Table III %s, studies" % name, sum(1 for r in rows if sel(r)), vals[4])
        check("Table III %s, boundary stated" % name,
              sum(1 for r in rows if sel(r) and r.bd == "yes"), vals[5])

    # --- every printed figure ------------------------------------------------
    per_file = {}
    for f, k, i, m in figures():
        key = "%s|%s|%d" % (f, k, i)
        a, b, d = per_file.setdefault(f, [0, 0, 0])
        per_file[f][0] += 1
        if key in CLAIMS:
            per_file[f][1] += 1
            num, den, label = CLAIMS[key]
            got = num.fn(rows) if isinstance(num, ROWS) else sum(1 for r in rows if num(r))
            printed = int(m.group(1))
            compare("%s: %s (%s)" % (f, k, label), got, printed)
            if den is not None:
                gotd = den.fn(rows) if isinstance(den, ROWS) else sum(1 for r in rows if den(r))
                compare("%s: %s, denominator (%s)" % (f, k, label), gotd, int(m.group(2)))
            if explain:
                print("     %-58s %s" % (key, label))
        elif key in AUDIT_CLAIMS:
            per_file[f][1] += 1
            pred, label = AUDIT_CLAIMS[key]
            compare("%s: %s (%s)" % (f, k, label),
                    sum(1 for b, c in audit_rows() if pred(b, c)), int(m.group(1)))
            if explain:
                print("     %-58s %s (simulator audit)" % (key, label))
        elif key in NOT_DERIVABLE:
            per_file[f][2] += 1
            if explain:
                print("  -  %-58s %s" % (key, NOT_DERIVABLE[key]))
        else:
            check("%s: %s is classified" % (f, k), "unclassified", "a claim or a stated reason")
    total = [sum(v[i] for v in per_file.values()) for i in range(3)]
    for f in SRC:
        if f in per_file:
            n, a, d = per_file[f]
            print("OK   %-32s %3d printed figures: %3d recomputed, %2d not derivable" % (f, n, a, d))
    print("OK   printed figures, all sources    %3d printed figures: %3d recomputed, %2d not derivable"
          % tuple(total))

    # --- printed tables ------------------------------------------------------
    for field, parts in TABLE_II.items():
        for label, value in [(r[0], r[1]) for r in table_ii_rows() if r[0] == field]:
            for word, pred in parts:
                m = re.search(r"%s (\d+)" % re.escape(word), value)
                if not m:
                    check("Table II %s: %s present" % (field, word), "absent", "a count")
                    continue
                check("Table II %s / %s" % (field, word), sum(1 for r in rows if pred(r)), int(m.group(1)))
    for name, (num, den) in TABLE_II_SINGLE.items():
        got = [r for r in table_ii_rows() if r[0] == name]
        if not got:
            check("Table II row present: %s" % name, 0, 1)
            continue
        check("Table II %s" % name, sum(1 for r in rows if num(r)), int(got[0][1]))
        check("Table II %s, denominator" % name, sum(1 for r in rows if den(r)), int(got[0][2]))

    s14 = s14_rows()
    check("Table S-14 rows", len(s14), 17)          # 16 combinations and the All row
    placed = 0
    for act, st, rw, n, bn, bd in s14:
        if act == "All":
            check("Table S-14 All row", sum(1 for r in rows if THREE(r)), n)
            check("Table S-14 All row, addressed", sum(1 for r in rows if THREE(r) and r.bd == "yes"), bn)
            continue
        pred = P(**dict([WORD[act], WORD[st], WORD[rw]]))
        placed += n
        check("Table S-14 %s / %s / %s" % (act, st, rw), sum(1 for r in rows if pred(r)), n)
        check("Table S-14 %s / %s / %s, addressed" % (act, st, rw),
              sum(1 for r in rows if pred(r) and r.bd == "yes"), bn)
        if bd != n:
            check("Table S-14 %s / %s / %s, coverage denominator" % (act, st, rw), bd, n)
    check("Table S-14 combinations placed", placed, sum(1 for r in rows if THREE(r)))
    check("Table S-14 combinations listed", len(s14) - 1,
          len(set((r.act, r.st, r.rw) for r in rows if THREE(r))))

    s15 = s15_rows()
    check("Table S-15 rows", len(s15), 7)
    for fam, n, obj, bn in s15:
        code = FAMILY.get(fam)
        if code is None:
            check("Table S-15 family named in Table A-1: %s" % fam, 0, 1)
            continue
        check("Table S-15 %s studies" % fam, sum(1 for r in rows if r.fam == code), n)
        check("Table S-15 %s boundary stated" % fam, sum(1 for r in rows if r.fam == code and r.bd == "yes"), bn)
        m = re.search(r"mixed or system in (\d+)", obj)
        if m:
            check("Table S-15 %s, mixed or system" % fam,
                  sum(1 for r in rows if r.fam == code and r.rw in ("mix", "sys")), int(m.group(1)))
        else:
            for word, val in (("individual", "ind"), ("mixed", "mix"), ("system", "sys")):
                m = re.search(r"%s (?:in )?(\d+)" % word, obj)
                if m:
                    check("Table S-15 %s, %s" % (fam, word),
                          sum(1 for r in rows if r.fam == code and r.rw == val), int(m.group(1)))
    check("Table S-15 families sum", sum(n for _, n, _, _ in s15), len(rows))
    check("Table S-15 boundary column sum", sum(b for _, _, _, b in s15),
          sum(1 for r in rows if r.bd == "yes"))

    print(f"countcheck: {asserts} assertions, {fails} failures")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
