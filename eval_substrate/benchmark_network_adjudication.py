# -*- coding: utf-8 -*-
"""Adjudicate the 'benchmark road network' substrate label, study by study.

WHY THIS FILE EXISTS
substrate.py assigns every substrate label by a lexical search over the full text and stores the
sentence that carried the FIRST match. For the benchmark-network label that rule counted 12
studies. Reading every occurrence rather than the first shows one of the 12, idx 38, carries its
only occurrence inside a related-work paragraph describing Wang et al.'s MARL-OD-DA, while its own
evaluation runs on a synthetic 5x6 grid and on a Toronto network built from OpenStreetMap. The
label is therefore withdrawn for that study and the count is 11.

Reading only the first match also reverses the verdict the other way. The first Braess occurrence in
idx 51 sits in a related-work sentence about the price of anarchy, so a first-match read discards a
study whose own Section evaluates on Sioux Falls ("B. Sioux Falls Network ... It has 24 vertices and
76 edges") and on the Ortuzar-Willumsen network. Every occurrence is enumerated here for that reason.

This file is the authority for the count and for the network names printed in Section IV-A.
Run: python3 benchmark_network_adjudication.py
"""
import csv, io, os, re, json, sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEXTS = os.path.join(HERE, '..', 'corpus_text')
# Two records can sit beside this directory. _CURRENT is the coding this version publishes, at the
# top level under its released name. _SNAPSHOT is corpus/corpus_v9_coded.csv, which is the coding as
# of v1.4.0 and is superseded on the boundary field; it exists in the repository layout and not in
# the deposited package. The current one wins wherever both are present, since a tally taken over
# the snapshot returns the withdrawn boundary counts. Set CORPUS_CSV to override either.
_CURRENT = os.path.join(HERE, '..', 'study_record_corpus_v9_coded.csv')
_SNAPSHOT = os.path.join(HERE, '..', 'corpus', 'corpus_v9_coded.csv')
CORPUS = os.environ.get('CORPUS_CSV') or (_CURRENT if os.path.exists(_CURRENT) else _SNAPSHOT)

PATTERN = (r'Sioux\s*Falls|Anaheim|Nguyen[- ]Dupuis|Winnipeg|Chicago Sketch|'
           r'Barcelona network|Friedrichshain|Braess')

# Canonical name -> the pattern that names it, for the per-study network list.
NAMED = [
    ('Sioux Falls',        r'Sioux\s*Falls'),
    ('Anaheim',            r'Anaheim'),
    ('Nguyen-Dupuis',      r'Nguyen[- ]Dupuis'),
    ('Winnipeg',           r'Winnipeg'),
    ('Chicago Sketch',     r'Chicago Sketch'),
    ('Barcelona',          r'Barcelona network'),
    ('Friedrichshain',     r'Friedrichshain'),
    ('Braess',             r'Braess'),
]

# Hand verdicts. A study is WITHDRAWN only where EVERY occurrence describes another study's work.
# The quote is the deciding one, copied from the text.
WITHDRAWN = {
    '38': ("its single occurrence is inside the related-work paragraph on OD-level assignment, "
           "describing Wang et al.'s MARL-OD-DA: \"This approach achieved superior convergence "
           "performance in networks such as SiouxFalls\". Its own substrates are a synthetic 5x6 "
           "grid of 30 intersections and a Toronto network derived from OpenStreetMap."),
}
# Studies whose FIRST occurrence is related work but which are KEPT on a later occurrence.
KEPT_ON_LATER = {
    '51': ("first occurrence is \"They show results for small networks such as the one used to "
           "illustrate the Braess paradox\"; the study's own evaluation section reads \"B. Sioux "
           "Falls Network The SF network ... has 24 vertices and 76 edges\"."),
}


def load_texts():
    rows = [r for r in csv.DictReader(io.open(CORPUS, encoding='utf-8'))
            if str(r.get('in_reviewed_corpus', '')).strip().lower() in ('1', 'true', 'yes', 'y')]
    out = {}
    for r in rows:
        idx = str(r['idx']).strip()
        p = os.path.join(TEXTS, idx + '.txt')
        if os.path.exists(p):
            out[idx] = (r['title'], io.open(p, encoding='utf-8', errors='replace').read())
    return out


def main():
    texts = load_texts()
    matched, adjudicated, per_network = [], {}, {}
    for idx, (title, t) in sorted(texts.items(), key=lambda kv: int(kv[0])):
        hits = list(re.finditer(PATTERN, t, re.I))
        if not hits:
            continue
        matched.append(idx)
        if idx in WITHDRAWN:
            continue
        names = sorted({n for n, p in NAMED if re.search(p, t, re.I)})
        adjudicated[idx] = {'title': title, 'occurrences': len(hits), 'networks': names}
        for n in names:
            per_network.setdefault(n, []).append(idx)

    print('lexical matches            : %d  %s' % (len(matched), ' '.join(matched)))
    print('withdrawn after reading all: %d  %s' % (len(WITHDRAWN), ' '.join(sorted(WITHDRAWN))))
    print('kept on a later occurrence : %d  %s' % (len(KEPT_ON_LATER), ' '.join(sorted(KEPT_ON_LATER))))
    print('ADJUDICATED COUNT          : %d' % len(adjudicated))
    print()
    for idx, v in sorted(adjudicated.items(), key=lambda kv: int(kv[0])):
        print('  [%2s] %-2d occ  %-46s %s' % (idx, v['occurrences'], ', '.join(v['networks']),
                                              v['title'][:44]))
    print()
    print('by network, most reused first:')
    for n, ids in sorted(per_network.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        print('  %-16s %2d  %s' % (n, len(ids), ' '.join(ids)))

    # --- negative control: the withdrawal must be visible, and the keep must be too -------------
    assert '38' in matched, 'control: idx 38 must still MATCH lexically, else the finding is moot'
    assert '38' not in adjudicated, 'control: idx 38 must be withdrawn'
    assert '51' in adjudicated, 'control: idx 51 must survive; a first-match read would drop it'
    ctrl = [i for i in adjudicated if re.search(r'Sioux\s*Falls', texts[i][1], re.I)]
    assert len(ctrl) == len(per_network.get('Sioux Falls', [])), 'control: Sioux Falls tally'
    print('\nnegative controls: 4/4 pass')

    json.dump({'adjudicated_count': len(adjudicated),
               'lexical_count': len(matched),
               'withdrawn': WITHDRAWN,
               'kept_on_later_occurrence': KEPT_ON_LATER,
               'per_study': adjudicated,
               'per_network': {k: sorted(v, key=int) for k, v in per_network.items()}},
              io.open(os.path.join(HERE, 'benchmark_network_adjudication.json'), 'w',
                      encoding='utf-8'), indent=1, sort_keys=True)
    print('wrote benchmark_network_adjudication.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
