# -*- coding: utf-8 -*-
"""What the 94 studies actually evaluate on: engine, network, size, metrics, baselines.

WHY THIS EXISTS. Two independent survey-track reviews named the same gap as the paper's largest:
Section IV describes the corpus by five DESIGN fields and by nothing about the EVALUATION. The
paper's own premise is that the reported travel times share no scale, and the direct evidence for
that premise is an inventory of what each study measured on. That inventory was missing.

The rule is lexical and its output is a distribution, not a per-study claim. Every detection
carries the matched sentence so a reader can disagree with any one of them against the same text.
Nothing is pooled and nothing is ranked, which is the constraint Section I sets on this review.

CONTROLS, both printed before any new column is read:
  1. The manuscript states, from manual reading, that 53 of the 94 name an evaluation environment
     and 47 of those name the same simulator. A lexical rule built independently should land near
     both. A wide miss means the rule is measuring something else and its other columns are void.
  2. idx 48's text file was quarantined as the wrong source until T-1945 recovered the 2019
     conference paper from pdfs/pdfs/ and regenerated the extraction; nothing is quarantined now
     and the scan reads all 91 full texts. corpus_text/48.README.txt records the recovery.

    python3 substrate.py
"""
import csv, io, json, os, re, sys
from collections import Counter

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
OUT = os.path.join(HERE, 'substrate.json')

QUARANTINED = set()           # emptied by T-1945; corpus_text/48.README.txt records why

ENGINES = [
    ('SUMO',        r'\bSUMO\b|Simulation of Urban Mobility'),
    ('VISSIM',      r'\bVISSIM\b|PTV Vissim'),
    ('Aimsun',      r'\bAimsun\b'),
    ('MATSim',      r'\bMATSim\b'),
    ('Paramics',    r'\bParamics\b'),
    ('CityFlow',    r'\bCityFlow\b'),
    ('Flow/RLlib',  r'\bFlow framework\b|\bRLlib\b'),
    ('SimMobility', r'\bSimMobility\b'),
    ('TransModeler', r'\bTransModeler\b|\bCORSIM\b|\bDynaMIT\b|\bDYNASMART\b'),
]
NETWORK = [
    ('benchmark road network',
     r'Sioux\s*Falls|Anaheim|Nguyen[- ]Dupuis|Winnipeg|Chicago Sketch|Barcelona network|'
     r'Friedrichshain|Braess'),
    ('real city or corridor',
     r'road network of [A-Z]|network of the city|city of [A-Z][a-z]+|Manhattan|Beijing|Shanghai|'
     r'Shenzhen|Chengdu|Hangzhou|Xi\'?an|Jinan|Singapore|Luxembourg|Cologne|Bologna|Monaco|'
     r'downtown|arterial corridor|real[- ]world (road )?network|OpenStreetMap|OSM'),
    ('synthetic grid or lattice',
     r'grid network|grid[- ]like|\d\s*[x×]\s*\d\s*grid|lattice|Manhattan[- ]style grid|'
     r'synthetic network|artificial network|toy network'),
]
# nodes and links are different objects and a median pooling them is not interpretable, which a
# reviewer identified in the first version of this script. They are counted separately.
NODES = re.compile(r'(\d[\d,]{0,6})\s+(?:nodes|intersections|junctions)', re.I)
LINKS = re.compile(r'(\d[\d,]{0,6})\s+(?:links|edges|road segments)', re.I)
METRICS = [
    ('travel time',      r'travel time|trip time|journey time|traveling time'),
    ('completion or arrival rate',
     r'success rate|arrival rate|completion rate|reach(?:ing)? (?:the |their )?destination rate|'
     r'\barrival ratio\b'),
    ('delay',            r'\bdelay\b|waiting time'),
    ('throughput or queue', r'throughput|queue length|number of vehicles (?:that )?(?:completed|arrived)'),
    ('reward or return', r'cumulative reward|average reward|episode reward|total reward|return curve'),
    ('distance',         r'path length|route length|travel distance|trip distance'),
    ('fuel or emission', r'fuel consumption|emission|CO2|energy consumption'),
]
BASELINES = [
    ('shortest path or Dijkstra', r'Dijkstra|shortest[- ]path (?:algorithm|baseline|method)|\bA\*\b'),
    ('static or fixed routing',  r'fixed(?:-| )route|static routing|pre[- ]?computed route|'
                                 r'shortest distance route|no[- ]guidance'),
    ('another RL method',        r'compared (?:with|to) (?:the )?(?:DQN|Q-learning|PPO|A3C|DDPG)|'
                                 r'baseline (?:DQN|Q-learning|PPO|DDPG)'),
    ('random or greedy',         r'random (?:policy|routing|baseline)|greedy (?:policy|baseline)'),
    ('user equilibrium',         r'user equilibrium|\bUE\b assignment|system optimum'),
]


# ---------------------------------------------------------------------------
# find() stores the FIRST match, so a label can rest on a sentence describing another study.
# Every occurrence of the benchmark-network pattern was read study by study in
# benchmark_network_adjudication.py. One study carries all of its occurrences inside a
# related-work paragraph and the label is withdrawn for it here, which is why this file reports
# 11 and not 12. The same reading KEPT idx 51, whose first occurrence is related work and whose
# own evaluation section runs on Sioux Falls; a first-match rule would have dropped it.
RELATED_WORK_ONLY = {
    '38': ('benchmark road network',),
}


def sentences(t):
    return re.split(r'(?<=[.!?])\s+', t)


def find(text, patterns):
    """Every label whose pattern the text carries, with the sentence that carried it."""
    hits = {}
    for label, pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            a = max(0, text.rfind('.', 0, m.start()) + 1)
            b = text.find('.', m.end())
            hits[label] = ' '.join(text[a:(b if b > 0 else m.end() + 90)].split())[:190]
    return hits


def main():
    rows = [r for r in csv.DictReader(io.open(CORPUS, encoding='utf-8'))
            if str(r.get('in_reviewed_corpus', '')).strip().lower() in ('1', 'true', 'yes', 'y')]
    print('corpus rows: %d' % len(rows))

    per, missing, skipped = {}, [], []
    for r in rows:
        idx = str(r['idx']).strip()
        if idx in QUARANTINED:
            skipped.append(idx)
            continue
        p = os.path.join(TEXTS, idx + '.txt')
        if not os.path.exists(p):
            missing.append(idx)
            continue
        t = io.open(p, encoding='utf-8', errors='replace').read()
        def counts(rx):
            v = [int(x.replace(',', '')) for x in rx.findall(t)]
            return sorted(set(x for x in v if 4 <= x <= 200000))[:8]
        nodes, links = counts(NODES), counts(LINKS)
        net_hits = find(t, NETWORK)
        for label in RELATED_WORK_ONLY.get(idx, ()):
            net_hits.pop(label, None)
        per[idx] = {
            'engines': find(t, ENGINES),
            'network': net_hits,
            'metrics': find(t, METRICS),
            'baselines': find(t, BASELINES),
            'nodes_stated': nodes,
            'links_stated': links,
        }

    print('texts read: %d   quarantined: %s   no text on disk: %d'
          % (len(per), ','.join(skipped) or 'none', len(missing)))
    if missing:
        print('   without a text file: %s' % ', '.join(sorted(missing, key=lambda x: int(x))))

    # ---- control 1 ---------------------------------------------------------
    named_engine = [i for i, v in per.items() if v['engines']]
    sumo = [i for i in named_engine if 'SUMO' in per[i]['engines']]
    print('\n--- control: against the manuscript\'s manually read figures ---')
    print('  studies naming a simulation engine   %3d   (manuscript states 53 of 94)' % len(named_engine))
    print('  of those, naming SUMO                %3d   (manuscript states 47 of those)' % len(sumo))
    ok = abs(len(named_engine) - 53) <= 8 and abs(len(sumo) - 47) <= 8
    print('  %s the lexical rule lands within 8 of both' % ('OK  ' if ok else 'FAIL'))
    if not ok:
        print('\nthe rule does not reproduce the one figure the manuscript already states from '
              'reading, so its other columns are not reported')
        sys.exit(1)

    # ---- the inventory -----------------------------------------------------
    def tally(field, title):
        c = Counter()
        for v in per.values():
            for k in v[field]:
                c[k] += 1
        none = sum(1 for v in per.values() if not v[field])
        print('\n--- %s (n = %d texts) ---' % (title, len(per)))
        for k, n in c.most_common():
            print('  %-30s %3d  (%.0f%%)' % (k, n, 100.0 * n / len(per)))
        print('  %-30s %3d  (%.0f%%)' % ('none detected', none, 100.0 * none / len(per)))
        return dict(c), none

    eng, eng_none = tally('engines', 'Simulation engine named')
    net, net_none = tally('network', 'Network the evaluation runs on')
    met, met_none = tally('metrics', 'Performance measure reported')
    bas, bas_none = tally('baselines', 'Comparison the study draws')

    import statistics
    print('\n--- Network size, nodes and links counted separately ---')
    for field, title in (('nodes_stated', 'nodes or intersections'), ('links_stated', 'links or edges')):
        vals = sorted(s for v in per.values() for s in v[field])
        k = sum(1 for v in per.values() if v[field])
        print('  %-22s stated by %3d of %d   median %5d   from %d to %d'
              % (title, k, len(per), statistics.median(vals), vals[0], vals[-1]))

    both = sum(1 for v in per.values()
               if 'synthetic grid or lattice' in v['network'] and
                  ('real city or corridor' in v['network'] or 'benchmark road network' in v['network']))
    print('\n  studies whose text carries BOTH a synthetic and a real network term: %d' % both)
    print('  (the rule is lexical; a study naming a grid as an illustration is counted in both)')

    json.dump({'per_study': per, 'engines': eng, 'network': net, 'metrics': met,
               'baselines': bas, 'n_texts': len(per), 'quarantined': skipped,
               'no_text': missing,
               'control': {'named_engine': len(named_engine), 'sumo': len(sumo)}},
              io.open(OUT, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print('\nwrote %s' % OUT)


if __name__ == '__main__':
    main()
