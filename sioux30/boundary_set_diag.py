# Read-only diagnostic: interior-destination Sioux Falls under the OLD boundary set
# (node 8 absent, the superseded default) vs the corrected set. The benchmark package is never written.
import os, sys, json, numpy as np
BENCH = os.environ.get('BENCHMARK_DIR',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'benchmark_network'))
if not os.path.exists(os.path.join(BENCH, "benchmark_demo.py")):
    sys.exit("benchmark_demo.py not found in %s.\n"
             "The environment module is published in benchmark_network/ of the repository "
             "at tag v1.8.0; set BENCHMARK_DIR to that directory." % BENCH)
sys.path.insert(0, BENCH)
import torch; torch.set_num_threads(1)
import benchmark_demo as B
seed=int(sys.argv[1]); variant=sys.argv[2]
net=dict(B.NETWORKS['sioux_falls']); net['od_mode']='dest_interior'
if variant=='old': net['boundary']=set(net['boundary'])-{8}
od=B.make_eval_od(net); eps=net['episodes']
out={}
for b in ('closed','open'):
    for r in ('time_min','aligned'):
        out['%s_%s'%(b,r)]=100.0*B.train_eval(net,b,r,seed,eps,od)
print(json.dumps({'seed':seed,'variant':variant,'n_eval':len(od),'distinct':len(set(od)),'cells':out}))
