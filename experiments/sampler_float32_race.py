"""Darkbloom junk tokens (2026-11-24): the exponential-race sampler with -log(1-u) in Float32 vs the -log1p(-u) fix,
using MLX's own RNG on an Apple GPU and a gpt-oss-sized vocabulary. Run on a Mac: pip install mlx numpy; python this.py
Result on M5 (MLX 0.32.4): old 54 picks below p=1e-10 per 10k sampled tokens, new 0 of 30,000."""
import mlx.core as mx, numpy as np, math
V=201088; B=100; STEPS=300  # 30,000 sampled tokens
mx.random.seed(0)
rng=np.random.default_rng(0)
stats={"old":[0,0],"new":[0,0]}; tiny=0; zero=0
for s in range(STEPS):
    # realistic-ish next-token distribution: 30 plausible tokens, the rest far in the tail (p ~ 1e-13..1e-20)
    logits=np.full((B,V),-35.0,np.float32)+rng.normal(0,3,(B,V)).astype(np.float32)
    idx=rng.integers(0,V,(B,30)); 
    for b in range(B): logits[b,idx[b]]=rng.normal(0,2,30)
    lg=mx.array(logits); logp=lg-mx.logsumexp(lg,axis=-1,keepdims=True); p=mx.exp(logp)
    u=mx.random.uniform(low=0.0,high=1.0,shape=(B,V),dtype=mx.float32,key=mx.random.key(s))
    tiny+=int(mx.sum(u<=2**-25).item()); zero+=int(mx.sum(u==0).item())
    for name,e in (("old",mx.maximum(-mx.log(1-u),mx.array(1e-20,mx.float32))),("new",-mx.log1p(-mx.maximum(u,mx.array(2**-33,mx.float32))))):
        w=mx.argmax(p/e,axis=-1); lw=mx.take_along_axis(logp,w[:,None],axis=-1)[:,0]
        stats[name][0]+=B; stats[name][1]+=int(mx.sum(lw<math.log(1e-10)).item())
n=STEPS*B
print(f"draws u<=2^-25: {tiny} of {n*V} = {tiny/(n*V):.3e} (2^-25 = {2**-25:.3e}); u==0: {zero}")
for k,(t,j) in stats.items(): print(f"{k}: {j}/{t} sampled tokens below p=1e-10 = {j/t*1e4:.1f} per 10k")
print("predicted old rate:", round((1-(1-2**-25)**V)*1e4,1),"per 10k")
