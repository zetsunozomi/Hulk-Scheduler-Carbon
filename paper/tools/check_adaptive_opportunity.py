#!/usr/bin/env python3
"""An exact illustrative construction, NOT an AMSP/trace experiment or RL result.

Eight-node toy cluster; four background nodes are occupied until hour 2.
U=7 work units, q4=1 and q8=3/2 units/hour, max allocation H=2 hours.
Zero overhead; unit per-node power; CI=1/10 on [2,4), otherwise CI=2.
All requests are immediate at the previous checkpoint; no scheduler changes.
"""
from fractions import Fraction as F


def exposure(start,end):
    low=max(F(0),min(end,F(4))-max(start,F(2)))
    return 2*(end-start-low)+low/10


def execute(sequence):
    remaining,t,c=7,F(0),F(0)
    for n in sequence:
        assert n in (4,8) and remaining>0
        q=F(1) if n==4 else F(3,2)
        progress=min(remaining,int(2*q))
        start=max(t,F(2)) if n==8 else t
        end=start+progress/q
        c+=n*exposure(start,end)
        t=end;remaining-=progress
    assert remaining==0
    return t,c


if __name__=='__main__':
    cases={'Fixed-4':[4]*4,'Fixed-8':[8]*3,'4-8-8':[4,8,8],'4-8-4':[4,8,4]}
    results={name:execute(path) for name,path in cases.items()}
    assert results['Fixed-4']==(F(7),F(204,5))
    assert results['Fixed-8']==(F(20,3),F(664,15))
    assert results['4-8-8']==(F(16,3),F(584,15))
    assert results['4-8-4']==(F(6),F(168,5))
    for dynamic in ('4-8-8','4-8-4'):
        for fixed in ('Fixed-4','Fixed-8'):
            assert all(x<y for x,y in zip(results[dynamic],results[fixed]))
    assert results['4-8-8'][0]<results['4-8-4'][0] and results['4-8-8'][1]>results['4-8-4'][1]
    print('Illustrative construction only; exact arithmetic verifies two nondominated dynamic points beat both fixed points.')
    for name,(t,c) in results.items():print(f'{name}: time={float(t):.6f}, carbon_units={float(c):.6f}')
