Before exp3 all wrong.

Exp4: encode time of day into sin and cos, also add carbon intensity into input.

Exp5: add entropy regularization and large batch.

Exp6: add different model:separate

Exp7: wait net do not takes queue wait time as input. After this works, add queue wait time as input.

Carbon:
The upper bound: all select 4 node, doesn't wait. 
Mean Carbon Emission:61377.5067

Our wait model：
Mean TAT (Turnaround Time):    45.1837
Mean Carbon Emission:          60937.4841

The lower bound: all select 4 node, meanwhile always using the lowest carbon intensity time slot.

Total node hour: 151.81347767968
Carbon Intensity: 155.93299474119112
Total Carbon Emission: 23672

exp8: extropy=0.2, add future carbon intensity as input on 2-stage------the pic

exp9: last try on flat.

exp10: extra input: sin,cos, carbon, future carbon, only queue wait time goes into layer norm, 0.2 entropy

exp11: add more entropy

exp12: scheduled entropy， gating bias init

exp13: add GAE, feature engineering.



