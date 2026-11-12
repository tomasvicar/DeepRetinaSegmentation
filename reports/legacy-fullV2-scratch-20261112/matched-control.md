# Matched firstTPE5 LAST60 control on the frozen common evaluator

The baseline completed on 12 November 2026 at16:04:43UTC. Actual child,
observer and outer transport returned0; all770VAL photos /993annotation records
were evaluated with unchanged inputs under frozenfe33/895 source. This is a
fresh evaluation of the original fixed LAST60, not another training run.

Protocol: FP32, noautocast, noTTA, threshold0.5, tile512/overlap64 on ownOld GTX1060.
The fixed LAST60 checkpoint0841d39a…dbc5d4 has its own2529TRAINphotos/5700record
coverage. Its separate BEST58 checkpoint is not substituted or used to borrow
coverage. Full source/model/report identities and allfivehead counts are in
[the result](matched-firstTPE5-LAST60-common-source.json).

Macro source Dice0.8088377564515558; pooled vessel Dice0.7608041998967329;
conditional artery/vein accuracy0.8217584236891364. Actual wall222.770s;
charged memory peak1012.7MiB, swap0 under3GiB cap. These are whole-validation
costs, not per-image benchmark claims. Root independently rehashed the report,
checked observed child/outer0 and recomputed allfivepooled integer counts andDice
from all770unique per-photo records, without another NN/GT read.

Future native architecture contrast uses the same evaluator and fixed LAST60
protocol/support. No native60 result exists at this baseline-recording step.
Historical nativeb269/Python3.12.13 report7570255b…022be is unchanged; tiny backend
rounding differences in this Python3.12.14 evaluation are not relabelled as that
historical report. The shipped raw512 rot4/.35 model uses a different protocol,
so its score is not a one-factor topology comparison with this baseline.

Original source/GT/CPUeligibility/custody/observer receipts remain in the private
operations repository under reports/TPE5-control-common-fe33-20261112. No TEST
or sampler/Optuna operation was performed for this control.
