# CPU audit inference V2: frozen 6b3dfad proti stable f1c4c99

**Výsledek: všechny měřené výstupy `compact_unet` jsou bitově shodné.** Použit byl
neměnný skutečný E2 snapshot a jeho DRIONS TRAIN fotografie. Nebyly otevřeny GT
masky ani spuštěna VAL/TEST evaluace nebo trénink. Nejde o důkaz kvality ani E60
completion; jde o kompatibilitu inference stejného checkpointu.

[Strojový report](actual-e2-cpu/verification.json) obsahuje hashe zdrojů, vstupů,
mezivýstupů a predikcí, celé receipts obou subprocessů a samostatný nearest příklad.

| Kontrola | Výsledek |
| --- | --- |
| Přímé logits, vstupy64×64, 37×53 a 17×19 | bitově shodné, max. rozdíl0 |
| Tiled pravděpodobnosti stejných syntetických vstupů | bitově shodné, max. rozdíl0 |
| Skutečný raw DRIONS TRAIN snímek 400×600, defaultní tile512 / overlap64 | všech 5 pravděpodobnostních map a FOV bitově shodné |
| Syntetický raw RGB 67×91, explicitní all-ones FOV | pravděpodobnosti/FOV bitově shodné |
| Vessel/artery/vein/disc/cup PNG, AV a overlay v obou raw případech | shodné pixely iPNG bytes |
| Dříve uchovaný skutečný E2 smoke proti novému stable běhu | všechny uložené NPZ arrays bitově shodné |

Skutečný DRIONS vstup má podle stejné uložené nominální politiky prepared tvar
500×750 a rescale1.25. V obou bězích byly shodné iúdaje preprocessing/geometry.
Checkpoint SHA256 je
`34e69e9d601092db1bbd71872009a24b183333da296f451113a1303da69eb2d7`,
vstupní JPEG SHA256
`c510f7c3a90c35a75cde75661668664b1c6d97f3c872476ed668d1ebdb190ff1`.
Originální snapshot, obrázek a frozen zdroj byly zachovány; tracked status frozen
worktree po auditu zůstal čistý. Do zdrojů se nepsal ani bytecode.

## Statická kontrola

`geometry_policy.py`, `dataset.py`, `inference.py`, `legacy/preprocessing.py`
a `legacy/single_net.py` jsou mezi 6b3dfad a f1c4c99 byte-identické. Deset relevantních
funkcí v unified.py je AST-identických, včetně `evaluate_unified_model`,
`load_unified`, inference konfigurace, aktivací, fusion, obnovy geometrie, dekódování,
tiled pravděpodobností a `predict_unified`.

Rozdíl v`model.py` odstraňuje pomocný návrat `(features,(h,w))` a inline provede stejný
head před stejným cropem. Nemění se vrstvy, state dictionary ani pořadí výpočtů pro
compact model. Z`unified.py` zmizela volitelná AV-residual větev a její položka vhashi
trénovacího zdroje. Snapshot je `compact_unet`, takže tuto větev nepoužívá.

Změněný hash trénovacího zdroje není oprávnění kresume starého běhu ve stable:
retraining/resume afresh VAL producer nadále používají správný frozen zdroj.
Tento audit prokazuje měřenou inference kompatibilitu konkrétního checkpointu.

## Nearest rozdíl je skutečný, ale mimo compact cestu

`legacy/matlab_resize.py::imresize_nearest_mask` ve stable mění float64 half-pixel
operace. Samostatný syntetický checkerboard1152×11→1005×11 má 22 odlišných pixelů,
v řádcích 502 a 837 (indexováno od 0). Tento výsledek není skryt ani připsán jako
bezzměnová kompatibilita původních MATLAB A/V cest.

Compact raw inference tento helper neimportuje/nevolá. Používá byte-identický
`legacy.preprocessing.prepare_single_net`, `ImageGeometry` a skimage resize;
compact decoder používá nezměněné bilinear interpolate. Proto tento nearest rozdíl
neovlivnil měřené compact logits, přípravu ani obnovu map. U samostatných původních
MATLAB A/V pipelines je třeba zachovat vlastní source/policy pin a znovu měřit jejich
konkrétní prepared vessel/FOV a class maps; compact audit jejich shodu netvrdí.

## Reprodukce aomezení

V obou subprocessích bylo použito stejné env87, CPU affinity0, nice10,
`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`, prázdné
`CUDA_VISIBLE_DEVICES` a`PYTHONDONTWRITEBYTECODE=1`. Worker nastavuje Torch intra/inter-op
na 1 a blokuje CUDA init. Jeden subprocess importuje skutečný immutable frozen6b3
worktree, druhý nový audit worktree založený na f1c4c99; importovaný package path se
kontroluje. Zapisují se pouze nové scratch výstupy mimo zdroje a snapshot.

Spusťte [worker](../../scripts/audit_v2_inference_parity_worker.py) dvakrát se stejnými
`--checkpoint`/`--image`, odlišným `--source`/`PYTHONPATH` a čerstvým `--output`.
Poté [summarizer](../../scripts/summarize_v2_inference_parity.py) přijme
`--left FROZEN_OUTPUT --right STABLE_OUTPUT --output NEW_REPORT_DIRECTORY`.
Skutečné scratch výstupy tohoto auditu jsou `/tmp/retina-parity-6b3dfad-final`
a`/tmp/retina-parity-f1c4c99-final`; do Git se ukládají jen metadata/hash evidence.

Pozorovaná bitová shoda platí pro tento checkpoint, tyto vstupy, CPU astejný software
backend. Není to univerzální tvrzení pro GPU, jiné knihovny, všechny budoucí váhy
nebo všechny raw kamery. Není zde měřena přesnost, klinická validita ani runtime
latence. Žádná produkční funkce, uložený model, input, TEST artefakt nebo live job
nebyl změněn. Ruff obou pomocných audit skriptů prošel.
