# Stav projektu

**Snímek: 11. 11. 2026, aktualizace po 01:55 UTC. Celkový cíl není dokončený.**
Tento přehled rozlišuje předatelnou implementaci a dosud nedokončený výzkum.
Běžící úlohy je vždy nutné znovu ověřit; časový snímek není doklad, že stále běží.

## Co lze použít z `for_Leipzig`

| Součást | Stav a hranice ověření |
| --- | --- |
| Původní modely | Pět původních checkpointů pro cévy, A/V a disk/cup; architektury, váhy a připravené vstupy numericky ověřené. |
| Predikce | Python CLI, masky, pravděpodobnosti, overlay a metadata v původní geometrii. Rodiny mají výslovné vstupní profily. |
| Nové trénování | Příprava 21 zdrojů z RAW dat v ownCloudu, unified trainer a čerstvé trénování původních architektur. Nativní recepty nejsou reprodukcí neznámého historického optimalizátoru. |
| Unified baseline V1 | Dokončených 60 epoch, 2 221 TRAIN fotografií a 5 281 anotovaných variant ze 17 zdrojů. Vybrán checkpoint E56; není finálním modelem na širším V2. |
| Piloty | Tři starší pokusy na jednotlivých databázích, jasně označené jako piloty. |
| Instalace a kontrola | `uv`, ověřený wheel a izolovaná instalace; integrační kontrola 2 605 CPU testů. Všech devět přibalených checkpointů zachovalo původní bajty. |

Začněte [instalací a predikcí](../README.md#install),
[přípravou celého korpusu](UNIFIED_QUICKSTART.md),
[nativním trénováním](NATIVE_LEGACY_RETRAINING.md) nebo
[přehledem modelů](../README.md#predict-immediately-with-included-models).

## Co je skutečně změřené

Připravené vstupy původních sítí zachovávají ověřenou numeriku. Celý raw A/V
řetězec není obecně bitově shodný s MATLABem; zbývající fotometrické rozdíly jsou
popsané v [dokumentaci profilu](MATLAB_AV_PROFILE.md). Oprava nearest resize
souhlasí s 24 syntetickými a 5 skutečnými FOV referencemi. MATLAB není závislostí
běžné instalace ani predikce.

Čerstvá původní sestava ze zdroje `193662c` na všech 770 VAL fotografiích přesně
zopakovala dřívější počty TP/FP/FN a podmíněné A/V: macro dataset/head Dice
0,66703, pooled vessel Dice 0,77012, conditional A/V accuracy 0,74857.
[Důkaz a celý report](https://github.com/tomasvicar/DeepRetinaSegmentation/tree/experiments/leipzig-hpo/reports/frozen-original-val-reproduction).
Po nearest opravě dokončil zdroj `d14aa14` nové měření všech 770 VAL snímků.
Počty pro cévy, disk a cup zůstaly stejné; A/V se nepatrně změnilo na dvou HRF
snímcích (o 18 správných pixelů méně z 2 626 284). Macro Dice je 0,667029585.
Jde o ověřenou opravu shody operátoru s MATLABem, nikoli o doložené zlepšení
kvality predikce. [Nový report a přesné rozdíly](https://github.com/tomasvicar/DeepRetinaSegmentation/tree/experiments/leipzig-hpo/reports/frozen-original-val-nearest).

Stejný skutečný raný compact checkpoint dává ve zmrazeném trénovacím zdroji
`6b3dfad` a předatelné větvi bitově stejné výstupy na testovaných CPU vstupech,
včetně raw DRIONS TRAIN snímku. To je
[důkaz kompatibility inference](../reports/v2-inference-parity/README.md),
nikoli důkaz finální kvality nebo obecné shody všech GPU backendů.

## Co právě pokračuje

Korpus V2 má 2 529 TRAIN fotografií / 5 700 anotovaných variant z 21 zdrojů,
770 VAL a 856 TEST fotografií. Původní VAL data zůstala beze změny. Geometrické
předpoklady nových zdrojů jsou výslovné a uložené s daty i checkpointem;
[přehled rozsahu a omezení](CORPUS_V2_PREPARATION.md).

Kontrolovaná studie porovnává control a šest variant loss, geometrie a augmentací
v etapách 3/9/27/60 na třech vlastních GPU. Při poslední zaznamenané kontrole
control dokončil 50 epoch; všechny fotografie a zdroje již přispěly, ale pouze
5 699 z 5 700 anotovaných variant. Ještě to není dokončený model. Návazná
validace má samostatnou kontrolu skutečného E60 a úplného pokrytí anotací.

Průběžný checkpoint E25 zlepšuje disk a cup, ale pooled vessel Dice a podmíněné
A/V zatím zaostávají za původní sestavou. Nelze jej označit za zlepšení všech úloh.
TRAIN audit navíc ukázal přibližně pětinový pokles očekávané cévní/A/V expozice
po přidání nových disk/cup zdrojů při stejném rozpočtu. Další experiment proto
má samostatnou implementaci úpravy samplingu na větvi
`experiments/v2-vascular-mixture`; její lepší modelová kvalita zatím změřená není.
[Průběžné důkazy a scénáře](https://github.com/tomasvicar/DeepRetinaSegmentation/tree/experiments/leipzig-hpo/reports/data).

## Co zbývá k dokončení cíle

1. Dokončit a zkontrolovat skutečné trénovací běhy včetně pokrytí všech vhodných TRAIN dat.
2. Změřit čerstvý párový VAL po úlohách a zdrojích; optimalizací řešit zbývající cévní/A/V regresi.
3. Zmrazit vybraný checkpoint, konfiguraci, prahy a inference protokol; poté provést samostatný TEST.
4. Doložit kvalitu, velikost a rychlost vybraného modelu a předat jeho váhy vedle původních sítí.

**Nový finální V2 checkpoint a jeho TEST výsledky zatím nejsou vydané.**
Podrobný cíl je v [GOAL.md](GOAL.md), historické záznamy v
[PROGRESS_HISTORY_2026-11.md](PROGRESS_HISTORY_2026-11.md). Operátorské přístupy a
spotřeba MetaCentra jsou vedené v samostatném soukromém provozním repozitáři.
