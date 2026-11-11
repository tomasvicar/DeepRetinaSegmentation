# Stav projektu

**Snímek: 11. 11. 2026, po vydání kontrolního V2 a cévně váženého preview. Celkový cíl není dokončený.**
Tento přehled rozlišuje předatelnou implementaci a dosud nedokončený výzkum.
Běžící úlohy je vždy nutné znovu ověřit; časový snímek není doklad, že stále běží.

## Co lze použít z `for_Leipzig`

| Součást | Stav a hranice ověření |
| --- | --- |
| Původní modely | Pět původních checkpointů pro cévy, A/V a disk/cup; architektury, váhy a připravené vstupy numericky ověřené. |
| Predikce | Python CLI, masky, pravděpodobnosti, overlay a metadata v původní geometrii. Rodiny mají výslovné vstupní profily. |
| Nové trénování | Příprava 21 zdrojů z RAW dat v ownCloudu, unified trainer a čerstvé trénování původních architektur. Nativní recepty nejsou reprodukcí neznámého historického optimalizátoru. |
| Unified baseline V1 | Dokončených 60 epoch, 2 221 TRAIN fotografií a 5 281 anotovaných variant ze 17 zdrojů. Vybrán checkpoint E56; není finálním modelem na širším V2. |
| Unified control V2 | Pevný terminální E60, 21 zdrojů, všech 2 529 TRAIN fotografií / 5 700 anotovaných variant skutečně přispělo. Čerstvá VAL; cévní regrese vůči původní sestavě, není optimalizovaný finální model. |
| Unified vascular2 preview | Dokončený běh 60 epoch; přibalený VAL-best E59 už pokrývá všech 2 529 fotografií / 5 700 variant. Macro 0,80390, conditional A/V 0,81026; vessel 0,75671 stále pod původním 0,77012. Experimentální, bez TEST. |
| Piloty | Tři starší pokusy na jednotlivých databázích, jasně označené jako piloty. |
| Instalace a kontrola | `uv`, ověřený wheel a izolovaná instalace; předchozí integrační kontrola 2 605 CPU testů; přidání V2 ověřuje 59 cílených testů, CPU RAW smoke a wheel s deseti modely. Vascular2 preview přidává 45 cílených CPU testů, RAW TRAIN smoke a wheel s jedenácti modely; všech deset dřívějších vah zachovalo původní bajty. |

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

Kontrolovaná studie sedmi konfigurací skončila: control dokončil 60 epoch, pět
variant bylo zastaveno a šestá varianta skončila po E21 na časovém limitu.
Control prošel přísnou QA: přispělo všech 2 529 fotografií
(minimálně 66 návštěv) a všech 5 700 variant (minimálně 2 příspěvky). Vydává se
pevný terminální E60 `unified-v2-control`, nikoli VAL-best E52, který měl jen
5 699 přispívajících variant. [Model card](UNIFIED_V2_CONTROL.md) rozlišuje role.
Samostatný cévně vážený experiment dokončil 11. 11. všech 60 epoch
(491 520 vzorků); původní supervisor zaznamenal návratový kód 0. Nezávislá
kontrola obou checkpointů i čerstvá VAL prošly. Přibalený BEST E59 pokrývá
všech 5 700 variant; terminální E60 a BEST mají samostatné výsledky.
Párové porovnání E60 proti kontrolnímu E60 dokládá růst cévního Dice o 0,00575
(95% interval [0,00226; 0,00934]) a podmíněné A/V přesnosti o 0,01407
([0,00691; 0,02144]). Cévy stále zaostávají za původní sítí.

Čerstvý E60 na 770 VAL fotografiích má macro Dice 0,78876 proti původní sestavě
0,66703, ale pooled vessel Dice 0,75047 proti 0,77012: přetrvává doložená cévní
regrese. Artery je nižší v bodovém odhadu (párový interval protíná nulu); vein,
disk/cup a podmíněná A/V jsou vyšší. Nelze jej označit za zlepšení všech úloh.
TRAIN audit navíc ukázal přibližně pětinový pokles očekávané cévní/A/V expozice
po přidání nových disk/cup zdrojů při stejném rozpočtu. Další experiment proto
má samostatnou implementaci úpravy samplingu na větvi
`experiments/v2-vascular-mixture`; dokončený běh a čerstvá VAL nyní podkládají
samostatný [preview model E59](UNIFIED_V2_VASCULAR2.md). Jeho cévní regrese přetrvává;
nejde o finální výběr ani důkaz zlepšení všech úloh.
[Průběžné důkazy a scénáře](https://github.com/tomasvicar/DeepRetinaSegmentation/tree/experiments/leipzig-hpo/reports/data).

## Co zbývá k dokončení cíle

1. Dokončit a zkontrolovat skutečné trénovací běhy včetně pokrytí všech vhodných TRAIN dat.
2. Porovnávat další kandidáty párově po úlohách a zdrojích; optimalizací řešit zbývající cévní regresi.
3. Zmrazit vybraný checkpoint, konfiguraci, prahy a inference protokol; poté provést samostatný TEST.
4. Doložit kvalitu, velikost a rychlost vybraného modelu a předat jeho váhy vedle původních sítí.

**Použitelný kontrolní V2 checkpoint a samostatný vascular2 preview jsou přibalené; optimalizovaný finální model a TEST výsledky zatím nejsou vydané.**
Podrobný cíl je v [GOAL.md](GOAL.md), historické záznamy v
[PROGRESS_HISTORY_2026-11.md](PROGRESS_HISTORY_2026-11.md). Operátorské přístupy a
spotřeba MetaCentra jsou vedené v samostatném soukromém provozním repozitáři.
