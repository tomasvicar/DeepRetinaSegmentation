# Stav projektu

**Snímek: 12. 11. 2026 v 10:36 UTC. Původní společná predikce je předatelná; finální optimalizovaný model ještě není dokončený.**
Tento přehled rozlišuje předatelnou implementaci a dosud nedokončený výzkum.
Běžící úlohy je vždy nutné znovu ověřit; časový snímek není doklad, že stále běží.

## Co lze použít z `for_Leipzig`

| Součást | Stav a hranice ověření |
| --- | --- |
| Původní modely | Pět původních checkpointů pro cévy, A/V a disk/cup; architektury, váhy a připravené vstupy numericky ověřené. |
| Predikce | Python CLI, masky, pravděpodobnosti, overlay a metadata v původní geometrii. Rodiny mají výslovné vstupní profily. |
| Společná původní predikce | `predict-original`: pevná sestava vessel + disc0 + universal cup + AV; všech pět masek na skutečné TRAIN DRIVE21 fotografii pixelově shodných s původním benchmarkem. Dodaná FOV a známá kamera/kalibrace jsou povinné. [Rozsah ověření](PREDICT_ORIGINAL.md). |
| Nové trénování | Příprava 21 zdrojů z RAW dat v ownCloudu, unified trainer a čerstvé trénování původních architektur. Nativní recepty nejsou reprodukcí neznámého historického optimalizátoru. |
| Unified baseline V1 | Dokončených 60 epoch, 2 221 TRAIN fotografií a 5 281 anotovaných variant ze 17 zdrojů. Vybrán checkpoint E56; není finálním modelem na širším V2. |
| Unified control V2 | Pevný terminální E60, 21 zdrojů, všech 2 529 TRAIN fotografií / 5 700 anotovaných variant skutečně přispělo. Čerstvá VAL; cévní regrese vůči původní sestavě, není optimalizovaný finální model. |
| Unified vascular2 preview | Dokončený běh 60 epoch; přibalený VAL-best E59 už pokrývá všech 2 529 fotografií / 5 700 variant. Macro 0,80390, conditional A/V 0,81026; vessel 0,75671 stále pod původním 0,77012. Experimentální, bez TEST. |
| Unified TPE preview | První dokončený TPE běh, BEST58 po 60 epochách; všech 2 529 fotografií / 5 700 variant. Čerstvá NONE macro 0,8108954, vessel 0,7628822, conditional A/V 0,8249115. Volitelné rot4/dihedral8 mají samostatné měření kvality i ceny; stále cévní regrese a TEST pending. [Karta a recept](UNIFIED_V3_TPE_PREVIEW.md). |
| Piloty | Tři starší pokusy na jednotlivých databázích, jasně označené jako piloty. |
| Instalace a kontrola | `uv`, ověřený wheel a izolovaná instalace; předchozí integrační kontrola 2 605 CPU testů; přidání V2 ověřuje 59 cílených testů, CPU RAW smoke a wheel s deseti modely. Vascular2 preview přidává 45 cílených CPU testů, RAW TRAIN smoke a wheel s jedenácti modely; všech deset dřívějších vah zachovalo původní bajty. TPE preview má 32 cílených testů, skutečný RAW TRAIN rot4 smoke a wheel s dvanácti modely; všech jedenáct starších checkpointů i aliasy zůstaly přesně stejné. |

Počty testů v tabulce jsou dřívější zdrojově vymezené kontroly, nikoli nové
spuštění celé aktuální sady; [katalog a důkazy](TESTING.md) rozlišují jejich rozsah.

Začněte [instalací a predikcí](../README.md#install),
[přípravou celého korpusu](UNIFIED_QUICKSTART.md),
[nativním trénováním](NATIVE_LEGACY_RETRAINING.md) nebo
[přehledem modelů](../README.md#predict-immediately-with-included-models).

Nové trénování původních architektur má také [kontrolu na skutečných datech](../reports/native-legacy-real-smoke/README.md):
vessel320, disc192, disc256 a universal288 dokončily po jedné epoše na samostatné
čtyřfotografové TRAIN/VAL podmnožině, se skutečnými aktualizacemi parametrů,
celosnímkovou validací a ověřeným načtením checkpointů. Nejde o plné přeučení ani
měření kvality. Celosnímkový A/V pokus na CPU zastavil limit RSS 3 GiB; tento
neúspěšný výsledek zůstává zachovaný. Následný samostatný běh na GTX1060 6GB
skutečně dokončil jeden FP32 krok na plném TRAIN snímku a celosnímkovou VAL,
poté bezpečné načtení a konečné CUDA výstupy; všech 54 parametrických tenzorů
se změnilo proti čerstvé inicializaci se stejným seedem. Všech pět nativních
presetů tak má funkční kontrolu na reálných datech. A/V pokrývá jedinou vhodnou
TRAIN fotografii/anotaci, nikoli celý korpus; paměťový peak je pouze z verifikace,
ne z trénování. [Podrobný GPU důkaz](../reports/native-legacy-real-smoke/av-GPU-functional-proof.json).
Původní váhy, produkční architektury, konfigurace a počet dvanácti dodávaných
modelů zůstaly stejné; jednokrokové kontrolní váhy se nevydávají.

Každá predikce vyžaduje nový nebo prázdný výstupní adresář. Opakované použití
zaplněného adresáře nyní skončí před načtením modelu a nezmění dřívější soubory;
předchozí chování mohlo například ponechat starou cup masku vedle nové cévní
predikce. Opravu ověřilo 66 cílených CPU testů exportu, původních modelů a CLI.

## Co je skutečně změřené

Nový `predict-original` prošel skutečným porovnáním celé sestavy na jedné TRAIN
fotografii: rozdíly pěti masek `[0, 0, 0, 0, 0]`, skutečný child i outer exit 0,
CPU1 a bez inicializace CUDA. To ověřuje shodu s Python benchmarkem na tomto
vstupu; neřeší obecnou bitovou shodu raw fotometrie s MATLABem.
[Originální doklady](../reports/original-composed-real-parity-20261112/README.md)
zachovávají i dva předchozí neúspěšné preflighty a opravu parametrů dlaždic.
Po opravě prošlo jedenáct cílených testů se skutečnou strukturou metadat dlaždic
a [izolovaná instalace](../reports/product-original-composed-packaging/README.md):
46 Python modulů a všech 12 vah souhlasilo se zdrojem, registry/aliasy i 42
existujících distribucí závislostí se nezměnily.

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

## Výzkum: časový snímek 12. 11. v 10:36 UTC

Výsledky níže jsou VAL, nikoli TEST; žádný nový kandidát se tím nestává výchozím
nebo finálním modelem. Dřívější smíšené časové aktualizace zůstávají beze změny
v [historickém přehledu](PROGRESS_HISTORY_2026-11.md#archivovaný-výzkumný-přehled-ze-zdroje-5545f3d).

- **Kontrolované raw/photometry běhy:** raw256, raw384, raw512, CLAHE384 a LC384
  dokončily trénink 60 epoch. Raw512 má čerstvou VAL obou rolí; LC pouze BEST,
  protože plánovaný LAST průchod zastavil původní časový limit. CLAHE má oba
  úplné průchody, ale proti odpovídajícímu raw384 zhoršilo všech pět pooled Dice
  i podmíněnou A/V accuracy, se zápornými párovými intervaly.
  [Negativní CLAHE důkaz](https://github.com/tomasvicar/DeepRetinaSegmentation/tree/experiments/leipzig-hpo/reports/clahe384-typed-fresh-val-20261112).
- **Vessel loss2:** celý trénink E60, QA i čerstvá LAST60 VAL jsou dokončené.
  Dense vessel Dice 0,74406 zlepšilo odpovídající první TPE LAST60 o 0,00555
  (95% párový interval [0,00349; 0,00797]), ale zůstává pod původními 0,76133.
  Cup source-macro Dice se zhoršilo; nejde o zlepšení všech úloh.
  [Kontrolované porovnání a všechny regrese](https://github.com/tomasvicar/DeepRetinaSegmentation/tree/experiments/leipzig-hpo/reports/loss2-LAST60-matched-val-20261112).
- **Nativní pomocný TPE trial8:** dokončený E60; BEST51 i LAST60 mají vlastní
  pokrytí všech 2 529 TRAIN fotografií / 5 700 variant a úplnou čerstvou VAL770.
  BEST dokončil výpočet a zápis reportu, ale dodatečná chybná kontrola metadat
  skončila worker/outer1; samostatná opravená CPU kontrola report přijala s kódem 0.
  Tento neúspěšný exit zůstává zachovaný. Nový LAST průchod skončil child/outer0.
  Macro BEST/LAST je 0,80853/0,80240, dense vessel Dice 0,74245/0,74510.
  Obě cévní regrese vůči původnímu modelu mají záporné celé párové intervaly.
  [Obě role, skutečné exity a úplné výsledky](https://github.com/tomasvicar/DeepRetinaSegmentation/tree/experiments/leipzig-hpo/reports/native8-completed-f13-VAL-20261112).
- **Šířka24:** pokračování původního modelu z E27 do E60 není dokončené;
  poslední uvedený záznam z 10:36:43 UTC je E39/79 872 kroků. Není tím doložená
  terminální QA ani výsledná VAL kvalita. Konfigurace a běžící trénink se nemění.
- **Raw512/rot4 prahy:** jednorázová VAL kalibrace běží; záznam z 10:35:58 UTC
  uvádí 395/770 fotografií. Výsledek není v tomto snímku dokončený a nelze z něj
  tvrdit nový práh, lepší kvalitu nebo změnu výchozího chování.

Primární doplňková dense analýza vylučuje pouze známý pozitivně anotovaný AVRDB:
87 fotografií / 55 známých skupin / 8 zdrojů. Jeho dalších 20 fotografií nemá
platné negativní cévní pixely; FP0/precision1 proto není důkaz husté segmentace.
Například native8 BEST převyšuje LAST při započtení AVRDB, ale na dense podpoře
je pořadí opačné. Raw512 BEST50 s rot4/0,35 má dense Dice 0,75132, stále pod
původními 0,76133; změnily se současně TTA i práh. Samotné macro nebo pooled
skóre zahrnující AVRDB nemůže zbývající cévní deficit nahradit.
[Raw512 měření a omezení](https://github.com/tomasvicar/DeepRetinaSegmentation/tree/experiments/leipzig-hpo/reports/raw512-BEST50-rot4-035-fresh-val-20261112).
Známé skupiny nejsou důkazem úplné nezávislosti pacientů; intervaly jsou průzkumné.
Dosavadní studijní cíle a zdrojové reporty se touto doplňkovou analýzou nemění.

Korpus V2 má 21 TRAIN zdrojů / 2 529 fotografií / 5 700 anotovaných variant,
770 VAL a 856 TEST fotografií; pevné VAL členství zůstalo zachované.
Geometrické předpoklady jsou výslovné v [datové dokumentaci](CORPUS_V2_PREPARATION.md).
Finální výběr, zmrazení přesného protokolu a následný TEST nejsou dokončené.

## Co zbývá k dokončení cíle

1. Dokončit a zkontrolovat skutečné trénovací běhy včetně pokrytí všech vhodných TRAIN dat.
2. Porovnávat další kandidáty párově po úlohách a zdrojích; kontrolovaně změřit patch, preprocessing a inference dlaždice a řešit zbývající cévní regresi.
3. Zmrazit vybraný checkpoint, konfiguraci, prahy a inference protokol; poté provést samostatný TEST.
4. Doložit kvalitu, velikost a rychlost vybraného modelu a předat jeho váhy vedle původních sítí.

**Přibalené jsou původní modely, kontroly V1/V2 a samostatné vascular2/TPE preview; optimalizovaný finální model a TEST výsledky zatím nejsou vydané.**
Podrobný cíl je v [GOAL.md](GOAL.md), historické záznamy v
[PROGRESS_HISTORY_2026-11.md](PROGRESS_HISTORY_2026-11.md). Operátorské přístupy a
spotřeba MetaCentra jsou vedené v samostatném soukromém provozním repozitáři.
