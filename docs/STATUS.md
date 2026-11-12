# Stav projektu

**Snímek: 12. 11. 2026 v 02:55 UTC. Původní společná predikce je předatelná; finální optimalizovaný model ještě není dokončený.**
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

## Co právě pokračuje

Kontrolované varianty raw256, raw384, raw512, CLAHE384 a LC384 dokončily trénink
60 epoch. Každý použitý checkpoint musí mít vlastní doložené pokrytí anotací;
dokončení tréninku samo nenahrazuje čerstvou validaci. Raw512 ji dokončilo pro BEST50 i LAST60; u LC uspěl pouze BEST, terminální
VAL zastavil limit.
CLAHE dokončilo oba průchody všech 770 VAL fotografií, ale proti odpovídajícím
raw384 checkpointům zhoršilo všech pět pooled Dice i podmíněnou A/V accuracy,
vždy se záporným párovým 95% intervalem. [Úplný negativní výsledek a omezení](https://github.com/tomasvicar/DeepRetinaSegmentation/tree/experiments/leipzig-hpo/reports/clahe384-typed-fresh-val-20261112).
Žádná z těchto vah tím nezískává status finálního modelu.

Pooled cévní skóre zahrnuje AVRDB s pouze pozitivní platnou podporou. Samostatný
výpočet na plně anotovaných cévách proto vylučuje jen AVRDB a zachovává 87
fotografií / 55 známých skupin / 8 zdrojů. Například TPE BEST58 s rot4/0,35 má
zde Dice 0,74824 proti původním 0,76133, párový rozdíl −0,01309
(95% CI [−0,01856; −0,00879]). Blízká pooled skóre tedy neznamenají shodu na
těchto datech; [karta modelu](UNIFIED_V3_TPE_PREVIEW.md) tento deficit uvádí.
Dokončený raw512 BEST50 s rot4/0,35 zlepšil stejný model proti NONE/0,5:
macro Dice o 0,01150 a dense vessel Dice o 0,00409, v obou případech s kladným
párovým intervalem. Změnily se současně TTA i práh; nelze připsat efekt pouze
jednomu z nich. Dense vessel Dice 0,75132 stále zaostává za původním 0,76133
a zůstávají také regrese některých zdrojů. [Výsledky a všechny úlohy](https://github.com/tomasvicar/DeepRetinaSegmentation/tree/experiments/leipzig-hpo/reports/raw512-BEST50-rot4-035-fresh-val-20261112).

Nativní Optuna trial7 dokončil všech 60 epoch. Jeho LAST60 má vlastní plné
pokrytí 5 700 variant; BEST46 má jen 5 699 a nesplňuje tuto podmínku předání.
Čerstvý LAST60 dense vessel Dice 0,74785 je vyšší než odpovídající první TPE
LAST60 (0,73851), ale stále pod původní sestavou. [Párové výsledky a omezení](https://github.com/tomasvicar/DeepRetinaSegmentation/tree/experiments/leipzig-hpo/reports/TPE7-LAST60-matched-val-20261112).

Samostatný kontrolovaný pokus s dvojnásobnou vahou vessel loss se skutečně učí
na Adě (poslední kontrola 02:54 UTC: dokončeno 24/60 epoch). Širší jednotný model
prošel první zkušební epochou; nové pokračování do E27 je odeslané přes PBS.
Tyto rozběhnuté pokusy dosud nemají konečné porovnání kvality. Finální výběr,
zmrazení přesného protokolu a následný TEST zůstávají nedokončené.

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

Další dvě úpravy tohoto preview byly skutečně změřené. Osm globálních prahů
vybralo 0,45 podle bodového worst6 skóre; malý růst cévního Dice měl interval
zahrnující nulu a cup mírně klesl. Přepočet BatchNorm statistik pouze z TRAIN
zhoršil macro i podmíněnou A/V přesnost. Výchozí model a práh 0,5 proto zůstávají.
[Prahy](https://github.com/tomasvicar/DeepRetinaSegmentation/tree/experiments/leipzig-hpo/reports/vascular2-thresholds)
a [BatchNorm](https://github.com/tomasvicar/DeepRetinaSegmentation/tree/experiments/leipzig-hpo/reports/vascular2-bn-recalibration)
obsahují všechny výsledky a párové intervaly, včetně neúspěšných úprav.

Na jednom DRIVE TRAIN21 snímku a jednom CPU vlákně má aktuální preview měřený
resident medián 6,03 s proti 63,32 s původních čtyř sítí; parametrů je 1,11 mil.
proti 5,94 mil. Jde o konkrétní příklad včetně předzpracování a obnovy geometrie,
bez načítání modelů/souborů. [Rozsah měření](UNIFIED_V2_VASCULAR2.md#size-and-measured-cpu-example).

## Další optimalizace: stav 11. 11.

Sampling s cévní vahou 4 dokončil všech 60 epoch. Čerstvá VAL jeho BEST E55
proti přibalenému vascular2 BEST E59 zlepšila cévní Dice o 0,00367
(95% párový interval [0,00203; 0,00564]) a conditional A/V o 0,01263.
Macro zlepšení má interval protínající nulu a na G1020 se disk/cup zhoršily.
Cévní Dice 0,76039 stále zaostává za původní sítí 0,77012. Tento kandidát
proto nenahrazuje výchozí model ani nedokládá zlepšení všech úloh a zdrojů.
[Párové výsledky obou rolí checkpointu](https://github.com/tomasvicar/DeepRetinaSegmentation/tree/experiments/leipzig-hpo/reports/vascular4-paired-val).

Skutečný návrh Optuna TPE z pěti srovnatelných pozorování E27 dokončil
na Adě všech 60 epoch. LAST E60 i BEST E58 prošly kontrolou všech 2 529
fotografií / 5 700 variant a samostatnou čerstvou VAL na 770 snímcích.
BEST má macro Dice 0,81090, vessel Dice 0,76288 a conditional A/V 0,82491.
Proti původní síti přetrvává cévní rozdíl −0,00724 (95% párový interval
[−0,01459; −0,00069]); nejde o zlepšení všech úloh ani konečný výběr.
Výsledek E27 byl samostatně zapsán do pomocné Optuna studie; výsledek E60
jej nenahrazuje. [Původ skutečného návrhu](https://github.com/tomasvicar/DeepRetinaSegmentation/tree/experiments/leipzig-hpo/reports/fixed27-proposal)
a [čerstvá měření dokončeného modelu](https://github.com/tomasvicar/DeepRetinaSegmentation/tree/experiments/tpe72-fresh-val-20261111/reports/tpe72-completed-fresh-val).

Dokončené V3/TPE běhy držely patch 384 a vstup raw25 pevné. Navazující
kontrolované porovnání zahrnuje patch 256/384/512 se stejným batchem 4
a dvě fotometrické varianty při patchi 384: CLAHE a korekci osvětlení s CLAHE.
Podpora uloženého předzpracování je nyní i v předatelné větvi; obě úplné
cache 3 299 TRAIN/VAL snímků jsou sestavené a ověřené. Patch256 dokončil
60 epoch a samostatnou čerstvou validaci BEST52 i LAST60 na všech 770 VAL
snímcích. BEST macro 0,80123, vessel 0,75944 a conditional A/V 0,80150
zaostávají za rodičovským TPE modelem; příslušné párové intervaly rozdílu jsou
záporné. Tento model nepřebíráme jako zlepšení. Další čtyři úlohy zůstaly při
kontrole 19:48 UTC aktivní; LC měla v 19:50 UTC dokončenou epochu 58/60.
[Úplné výsledky a omezení porovnání](https://github.com/tomasvicar/DeepRetinaSegmentation/tree/experiments/leipzig-hpo/reports/curated-raw256-fresh-val-20261111).
Stejný nativní pomocný TPE běh na Adě dosáhl 54/60 epoch; jeho dokončení
a čerstvá validace ještě nejsou prokázané.
[Dvě skutečné TRAIN kontroly kompatibility](../reports/product-photometry-real-train-parity)
daly při portu preprocessingu přesně stejné FP32 pravděpodobnosti a masky pro
dosavadní TPE preview i původní raw256 checkpoint; nejde o měření kvality.
Předzpracování se aplikuje na celý snímek před výřezem
a stejně při predikci, s původními pixely mimo FOV. Vyšší fyzické rozlišení
vyžaduje novou přípravu z RAW; zvětšení existující cache je nenahrazuje.
Přímá příprava 35 pixelů/stupeň už prošla malou kontrolou tří TRAIN fotografií
a sedmi anotovaných záznamů včetně aliasů, stereo výřezu a chybějící anotace.
Nejde o dokončenou přípravu ani trénování celého korpusu při tomto rozlišení.

Na Oldu skončilo samostatné měření vlivu velikosti a překryvu inference dlaždic
nad totožnými vahami vascular4 BEST E55 a všemi 770 VAL snímky. Vlastní
baseline 512/64 se porovnává s 512/192, 768/192 a 1024/256. Jde o ladění
predikce, nikoli o experiment s velikostí trénovacího patche. Největší dlaždice
zvýšila macro jen nepatrně a zhoršila cévy; výchozí 512/64 proto zůstává.
[Úplné výsledky](https://github.com/tomasvicar/DeepRetinaSegmentation/tree/experiments/leipzig-hpo/reports/vascular4-inference-tiling-val).

Další dokončené měření porovnalo žádné TTA, dvě zrcadlení, čtyři rotace a všech
osm transformací nad totožným vascular4 BEST E55 a všemi 770 VAL snímky.
Osm transformací zvýšilo macro Dice z 0,80597 na 0,81308
(párový rozdíl 0,00711; 95% interval [0,00497; 0,00942]) a conditional A/V
z 0,82288 na 0,83544. Celá invokace trvala 1 002 s proti 205 s bez TTA.
Na DRHAGIS se bodové A/V výsledky zhoršily; jde o průzkumné výsledky na VAL,
nikoli nezávislé potvrzení generalizace. TTA kód a výsledky zůstávají na
[samostatné větvi](https://github.com/tomasvicar/DeepRetinaSegmentation/tree/experiment/inference-tta-science/reports/inference-tta-vascular4);
nejsou součástí výchozí predikce tohoto vydání. TEST se pro ladění nepoužívá.

Volitelný BF16 TRAIN je již také v předatelné větvi. Na skutečné CUDA Adě
prošel krátký nezávislý krok s bitově shodnými vahami, stavem optimalizátoru,
loss, predikcí a CUDA RNG vůči připnutému experimentálnímu zdroji. Toto je
ověření portu, nikoli měření přínosu BF16 pro kvalitu plného modelu.
[Rozsah ověření](UNIFIED_TRAIN_BF16.md).

## Co zbývá k dokončení cíle

1. Dokončit a zkontrolovat skutečné trénovací běhy včetně pokrytí všech vhodných TRAIN dat.
2. Porovnávat další kandidáty párově po úlohách a zdrojích; kontrolovaně změřit patch, preprocessing a inference dlaždice a řešit zbývající cévní regresi.
3. Zmrazit vybraný checkpoint, konfiguraci, prahy a inference protokol; poté provést samostatný TEST.
4. Doložit kvalitu, velikost a rychlost vybraného modelu a předat jeho váhy vedle původních sítí.

**Použitelný kontrolní V2 checkpoint a samostatný vascular2 preview jsou přibalené; optimalizovaný finální model a TEST výsledky zatím nejsou vydané.**
Podrobný cíl je v [GOAL.md](GOAL.md), historické záznamy v
[PROGRESS_HISTORY_2026-11.md](PROGRESS_HISTORY_2026-11.md). Operátorské přístupy a
spotřeba MetaCentra jsou vedené v samostatném soukromém provozním repozitáři.
