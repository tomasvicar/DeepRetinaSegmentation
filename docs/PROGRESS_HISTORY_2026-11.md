# Historie práce — listopad 2026

Toto je zachovaný historický obsah původní stránky cíle. Časové údaje, běhy
a stavy níže patří k uvedenému okamžiku. Aktuální přehled je ve
[STATUS.md](STATUS.md); závazný cíl v [GOAL.md](GOAL.md).

---

# Cíl práce: původní reference i nový lepší jednotný model

Výslovné zadání uživatele: **mít oboje** — původní ověřené modely a vedle nich
nové jednodušší jednotné řešení s prokazatelně lepšími výsledky. Přístup do
repozitáře a ke sdíleným datům v ownCloudu mají i kolegové. Zdrojový ZIP nevytvářet.

## Aktuální stav: 11. 11. 2026, 00:55 UTC

`for_Leipzig` obsahuje původní modely, plný baseline V1 a nově i přípravu všech
21 trénovacích zdrojů přímo z RAW archivů v ownCloudu. Není nutný neveřejný
mezilehlý korpus. Kopírování dat je celé v Pythonu, s ověřenou nezávislostí
výsledku na rodičovských souborech. Nový `train-legacy` umožňuje čerstvě učit
původní architektury cév, disku, universal a A/V; jde o výslovně nové trénovací
recepty, nikoli o reprodukci nedoloženého historického optimalizátoru. Viz
[NATIVE_LEGACY_RETRAINING.md](NATIVE_LEGACY_RETRAINING.md) a
[UNIFIED_QUICKSTART.md](UNIFIED_QUICKSTART.md).

Integrační kontrola prošla 2 605 CPU testy, ověřením wheelu a zachováním všech
9 přibalených checkpointů. Oprava MATLAB nearest resize přesně odpovídá 24
syntetickým a 5 skutečným FOV referencím. Zbývající fotometrické rozdíly jsou
zdokumentované; úplnou shodu celého raw MATLAB řetězce netvrdíme.

Čerstvé měření původní sestavy na všech 770 VAL fotografiích přesně zopakovalo
všechny dřívější počty TP/FP/FN a podmíněné A/V. Průměr Dice přes datasety a
úlohy je 0,66703, pooled Dice cév 0,77012. Důkaz je na experimentální větvi v
`reports/frozen-original-val-reproduction/`. Po opravě nearest resize probíhá
samostatné měření s vlastní identitou zdrojů, které tento starší report nepřepisuje.

Kontrolovaná Optuna studie běží na třech vlastních GPU. Control prošel 27 epoch
z plánovaných 60; dvě varianty pokračují nebo se prověřují, jedna slabší byla
ukončena po třech epochách. Všech 2 529 fotografií a 21 zdrojů přispělo, ale
v poslední kontrole E31 zatím přispělo 5 697 z 5 700 anotovaných variant.
Dokončení proto vyžaduje skutečný plný rozpočet a kontrolu všech anotací, čerstvý
párový VAL, zmrazení výběru a teprve potom nezávislý TEST. **Nový finální V2
model ani jeho TEST výsledek zatím nejsou hotové. Celkový cíl zůstává aktivní.**

Následující časové aktualizace zachovávají historii práce.

## Aktuální stav: 10. 11. 2026, 22:50 UTC

Předatelná větev zachovává pět původních modelů, tři označené piloty a úplný
baseline **korpusu V1 se 17 zdroji**. Instalovaný balíček i skutečná Python
source AV predikce jsou ověřené; přetrvávající rozdíly proti MATLABu jsou
zdokumentované. Cíl obou variant a prokázaného zlepšení zůstává nedokončený.

Na experimentální větvi je už skutečně připravený neměnný **korpus V2**:
2 529 TRAIN fotografií / 5 700 anotovaných variant z 21 zdrojů, 770 VAL a
856 TEST fotografií. Přibyly DRIONS, GAMMA, nativní IDRiD disc a BIOMISA
s explicitní auditovanou exportní politikou. Původní VAL soubory jsou byte
přesně zachované; geometrické předpoklady jsou přiznané a uložené v politice.
Detailní důkazy jsou na `experiments/leipzig-hpo`, zejména
`docs/CONTINUATION_2026-11-10_2250.md`. OwnCloud originály se nemění.

V 22:47 UTC byla spuštěná nová kontrolovaná Optuna studie na V2: nově učený
control0 a šest přesně vymezených variant ztráty, geometrie a augmentací,
se skutečnými etapami 3/9/27/60. Spuštění není dokončený model ani důkaz
zlepšení; výsledky se budou dokládat skutečnými epochami a párovým VAL.
Starší V1 studie je oddělená a její výstupy zůstávají zachované. Žádný nový
model zatím nebyl skórován na TEST. Následující časové aktualizace jsou historie.

## 1. Původní modely jako použitelná reference

- Zachovat původní architektury a dostupné natrénované váhy pro cévy, A/V a disk/cup.
- Přenést odpovídající preprocessing, škálování vstupů, FOV, geometrii a postprocessing;
  jednotlivé rodiny modelů mají odlišné vstupní kontrakty.
- Ověřit numerickou shodu portu s původním kódem na identických vstupech. Historické
  textové skóre bez nového vyhodnocení nevydávat za právě reprodukovaný výsledek.
- Zajistit jednoduchou predikci, instalaci přes uv a dokumentaci. Předatelná větev
  `for_Leipzig` má obsahovat funkční původní modely, nikoli jen jejich náhrady.

## 2. Nový jednoduchý jednotný model

- Společné řešení pro cévy, artery/vein a disk/cup; cílově jedna síť s odpovídajícími
  výstupy a jasným zacházením s částečnými anotacemi.
- Trénovat na všech vhodných dostupných anotovaných **trénovacích** datech pro každou
  úlohu. Neoznačené úlohy nesmějí být negativním targetem. Body, AVR a center markery
  nevydávat za husté segmentační masky. Velká chybějící předučovací databáze není nutná.
- Dopracovat a měřit augmentace, sampling, architekturu, optimalizátor/scheduler,
  hyperparametry a dostatečný trénovací rozpočet. Posuzovat i jednoduchost, počet
  parametrů, paměť a rychlost predikce.
- Zachovat nezávislé validační a testovací skupiny napříč úlohami, anotátory,
  pacienty a známými sdílenými databázemi. Test nepoužívat pro ladění.
- Lepší výsledky **doložit** porovnáním s původními modely při deklarované stejné
  geometrii a podpoře anotací; uvést jednotlivé úlohy, datasety a omezení. Malý
  pilot nebo pouhý pokles trénovací loss nejsou splněním tohoto cíle.

## Větve, data a pořadí práce

- `for_Leipzig`: použitelná, dokumentovaná varianta pro kolegy, prioritně původní modely.
- `experiments/legacy-model-port`: izolovaná integrace a kontrola původních modelů
  před přenesením ověřené implementace do `for_Leipzig`.
- `experiments/leipzig-hpo`: nový model, augmentace a optimalizace; experimentální
  změny neprezentovat jako ověřené před dokončením porovnání.
- Data z `ownCloud/fundus_databases` číst; původní data/modely nepřepisovat.
  Zálohy nových modelů do ownCloudu jsou uživatelem povolené.
- Dostupné GPU: Ada, Pascal a old. Úlohy rozdělovat mezi subagenty a GPU.

## Rozsah korpusu: aktualizace 10. 11. 2026,21:02UTC

Širší kontrola archivů našla dosud nezaregistrované DRIONS-DB (110 fotografií,
220 expertních obrysů disku) a GAMMA (100 trénovacích fundus fotografií/masek
disk/cup). Plný dokončený baseline i noční HPO jsou na **korpusu v1 se17 zdroji**,
nikoli na všech potenciálně použitelných anotovaných archivech. Tato hranice
je nyní explicitně zaznamenaná i v katalogu modelů. Adaptéry, geometrie,
patient/photo kontrola a korpusv2 se připravují odděleně; běžící zdroje/data se
nepřepisují. Cíl všech vhodných dostupných trénovacích dat proto vyžaduje také
dokončení této širší kontroly a následné doplnění finálního tréninku.

## Aktualizace 10. 11. 2026, večer

`for_Leipzig` nyní obsahuje vedle pěti původních modelů a tří označených pilotů
použitelný `unified-full-data-baseline`. Trénink dokončil60 epoch a doloženě
použil všech2 221 vhodných TRAIN fotografií i všech5 281 variant anotací ze17
zdrojů. Váhy E56 byly vybrané validací; TEST se zatím neotevřel. Přibalené váhy,
celé CSV a E60 kontrola jsou v `reports/release/control60-baseline/`.
Po začlenění explicitní Python source FOV/AV cesty prošlo všech2 440 CPU testů.
Původní profily/defaulty zůstávají a zdokumentované fotometrické rozdíly vůči
MATLABu se nevydávají za úplnou shodu.

Následuje Optuna na experimentální větvi: nejvýše48 nových pokusů, rungs3/9/27/60,
pevná validační volba prahu0.5 a cílem jsou kvalitnější jednodušší modely.
Povoleno je až5 Meta GPU a3 vlastní GPU přes noc; provozní přístupy, skutečné
joby a jejich spotřeba se zapisují pouze do soukromého provozního repozitáře.
Každý kandidát při pokračování zachovává konfiguraci/GPU/cache a plán60epochového
scheduleru. Plné nezávislé porovnání kvality, rychlosti/velikosti a výběr finálního
nového modelu ještě zbývají. **Celkový cíl proto není dokončený.**

Níže je zachovaný dřívější průběžný stav, který tato aktualizace doplňuje.

## Stav k 10. 11. 2026

Na `for_Leipzig` jsou nyní přibalené čtyři původní single_net checkpointy a
původní A/V v9 model. Převod zachovává váhy i architektury; připnuté referenční
implementace dávají na zaznamenaných shodných připravených vstupech stejné logity.
Při prvním předání prošlo 119 testů a kontrola balíčku s osmi checkpointy (pět původních a tři pilotní).

Historická volba raw/normalizovaného vstupu single_net není průkazně doložená.
Raw A/V Python preprocessing je výslovně označený jako aproximace MATLABu;
nativní připravený A/V vstup zachovává původní numerický kontrakt. Tyto nejistoty
nejsou považovány za nově ověřenou kvalitu na skutečných fotografiích.

Migrace 31 datových adaptérů je implementovaná. Dříve vydané malé modely jsou
pouze 30epochové piloty z jednotlivých databází, nikoli finální náhrada originálů.
Plný korpus prošel kontrolou: 3820 fotografií, 7365 použitelných variant anotací
ze 17 zdrojů; 2221 fotografií train, 770 validation a 829 test. G1020 používá
CSV pacientské skupiny, ARIA sjednocuje prefixy a RIGA/UoA jsou kvůli chybějící
pacientské vazbě pouze v tréninku. První plné epochy již proběhly a skutečně
použily všech 2221 trénovacích fotografií. Pracuje se na validačním porovnání
learning rate, šířky sítě a normalizace; původní modely se zároveň vyhodnocují
na stejném validačním korpusu. Testovací skóre nových modelů se nepoužilo pro
ladění. Úplný trénovací rozpočet, optimalizace a konečné porovnání ještě hotové
nejsou; první epocha není finální předatelný model.

MATLAB R2026b na Adě nyní slouží jako numerická reference pro převod původního
A/V preprocessingu. Samostatný Python CLAHE krok odpovídá 80 syntetickým testům
a oběma skutečným V rovinám DRIVE/HRF do maximální absolutní chyby 4,45e-16.
Vývoj je na `experiments/matlab-clahe-parity`; celý raw řetězec se ještě ověřuje
a jeho kalibrovaná varianta je nyní dostupná přes explicitní `--av-raw-matlab`.
Vyžaduje původní FOV; pro obecnou větev se střed/poloměr odhadne přesně podle
ověřeného `souradnice`, ručně zadaný průměr má přednost. Speciální datasetové
větve a degenerované masky bez kalibrace skončí chybou před spuštěním sítě.
Hrany, středy a poloměry souhlasí s MATLABem na DRIVE/HRF i 73 syntetických
maskách; metadata rozlišují automatický odhad a explicitní hodnotu.
Na DRIVE/HRF daly skutečné A/V váhy se
stejnými ručními cévními maskami stejné výsledné třídy uvnitř cév. Zbývající
numerická rezidua a syntetický protipříklad jsou v `MATLAB_AV_PROFILE.md`;
obecnou bitovou shodu netvrdíme. Navazující varianta s kalibrací dodané FOV prošla 374 testy.
MATLAB není závislostí výsledného balíčku.

Na `experiments/unified-loader-fast` je oddělená optimalizace načítání anotací.
663 párových vzorků zachovalo shodné tensory, anotátory a stavy RNG; 227 testů
prošlo. Potvrzovací CPU měření snížilo čas přípravy vzorku přibližně o 30 %.
Rychlost celé GPU epochy tím ještě doložená není. Běžící tréninky nadále používají
neměnný kód `87cf045`, původní plány a stejný ověřený datový korpus.
Následný skutečný DataLoader se dvěma workery zachoval dalších 1496 párových
vzorků včetně RNG a dodával shodná data 1,59× rychleji. Jde o CPU dodávku dat.

Párové validační porovnání E3 zahrnuje 770 fotografií a 612 známých skupin
pacientů či fotografií. Nový model má lepší celkový průměr díky disku/cupu,
ale cévy a A/V jsou zatím horší než u původní sestavy. Zlepšení průměru proto
nepovažujeme za splněný cíl všech úloh; podrobnosti jsou na experimentální
větvi v `reports/unified/paired-epoch3/`.

Provozní MetaCentrum poznámky, PBS skripty a evidence testů jsou vedené samostatně
v soukromém repozitáři `metacentrum-workflows`. Cíl zůstává aktivní, dokud nejsou
dokončené obě části a doložené jejich porovnání.


---

## Archivovaný výzkumný přehled ze zdroje `5545f3d`

Následující oddíl byl přesunut beze změny z původního `STATUS.md`, jehož titulní
snímek uváděl 12. 11. 2026 v 02:55 UTC. Obsahoval i starší listopadové průběžné
stavy a pozdější doplňky. Přítomný čas a označení běžících úloh uvnitř jsou
historická, nikoli současné pozorování. Aktuální stav je v [STATUS.md](STATUS.md).

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

*Konec archivovaného oddílu; nejnovější časový snímek je v [STATUS.md](STATUS.md).*
