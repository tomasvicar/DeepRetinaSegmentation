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
