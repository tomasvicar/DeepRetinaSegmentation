# Cíl práce: původní reference i nový lepší jednotný model

Výslovné zadání uživatele: **mít oboje** — původní ověřené modely a vedle nich
nové jednodušší jednotné řešení s prokazatelně lepšími výsledky. Přístup do
repozitáře a ke sdíleným datům v ownCloudu mají i kolegové. Zdrojový ZIP nevytvářet.

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
