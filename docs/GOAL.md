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

## Aktuální stav při upřesnění cíle

Migrace 31 datových adaptérů je implementovaná a otestovaná. Dříve vydané tři malé
modely jsou pouze 30epochové piloty: každý byl učen na jediné databázi. **Nejsou
finálním novým řešením a nenahrazují původní modely.** Výrok „hotovo“ byl předčasný.

Původní váhy jsou dostupné v ownCloudu; tři subagenti právě portují univerzální
model, A/V model a jejich přesný preprocessing. Proběhlé pilotní/HPO výsledky
zůstávají zaznamenané, ale dalšímu vývoji musí předcházet použitelná původní reference.
Cíl bude splněn až po dokončení obou částí a doloženém porovnání.
