# Cíl: původní reference i lepší jednotný model

Uživatel chce **obě použitelné varianty**: původní ověřené modely a jednodušší
jednotné řešení s doloženými výsledky. Finální předání je přijato: samostatný
`unified-raw512-selected` a původní modely jsou přibalené, instalace/predikce,
retraining, zmrazený TEST i měření velikosti a rychlosti jsou doložené.

Uživatel výslovně přijal malý cévní kompromis a uzavřel další optimalizaci:
dense TEST Dice je **0,776547691 proti původnímu 0,780767926**,
rozdíl **−0,004220235 (−0,422 procentního bodu)**. Tepny, žíly, disk a cup
se zlepšily; toto přijetí není důkazem superiority ve všech úlohách ani ekvivalence.
Původní modely zůstávají dostupné pro prioritu cévní kvality.
[Finální předání a matice důkazů](HANDOVER.md).

Aktuální výsledky a zbývající práci shrnuje [STATUS.md](STATUS.md). Starší časové
záznamy jsou zachované v [historii práce](PROGRESS_HISTORY_2026-11.md).

## Předatelná větev `for_Leipzig`

- Zachovat původní architektury a dostupné natrénované váhy pro cévy, A/V a disk/cup.
- Přenést odpovídající preprocessing, škálování, FOV, geometrii a postprocessing.
  Rodiny modelů mají odlišné vstupní kontrakty; nedoloženou historickou volbu
  nebo zbývající numerické rozdíly výslovně uvést.
- Ověřit shodu portu na identických vstupech. Historický textový výsledek
  nevydávat za nově změřený výkon na skutečných fotografiích.
- Nabídnout snadnou predikci i nové trénování, dokumentaci a instalaci přes `uv`.
  Aplikační kód je v Pythonu; MATLAB slouží pouze jako ověřovací reference.
- Předat kolegům přístup k repozitáři s použitelnými modely. Zdrojový ZIP nevytvářet.

## Doložený rozsah jednotného modelu a hranice tvrzení

- Jedna jednoduchá síť pro cévy, artery/vein a disk/cup. Zacházet správně
  s částečnými anotacemi: neoznačená úloha není negativní target. Body, AVR
  ani center markery nevydávat za husté segmentační masky.
- Použít všechna vhodná dostupná anotovaná **trénovací** data. Velká chybějící
  předučovací databáze není nutná. Doložit skutečné pokrytí fotografií a anotací;
  plánovaný počet epoch není doklad dokončení.
- Dopracovat a měřit augmentace, sampling, architekturu, optimalizátor/scheduler,
  hyperparametry a dostatečný trénovací rozpočet. Hodnotit také jednoduchost,
  počet parametrů, paměť a rychlost predikce.
- Výslovně porovnat velikost patche a fotometrické předzpracování: raw proti
  CLAHE a zdůvodněné korekci nerovnoměrného osvětlení. První V3/TPE studie
  měla patch384 a raw25 pevné. Navazující dokončené běhy již porovnaly
  patch256/384/512 a raw/CLAHE/korekci osvětlení; výsledky a rozdílné rozpočty
  jsou zaznamenané na výzkumné větvi. Starší
  pokus256/batch8 není izolovaným důkazem vlivu patche. Kontrolované porovnání
  musí držet batch a optimizer nastavení, uvést počet aktualizací i pixelový
  rozpočet/čas a zachovat shodný preprocessing při trénování i predikci.
  Fyzické rozlišení a velikost kontextu rozlišovat; interpolace již zmenšené
  cache nenahrazuje přípravu vyššího rozlišení přímo z RAW dat.
- Zachovat nezávislé TRAIN/VAL/TEST skupiny mezi úlohami, anotátory, známými
  pacienty a sdílenými fotografiemi. Neznámou pacientskou nezávislost nepředstírat.
  TEST nepoužívat pro ladění; vyhodnotit jej až po zmrazení výběru a protokolu.
- Zlepšení doložit vůči původním modelům při deklarované geometrii a stejné
  podpoře anotací. Uvést jednotlivé úlohy a zdroje i jejich omezení. Růst průměru
  způsobený disk/cup neprokazuje zlepšení cév. Přijaté předání výše tento
  konkrétní cévní kompromis uvádí; původní ambice zlepšit každou úlohu prokázána není.
- Ověřený nový model zpřístupnit vedle původních modelů s přesným původem vah,
  konfigurací, vstupním kontraktem a výsledky. Neoznačovat pilot z jedné databáze
  za finální náhradu originálů.

## Práce a data

`for_Leipzig` je použitelná varianta pro kolegy; `experiments/leipzig-hpo`
obsahuje výzkum a měření. Nové zásahy se nejprve ověřují na samostatných větvích.
Běžící experiment zachovává připnuté zdroje, data, konfiguraci a runtime.
Poznámky, důkazy a průběžné commity patří do repozitáře.

Originály v `ownCloud/fundus_databases` se pouze čtou. Kolegům lze odkazovat
na jejich vlastní kopii ownCloudu; zdrojová data se nepřepisují. Zálohy modelů
jsou uživatelem povolené. Používat subagenty a rozumně rozdělovat úlohy mezi
Ada, Pascal a old. Provozní přístupy, PBS skripty, skutečné úlohy a spotřeba
MetaCentra patří do samostatného soukromého repozitáře `metacentrum-workflows`.
