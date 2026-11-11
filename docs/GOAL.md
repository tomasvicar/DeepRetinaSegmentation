# Cíl: původní reference i lepší jednotný model

Uživatel chce **obě varianty**: původní ověřené modely a nové jednodušší jednotné
řešení s doloženým zlepšením. Samotný malý pilot, dobrý průměr nebo úspěšné testy
programu tento cíl nesplňují. **Cíl zůstává nedokončený.**

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

## Nový jednotný model na experimentální větvi

- Jedna jednoduchá síť pro cévy, artery/vein a disk/cup. Zacházet správně
  s částečnými anotacemi: neoznačená úloha není negativní target. Body, AVR
  ani center markery nevydávat za husté segmentační masky.
- Použít všechna vhodná dostupná anotovaná **trénovací** data. Velká chybějící
  předučovací databáze není nutná. Doložit skutečné pokrytí fotografií a anotací;
  plánovaný počet epoch není doklad dokončení.
- Dopracovat a měřit augmentace, sampling, architekturu, optimalizátor/scheduler,
  hyperparametry a dostatečný trénovací rozpočet. Hodnotit také jednoduchost,
  počet parametrů, paměť a rychlost predikce.
- Zachovat nezávislé TRAIN/VAL/TEST skupiny mezi úlohami, anotátory, známými
  pacienty a sdílenými fotografiemi. Neznámou pacientskou nezávislost nepředstírat.
  TEST nepoužívat pro ladění; vyhodnotit jej až po zmrazení výběru a protokolu.
- Zlepšení doložit vůči původním modelům při deklarované geometrii a stejné
  podpoře anotací. Uvést jednotlivé úlohy a zdroje i jejich omezení. Růst průměru
  způsobený disk/cup při horších cévách či A/V není splněním celého cíle.
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
