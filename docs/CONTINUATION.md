# Stav pro navázání po compactu

**Novější stav: [CONTINUATION_2026-11-10_2029.md](CONTINUATION_2026-11-10_2029.md).**
Vyhodnocovací kód a novější důkazy: [freeze v18:03UTC](handoffs/EVALUATION_FREEZE_20261110.md).
Níže je historický snímek ve 14:04 UTC, nikoli aktuální seznam procesů.

Ověřeno **10. 11. 2026 ve 14:04 UTC / 15:04 CET**. Je to časový snímek;
před další akcí znovu ověřit procesy a soubory. Compact se týká konverzace,
nikoli restartování tréninků. Celkový cíl zůstává aktivní a není dokončený.

## Cíl a podmínky dokončení

Úplné zadání je v [GOAL.md](GOAL.md). Zachovat oboje:

1. Předatelný `for_Leipzig`: původní architektury a váhy pro cévy, A/V a disk/cup,
   věrný preprocessing a jednoduchá dokumentovaná predikce v Pythonu, instalace uv.
2. Odděleně vyvinout skutečně lepší jednoduchý jednotný RGB model s pěti výstupy,
   trénovaný na všech vhodných trénovacích datech, s augmentacemi a HPO. Doložit
   jednotlivé úlohy, datasety, rychlost a velikost proti nejsilnější původní sestavě.

Test se nepoužívá pro ladění. Původní ownCloud data/modely zůstávají nezměněné;
nové modely lze zálohovat. Předání přes repo, bez zdrojového ZIPu. Pilotní modely
z jediné databáze nejsou finální řešení. Používat subagenty; Astra jen výjimečně.

## Co je předatelné

`for_Leipzig` je pushnutý po commit `5210a86`: pět původních checkpointů,
samostatně označené tři piloty, predikce a dokumentace. Aktuální větev prošla
374 testy. Nový kalibrovaný MATLAB profil běží celý v Pythonu:

```bash
uv sync --frozen
uv run deepretina predict --model legacy-av --av-raw-matlab \
  --input /path/to/fundus.jpg --fov /path/to/fov.png \
  --dataset drive --output predictions/av
```

U obecné dodané FOV se průměr odhadne věrně podle původního `souradnice`;
explicitní `--fov-diameter-pixels` má přednost. Speciální datasetové větve vyžadují
explicitní kalibraci. Není ještě převedený automatický MATLAB thresholding FOV.
Zbývající numerická rezidua, syntetický protipříklad i skutečné modelové testy
jsou v [MATLAB_AV_PROFILE.md](MATLAB_AV_PROFILE.md). MATLAB není runtime závislost.

## Co právě běží

| Host | Úloha | Proces | Poslední dokončená epocha / stav |
|---|---|---:|---|
| Pascal GTX1070 | unified baseline LR 0.001 |3910854|E7, macro Dice 0.673977|
| old GTX1060 | unified LR 0.0003 |287975|E7, macro Dice 0.706392|
| Ada RTX4070TiSUPER | unified silné augmentace |1041876|E4, macro Dice 0.584466|
| Ada CPU4, nice5 | původní disc0/1 × raw/normalized |1045309|plná validace běží|
| Ada CPU2, nice5 | build odvozené train+val cache |1053957|běží od 14:03 UTC; odhad 8.2 GB|

Procesy byly potvrzené přes `ps`, nejen podle záznamu v logu. GPU tréninky mají
screening do E8 a původní cosine horizon 60; nejsou to dokončené 60epochové modely.
Na Adě běží jen jeden náš GPU trénink. Při ověřování PID kontrolovat i příkaz,
aby se případně nerecyklované číslo nezaměnilo za náš proces.

MetaCentrum: dokončený osmý job má E8, nejlepší E7 macro0.671354, Exit0,
wall 1:09:40. Všech osm dokončených jobů má součet wall 1:51:39. Checkpointy a nezávislý milestone jsou ověřené; následný jediný autorizovaný job
byl právě odeslán (aktuální frontu znovu ověřit):
`legacy_topology_scratch`, zatím **jen E1**, limit 45 min, scheduler horizon 60.
Maximálně jeden náš GPU job queued/running. Účet, job ID, PBS skripty, piny,
tunely a aktuální ledger jsou výhradně v soukromém `metacentrum-workflows`;
po compactu přečíst jeho aktuální HANDOFF, neodvozovat stav fronty z tohoto snímku.

Dokončené a nyní neběžící: Ada GroupNorm E8 (best E6 macro 0.627943), width16
zatím pauza po E1; raw/normalized původní reference; nová kalibrovaná A/V reference.

## Data, zdroje a reprodukovatelnost

Korpus má 3820 fyzických fotografií a 7365 použitelných variant anotací ze 17 zdrojů:
2221 train /770 validation /829 test. G1020 používá známé pacienty z CSV,
ARIA sjednocené identity; RIGA/UoA jsou bez pacientské vazby pouze v tréninku.
Párové a shodné fotografie se nesmějí rozdělit napříč oddíly. Opaque identity
neprezentovat jako prokázanou pacientskou nezávislost.

Všechny dosavadní plné tréninky stále používají neměnný checkout
`DeepRetinaSegmentation_unified_run_87cf045`, vlastní frozen uv prostředí:

- source `87cf045ab89425c4994df577fd510d210a1a0eab`
- executable SHA256 `8172e06bfa2c61497d3a34d443027f2f0193834090e95530322e3a6f795254fe`
- manifest `529edba7743eb5b9175da429c2124d53f48c419b090a284b5650fd68f1fdcf69`
- canonical content `61032be218c623a910c0f3f5e2df3bfdc58e978fe494433a4e6f18d55e41693e`
- původní HPO plan `907713f6c4a94c7281e2a5e226777fb02a5cdf613b0d79c20e0fe14831cf5e14`
- supplemental GroupNorm plan `f161ab236ac42aa767cd3f6324a459fae3af2be5f3027a3b5a01b6efbdc903e4`

HPO pracovní větev se dále mění; nesmí se zaměnit s tímto execution checkoutem.
Cache ani změny loaderu nepřenášet potichu do běžících runů. Budoucí finální
trénink s novým kódem/cache dostane nový plán, source a cache fingerprint.

## Aktuální porovnání a význam skóre

Vyšší průměr nového modelu ještě není důkaz zlepšení všech úloh. E3/E5 zaostávaly
v cévách a A/V, zatímco disk/cup byly lepší. Před výběrem znovu porovnat aktuální
checkpointy po jednotlivých úlohách a zdrojích. Dosud nebyly otevřeny nové testové
výsledky pro ladění ani doložené finální zlepšení všech úloh.

Nová původní A/V reference už používá kalibrovaný MATLAB-style Python profil:
všech 42 A/V validačních fotografií, zachované ostatní predikce a shodné GT podpory.
Artery Dice 0.650863→0.689529, vein 0.664795→0.697298, conditional accuracy
0.710883→0.748573; celkový dataset-balanced průměr 0.610351→0.624398.
Používat tuto silnější referenci. Historická expozice a původní trénovací soubory
nejsou nezávisle zrekonstruované; srovnání zůstává takto kvalifikované.
Podrobnosti jsou na HPO větvi v `docs/ORIGINAL_AV_REFRESH.md` a
`reports/unified/original-calibrated-av-val.json`.

Úplná normalizovaná původní reference dopadla hůře než raw: macro 0.308036 vs
0.610351. To podporuje raw default, nikoli tvrzení o historickém souboru.
Výběr dedicated disc0/1 ještě čeká na dokončení právě běžící validace.

## Rozdělané větve a subagenti

- Hlavní experimenty: `/var/home/tom/code/DeepRetinaSegmentation_hpo`,
  `experiments/leipzig-hpo`; jsou zde centralizované vědecké reporty a kontroly.
- `/root/data`: `DeepRetinaSegmentation_derived_cache`,
  `experiments/unified-derived-cache`; přečíst `docs/HANDOFF_DERIVED_CACHE.md`.
  Implementace ef5413b, sample gate/capacity c141493, handoff e7a2e6e. Všech 240 testů
  prošlo; plná cache se teprve staví. Sample workers2 byl 1.433× rychlejší oproti
  už optimalizovanému loaderu; nejde o zrychlení celé GPU epochy.
- `/root/engine`: `DeepRetinaSegmentation_unified_cli`,
  `experiments/unified-public-cli`; přečíst `docs/handoffs/UNIFIED_CLI_WIP.md`.
  Větev je nyní pushnutá na `40f5237`, 505 testů a Ruff/format prošly.
  Public unified predict/train/val-evaluate čeká na root review a HPO merge.
  Není na shipping větvi; nepřidávat nový modelový alias bez finálního modelu.
- `/root/audit`: MetaCentrum ledger/QA a následný omezený legacy E1 job;
  soukromý `/var/home/tom/code/metacentrum-workflows`. Dedicated disc helper je
  v `DeepRetinaSegmentation_disc_reference`, `experiments/legacy-disc-reference`.
- `DeepRetinaSegmentation_loader_fast`, `experiments/unified-loader-fast` 15672d6:
  bezeztrátová optimalizace merge anotací, 227 testů a přesné párové vzorky;
  actual workers2 delivery 1.587× rychlejší oproti 87. Tréninky 87 ji nepoužívají.

## Nejbližší postup

1. Znovu ověřit běžící procesy/frontu a mailbox agentů; nevytvářet duplicitní runy.
   Při timeoutu pozorování znovu ověřit stejný proces, automaticky nerestartovat.
2. Dokončit a ověřit cache: canonical/cache FP, rozsah všech 2991 train+val fotografií
   a 6274 variant, velikost/čas a porovnání workerů; žádný zápis do ownCloudu.
3. Zachovat E8 milestones po doběhu screeningů. Dokončit zbývající původní kandidáty
   width8, width16 (pauza E1) a legacy topology; na Meta nejdřív E1 timing/backward
   gate, další resume až po kontrole skutečné doby a výsledku. Bez starého teacheru.
4. Zkontrolovat a přenést hotové experimentální unified CLI; 505 testů prošlo.
   Testové hodnocení potřebuje
   skutečný frozen selection artifact; nepovažovat libovolný soubor/hash za výběr.
5. Vybrat slibné konfigurace na validaci a dokončit plný 60epochový rozpočet;
   budoucí cache/loader kód použít v novém frozen plánu, bez oslabení source guardu.
6. Porovnat kvalitu jednotlivých úloh/datasetů, nejistotu, velikost a rychlost
   proti kalibrované nejsilnější původní sestavě; poté teprve zmrazit výběr a test.
7. Přenést ověřený nový model, inference/training API a dokumentaci do předatelné
   větve, zachovat původní modely. Cíl označit za dokončený až po všech bodech.
