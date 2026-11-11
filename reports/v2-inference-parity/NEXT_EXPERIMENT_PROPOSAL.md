# Další cílený V2 experiment — návrh, bez spuštění

**Doporučení:** po skutečném dokončení současného protokolu nejdřív jeden scratch
V2 běh se `vascular_sampling_weight=2`, ostatní parametry beze změny. AV observer
choice zůstane 0.5. Inicializaci z vlastního V1E56 ponechat jako druhý, samostatný
experiment. Nyní se nemění kód, data, live protokol ani přidělení GPU.

## Proč právě toto

[TRAIN audit, HPO06060d2](https://github.com/tomasvicar/DeepRetinaSegmentation/blob/06060d2/reports/data/sampling-exposure-v1-v2/README.md)
prokazuje změnu očekávané expozice před cropem, nikoli příčinu horší přesnosti.
Při 8192 vzorcích, coverage-first a stejném sampleru vzrostl počet source/task klíčů
29→35. Očekávaná vybraná vessel supervize klesla 4220→3426 a AV1134→905 za epochu;
disc/cup naopak rostou. V2 E25 vessel0.74383 / conditionalAV0.74044 nelze hodnotit jako
hotový výsledek proti V1 E56 z dokončeného 60epochového běhu (0.76082/0.79866).
V2 control je při zadání návrhu 36/60: nejdřív čekat na skutečná data a úplnou coverage.

Nové analytické TRAIN scénáře dodané datovým auditem:

| Váha vascular klíčů; AV choice | vessel | conditionalAV | disc | cup |
| --- | ---: | ---: | ---: | ---: |
| V2 control:1;0.5 |3426.40|905.16|6156.16|3871.28|
| V2:1.5;0.5 |3911.80|1069.65|5731.26|3479.12|
| **Navržený V2:2;0.5** |**4251.58**|**1184.79**|5433.83|3204.60|
| V1 reference:1;0.5 |4220.14|1134.27|5664.01|3514.93|

Váha2 přibližně obnovuje starou vessel/AV expozici bez změny observer choice.
Cena je předem známá: oproti V2control asi −11.7% disc a −17.2% cup příležitostí.
Váha1.5 je kompromis, ale ještě neobnovuje vascular úroveň V1. Žádná z těchto
hodnot nezaručuje Dice ani skutečnou post-crop kontribuci.

AVchoice0.8 by při nezměněném V2 sampleru zvýšila dostupné AV příležitosti zhruba
na 1448 místo 905, ale neobnovila by celkovou vessel expozici. Navíc mění volbu
AV/vessel GT observerů; AV bundle sice obsahuje konzistentní vlastní vessel target,
nejde však o tentýž observer mix. Proto tento druhý parametr teď nepřidávat.
Threshold framework není další priorita: existující V1 sweep0.35–0.60 měl nejlepší
vessel0.761762 při 0.40, stále pod original0.770123; conditionalAV zůstala0.798662.
Primární V2 porovnání ponechat při 0.5.

## A: nejmenší změna sampleru a požadovaný SOURCE

Pouze nový validovaný skalár `vascular_sampling_weight` (default1, experiment2)
v `UnifiedConfig` a ve výpočtu `UnifiedDataset.sampling_weights`: stávající příspěvek
`1/N[source,task]` násobit2 jen pro klíče vessel/av, pro disc/cup ponechat1. Normalizace
proběhne jako dosud při replacement draws. Povinný shuffled průchod všemi2529 TRAIN
fotografiemi se nemění; nemění se8192, architektura, loss, crop/augmentation, optimizer,
cosine60, seed42, VAL770, cache/policy ani observer výběr v`sample`. Žádné quotas,
nový batcher nebo další sampler framework. U jiných sampling režimů nejednoznačnou
kombinaci odmítnout.

Před případnou implementací/launch: samostatná implementační větev z experimentálního HPO `1d310a7`, commit a frozen SOURCE/execution/
training SHA, explicitní recipe/config SHA, nové run/plan/output identity a analytický
TRAIN scénář s pinned reportem. Default1 musí zachovat přesné staré váhy aRNG cestu;
testy mají ověřit key populations, coverage-first, nezměněný AV/DC observer bundle
výběr a finite/kladné váhy. Změna nesmí přesměrovat existující study.

Současný V2 pool completion adapter je záměrně připnutý k6b3 a původnímu protokolu.
Nová recipe potřebuje vlastní explicitně připnutou typed completion evidenci,
včetně všech5700 skutečně přispívajících TRAIN annotation IDs. Neobcházet ji
přeznačením starých receipts ani upravením gate běžící study. Pokud pro nové SOURCE
použijeme hotový control6b jako referenci, doložit default1 numerickou/RNG kompatibilitu
a označit rozdíl zdroje; malý/nejasný efekt vyžaduje párový nový control ve stejném SOURCE.

## B: vlastní V1E56 jako rodič — až jako samostatná otázka

Dnešní `legacy_initialization` je jen pro `legacy_universal` s neznámou historickou
expozicí. `--resume` správně odmítá jiný manifest/content/policy/cache/source/config;
není to cesta pro V1→V2. Vlastní compact parent vyžaduje nový explicitní init kontrakt:
parent best SHA
`bfe957460dfed168f2344b1b84ef99b7f502e9dbc8ce1fd4d68632c3234708b1`, jeho skutečné
E56/steps, completed-V1 QA a source3157, ověřenou scratch ancestry bez externích vah,
parent manifest/content a hash přesných TRAIN record/photo IDs. Dokázat byte-exact
parent5281 TRAIN řádků jako subset child5700 TRAIN; chránit VAL/TEST a známé aliasy.
Samotný název souboru nebo `historical_exposure=false` nejsou takový důkaz.

Kopírovat striktně pouze celý kompatibilní `state_dict`, včetně BN buffers, bez
head remappingu. Nový child má fresh optimizer/scheduler/RNG, epoch0, steps0,
coverage0, nový V2 manifest/cache/policy a celých60 epoch. Parent metadata zůstávají
samostatně v lineage. Současný selection guard kontroluje hlavně dva lineage flags;
nový typ musí validovat parent proof, ne jen nazvat warm start „scratch“.

Gradientová historie parentu je TRAIN-only, ale **E56 bylo vybráno na stejném VAL770**:
nejde o validačně neexponované váhy. Parent nese56epochovou gradientovou historii,
jeho zdrojový běh spotřeboval60 epoch; child60 není compute-matched scratch60.
Měřit tento transfer jako praktické znovupoužití vlastních vah, ne jako důkaz
lepší architektury nebo vyšší efektivity při stejné celkové expozici.

## Co skutečně změřit a jak rozhodnout

Po dokončení nynějších výsledků zmrazit jednu kandidátní recipe před GPU během;
neprunovat ji poE3 jen podle cup-heavy macro. V novém plném60epochovém běhu uchovat
skutečné post-crop head/source/observer příspěvky, knownAV pixels, coverage všech
TRAIN IDs, optimizer steps a walltime vedle analytických očekávání. Fresh paired
VAL770 při 0.5 a stejném inference protokolu: všech5 head Dice, vessel, conditional
/end-to-endAV, disc/cup, per-source výsledky a grouped paired intervaly. Pro V1/V2
použít existující byte-verified VAL bridge, ne přeznačení manifestu.

Cup/macro růst s horším vessel neboAV není splnění cíle. Přínos musí zasáhnout
vascular úlohy bez jasné disc/cup regrese; tvrzení „lepší všechny úlohy“ vyžaduje
podporu po jednotlivých úlohách a zdrojích, ne jen součet. B případně porovnávat až
při stejném zvoleném V2 samplingu jako A/scratch, aby se nemíchaly dvě změny.
TEST zůstává uzavřený do samostatného reviewed final-selection postupu.
