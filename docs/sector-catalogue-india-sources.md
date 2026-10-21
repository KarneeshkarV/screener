# India sector catalogue evidence

## Scope and observation date

Review date: 2026-10-20, from the local `date -I` command.
The catalogue contains 106 groups and 320 distinct NSE symbols.
The change preserves all 65 inherited group IDs, names, parents, kinds, member lists and video timestamps.
It adds 41 independently curated groups containing 137 distinct symbols, of which 125 are additional to the original catalogue.
All 106 groups now have descriptions, driver lists, risk lists and primary-source URL lists.

The inherited `observed_on: 2026-10-20` is unchanged.
The original catalogue already used this date for its video-label and symbol observations.
This review does not repeat or certify the original video check.
The new review also occurred on 2026-10-20, but it checks exchange files and selected issuer pages, not video content.
There is no point-in-time constituent history behind this date.

The top-level source remains [the original video](https://www.youtube.com/watch?v=d-I6BG5F-Yg).
That link supports inherited label provenance only, not the selected constituents or the new descriptions.
None of the 41 added groups has a `video_at` field or claims video provenance.
Per-group `sources` give the new issuer and exchange provenance.

## Membership and return limits

These are selected research baskets, not official indices, exhaustive industries or recommendations.
The added groups separate useful economic exposures rather than split one list into arbitrary size bands.
They cover banks, non-bank finance, insurance, fund management, IT services, engineering and software, vehicles and components, telecom, utilities, metals, property, consumer businesses, energy, ports and healthcare.
Each added group has three to seven common-stock members.
Real-estate baskets contain developer or landlord company shares, not REIT units.

Membership uses companies present in the reviewed NSE files.
Historical basket returns therefore have survivorship and selection bias.
The files do not establish that a company was listed throughout a requested backtest period.
They also do not prove current trading activity, an absence of suspension, order-book depth or a minimum daily turnover.
No full live liquidity, corporate-action, financial-report or sector-classification audit was done.
Do not read the review date as a guarantee that a symbol remains valid later.

The analysis uses adjusted stock returns, which include distributions, against the Nifty 500 price index, which does not.
That difference can overstate basket performance relative to a total-return comparison.
Diversified companies and repeated constituents create mixed and overlapping exposures.

## Exchange symbol checks

Both exchange files below returned HTTP 200.
The review parsed their symbol, company-name, series and ISIN columns and matched every selected symbol against their combined symbol set.
All 320 matched: 317 in the main-board file and three in the SME file.
The original 195 distinct symbols all matched too.
This is a symbol-identity check, not proof of the detailed classification assigned to every member.

| File | Rows | SHA-256 of downloaded text |
| --- | ---: | --- |
| [NSE equity securities](https://archives.nseindia.com/content/equities/EQUITY_L.csv) | 2,619 | `64bc3997cc8e6d67f77fec2d04e4f74c45d7f7cb712ffd55c0885fb5a56722d8` |
| [NSE SME securities](https://archives.nseindia.com/emerge/corporates/content/SME_EQUITY_L.csv) | 581 | `6b9b3c1af3290d9abb8dae73fed1141b78923de9162f888dd59f29d7ea33bfc1` |

The SME matches are `RAPPID`, Rappid Valves India; `BASILIC`, Basilic Fly Studio; and `DIGIKORE`, Digikore Studios.
They remain in their inherited groups, with explicit SME exposure notes.
The small existing tea, valves, VFX and single-stock cryogenics baskets must not be treated as liquid diversified sectors.

`HFCL` is in the BE trade-to-trade series in the reviewed main-board file.
The telecom-equipment group retains it as a relevant common stock and explicitly notes that restriction.
All other added-group members are in EQ series in that file.
Neither series designation guarantees liquidity.

The checks rejected proposed stale or incorrect symbols before writing the catalogue:

| Proposed symbol | Evidence-based selection |
| --- | --- |
| `LTIM` | `LTM`, LTM Limited, ISIN `INE214T01019`; also present in the reviewed IT index file |
| `TATAMOTORS` | `TMPV`, Tata Motors Passenger Vehicles Limited, and `TMCV`, Tata Motors Limited; the exchange lists them separately |
| `OBEROIRLT` | `OBEROIRLTY`, Oberoi Realty Limited |
| `BLUESTAR` | `BLUESTARCO`, Blue Star Limited |
| `GUJGASLTD` | Excluded because it was absent from the reviewed files; no replacement or corporate-action history was assumed |

The files establish the current company names above.
They do not by themselves verify the legal history, timing or price adjustment of each corporate action.

## Exchange classification checks

The following archived NSE constituent files returned HTTP 200 and parsed as CSV.
The review compared their `Symbol` and `Industry` columns with the catalogue.
Their broad industry labels support sector-level context, not each fine-grained business distinction.
The baskets are not copies of these indices and can include non-constituents.
Index inclusion is not used as a claim about index weights or historical membership.

| NSE archived file | Rows | Distinct catalogue symbols also in file |
| --- | ---: | ---: |
| [Bank](https://archives.nseindia.com/content/indices/ind_niftybanklist.csv) | 14 | 11 |
| [PSU bank](https://archives.nseindia.com/content/indices/ind_niftypsubanklist.csv) | 12 | 5 |
| [Financial services](https://archives.nseindia.com/content/indices/ind_niftyfinancelist.csv) | 20 | 16 |
| [IT](https://archives.nseindia.com/content/indices/ind_niftyitlist.csv) | 10 | 8 |
| [Auto](https://archives.nseindia.com/content/indices/ind_niftyautolist.csv) | 15 | 13 |
| [Realty](https://archives.nseindia.com/content/indices/ind_niftyrealtylist.csv) | 10 | 7 |
| [Metal](https://archives.nseindia.com/content/indices/ind_niftymetallist.csv) | 15 | 9 |
| [FMCG](https://archives.nseindia.com/content/indices/ind_niftyfmcglist.csv) | 15 | 13 |
| [Consumer durables](https://archives.nseindia.com/content/indices/ind_niftyconsumerdurableslist.csv) | 14 | 9 |
| [Oil and gas](https://archives.nseindia.com/content/indices/ind_niftyoilgaslist.csv) | 15 | 10 |
| [Energy](https://archives.nseindia.com/content/indices/ind_niftyenergylist.csv) | 40 | 25 |
| [Pharma](https://archives.nseindia.com/content/indices/ind_niftypharmalist.csv) | 20 | 13 |
| [Healthcare](https://archives.nseindia.com/content/indices/ind_niftyhealthcarelist.csv) | 20 | 14 |
| [Commodities](https://archives.nseindia.com/content/indices/ind_niftycommoditieslist.csv) | 30 | 24 |
| [Nifty 500](https://archives.nseindia.com/content/indices/ind_nifty500list.csv) | 500 | 237 |

The Bank overlap counts every catalogue group, including the small-finance basket.
The Nifty 500 check supplies broad industry labels for 237 of 320 selected symbols.
It does not verify a fine-grained product classification for those 237.
The remaining 83 have exchange symbol evidence but no classification evidence from that particular file.
The reviewed [Midcap 150](https://archives.nseindia.com/content/indices/ind_niftymidcap150list.csv), [Smallcap 250](https://archives.nseindia.com/content/indices/ind_niftysmallcap250list.csv) and [Media](https://archives.nseindia.com/content/indices/ind_niftymedialist.csv) files also parsed successfully.
No separate membership claim depends on these three files.

## Issuer content checks

The review attempted bounded HTTP GETs for every URL retained in the per-group `sources` lists.
It checked status, redirects, page title and extracted text where available.
It read representative business text across the returned pages, but did not inspect every issuer annual report, revenue segment or member-to-group classification.
Driver and risk lists are research summaries, not issuer forecasts or quoted investment guidance.
A source can support only part of a group's mixed exposure.

Examples of readable issuer evidence follow.
These checks support the distinctions in the descriptions without claiming a complete membership audit.

| Issuer source | Business evidence read |
| --- | --- |
| [Divis](https://www.divislabs.com/) | API manufacturing and CDMO activities |
| [Syngene](https://www.syngeneintl.com/) and [Sai Life](https://www.sailife.com/) | Contract research, development and manufacturing |
| [Laurus](https://www.lauruslabs.com/) | Pharmaceutical ingredients, finished dosages and newer biotechnology activities |
| [LMW](https://www.lmwglobal.com/) | Textile machinery, machine tools, foundry and advanced technology divisions |
| [Thermax](https://www.thermaxglobal.com/) | Heating, process heat, steam, cooling, pollution-control and water systems |
| [Atam](https://www.atamvalves.in/) | Valves, fittings and boiler mountings |
| [KRN](https://krnheatexchanger.com/) | Heat exchangers and cooling equipment |
| [Cera](https://www.cera-india.com/) | Sanitaryware, faucets, showers and tiles |
| [Fine Organics](https://www.fineorganics.com/) | Specialty additives for food, plastics and other end markets |
| [Gujarat Fluorochemicals](https://www.gfl.co.in/) | Fluorine chemistry and fluoropolymers |
| [Avanti](https://www.avantifeeds.com/) | Feed, farms, hatchery and frozen shrimp divisions |
| [JK Paper](https://www.jkpaper.com/) | Writing and printing papers and packaging boards |
| [HDFC Bank](https://www.hdfcbank.com/) and [ICICI Bank](https://www.icicibank.com/) | Deposit, account and loan products; redirects led to official `bank.in` sites |
| [AU](https://www.aubank.in/) | Bank identity, savings accounts and term deposits; no claim that its small-finance regulatory status was independently audited |
| [LIC Housing](https://www.lichousing.com/) | Home, construction and plot loans |
| [Manappuram](https://www.manappuram.com/) | Gold lending and other loan products |
| [HDFC Life](https://www.hdfclife.com/) | Life insurance product categories |
| [LTTS](https://www.ltts.com/) | Engineering research and development services |
| [Newgen](https://www.newgensoft.com/) | Enterprise workflow and software products |
| [Maruti](https://www.marutisuzuki.com/) | Passenger vehicle models and corporate identity |
| [Hero](https://www.heromotocorp.com/) | Motorcycle and scooter product categories |
| [Ashok Leyland](https://www.ashokleyland.com/) | Trucks, buses and light commercial vehicles |
| [Uno Minda](https://www.unominda.com/) | Components and systems supplied to vehicle manufacturers |
| [Tejas](https://www.tejasnetworks.com/) | Optical transport and broadband network equipment |
| [NTPC](https://www.ntpc.co.in/) | Integrated power generation and capacity development |
| [Power Grid](https://www.powergrid.in/en) | Electricity transmission business |
| [Hindustan Zinc](https://www.hzlindia.com/) | Zinc, lead, silver, mining and smelting |
| [DLF](https://www.dlf.in/) | Homes, retail, offices and hospitality properties |
| [HUL](https://www.hul.co.in/) | Personal care, cleaning and food businesses |
| [Voltas](https://www.voltas.com/) | Air conditioners and household appliances |
| [Kalyan](https://www.kalyanjewellers.net/) | Jewellery retail and official store identity |
| [Oil India](https://www.oil-india.com/) | Upstream production and pipeline activities |
| [Adani Ports](https://www.adaniports.com/) | Ports and integrated logistics |
| [Eternal](https://www.eternal.com/) | Zomato, Blinkit and other group businesses |

### Access limits

There are 121 distinct per-group source URLs after replacing the unresolved proposed `www.bajajfinance.in` link with the checked Bajaj Finserv site.
All 121 were attempted during this review.
Their recorded outcomes are 88 HTTP 200, 14 HTTP 403, eight HTTP 406, two HTTP 503, two HTTP 307 responses without usable business content, and seven connection or TLS errors.
HTTP 200 alone is not a successful classification check.
The readable-text review found JavaScript shells or near-empty pages at HAL, Airtel, CG Power, Bajaj Finserv and IGL.
Nippon India's fund site returned some readable identity and fund text but also a content-loading warning.

The following retained issuer links did not provide readable evidence in this run.
They remain issuer reference links, not claims of successful content verification.
Exchange sources still establish the corresponding symbol identities.

| Result | Retained URLs |
| --- | --- |
| HTTP 403 | <https://sunpharma.com/>, <https://www.apollohospitals.com/>, <https://www.dabur.com/>, <https://www.elecon.com/>, <https://www.godrejcp.com/>, <https://www.godrejproperties.com/>, <https://www.hdfcfund.com/>, <https://www.icicilombard.com/>, <https://www.ihcltata.com/>, <https://www.infosys.com/>, <https://www.muthootfinance.com/>, <https://www.nestle.in/>, <https://www.schaeffler.co.in/>, <https://www.tvsscs.com/> |
| HTTP 406 | <https://www.aarti-industries.com/>, <https://www.bluestarindia.com/>, <https://www.coromandel.biz/>, <https://www.hindalco.com/>, <https://www.interarchbuildings.com/>, <https://www.jswsteel.in/>, <https://www.lalpathlabs.com/>, <https://www.piindustries.com/> |
| HTTP 503 | <https://www.jayshreetea.com/>, <https://www.kirloskarpumps.com/> |
| HTTP 307 or challenge page | <https://www.indo-mim.com/>, <https://www.iocl.com/> |
| Read timeout | <https://new.abb.com/indian-subcontinent>, <https://www.hcltech.com/> |
| Certificate validation failure | <https://www.inoxcva.com/>, <https://www.sanghvicranes.com/>, <https://www.sudarshan.com/>, <https://www.thephoenixmills.com/> |
| TLS protocol failure | <https://www.pfcindia.com/> |

The first Inox candidate, `https://inoxindia.com/`, returned only an old company title with no useful business text.
The first INDO-MIM candidate, `https://www.indomim.com/`, returned an empty body.
Neither is retained as group evidence.
The final Inox CVA and INDO-MIM issuer links above were attempted but blocked as recorded.
No TLS verification bypass was used.

Other discovery attempts are not evidence of successful classification:

- `https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv`, `https://nsearchives.nseindia.com/emerge/corporates/content/SME_EQUITY_L.csv` and `https://nsearchives.nseindia.com/content/equities/sme/sme_eligible.csv` timed out.
- Seven `https://www.niftyindices.com/IndexConstituent/ind_nifty...list.csv` requests timed out for bank, IT, auto, finance, realty, metal and FMCG.
  The archived NSE equivalents above succeeded.
- `https://archives.nseindia.com/content/equities/sme/sme_eligible.csv` returned 404.
- Archived `ind_niftyprivatebanklist.csv`, `ind_niftyfinserviceexbanklist.csv` and `ind_niftyinfrastructurelist.csv` candidates returned 404.
  These are not retained sources.
- The TCS home page and Tata Steel home page returned 403.
  Alternate Infosys, Apollo, ICICI Lombard and Blue Star business-page requests also returned access blocks.
- `https://www.bajajfinserv.in/bajaj-finance` returned 404, `https://www.bajajfinance.in/` failed DNS resolution, and `https://www.pfcindia.co.in/` failed TLS negotiation.
- `https://www.dlf.in/businesses/offices` redirected to the DLF home page, so no separate office-page evidence is claimed.
- The Power Grid root redirected to an inaccessible HTTP endpoint.
  Its explicit HTTPS English page succeeded and is the retained source.
- The NSE main home page timed out.
  The SBI legacy URL led to a redirect page; the retained `https://sbi.bank.in/` page returned readable bank content.

## Mixed exposure decisions

The descriptions retain useful warnings instead of implying exact industry purity:

- CDMO, API, branded pharma and US generics overlap within diversified drug makers.
- The Ayurveda group consists of consumer businesses, including edible oils, not pure healthcare companies.
- Cranes includes rental; conveyors includes broader material handling; castings includes metal injection moulding.
- Kajaria is principally tiles; LMW includes textile machinery; HBL Engineering includes railway electronics.
- Apex is primarily shrimp processing and exports, so it is not a direct feed-manufacturer peer.
- Renewable generation and solar manufacturing have different revenue models within the inherited renewable group.
- PFC and REC overlap through ownership; Vedanta and Hindustan Zinc also overlap.
- Max Financial is a holding-company route to life insurance; GIC Re is reinsurance, not a direct health insurer.
- Eicher includes motorcycles and commercial vehicles; Mahindra includes tractors and other businesses.
- Tata Communications adds enterprise connectivity, not a third comparable domestic mobile operator.
- Property developers can own rental assets while retaining substantial development risk.
- Reliance includes telecom and retail; consumer and chemical conglomerates also carry non-sector businesses.

Fine-grained groups such as small finance banks, engineering R&D, general insurance and commercial property are curated exposure labels.
Readable representative issuer content or a broad exchange classification supports the exposure rationale, but this review did not independently verify every member's current revenue split or regulatory status.
The source lists are a starting point for further review, not a substitute for current filings.

## Offline regression checks

`tests/test_sector_catalogue_india.py` makes no network requests.
It checks all 65 original IDs and timestamps, required expansion areas, nonempty details and source lists, symbol syntax, duplicate members, three-to-seven-member added baskets and no added video timestamps.
A fixed set of the 320 reviewed symbols detects accidental unreviewed symbols.
That set is a dated regression fixture, not a live exchange feed.
Tests also retain the curated-membership, survivorship, mixed-exposure and adjusted-return warnings.
Future membership changes need fresh evidence and an explicit update to that fixture.

## 2026-10-21 additions

Review date: 2026-10-21.
This review adds seven groups and 26 symbols, so the catalogue has 113 groups and 346 distinct NSE symbols.
The themes come from SOIC's [Q2 FY27 results review](https://www.youtube.com/watch?v=RXjwpqeBgkE).
Only a Hindi auto-generated transcript was available; the review read it with the published chapter list.
That video is not the catalogue's top-level source, so the new groups have no `video_at` field.
The video discussed companies with strong quarterly results; it did not define these baskets.
Peers beyond the companies the video named are independent selections.

| Group | Members | Video chapter |
| --- | --- | --- |
| `wealth_management` | 360ONE, NUVAMA, ANANDRATHI | 360 ONE WAM and Angel One (5:38) |
| `music_media_content` | TIPSMUSIC, SAREGAMA, SUNTV | Tips Music (35:32) |
| `mining_equipment` | EIMCOELECO, BEML, RVTH | Eimco Elecon (39:13) |
| `midcap_it_services` | PERSISTENT, COFORGE, MPHASIS, BSOFT | AI and IT services (42:33) |
| `regional_private_banks` | SOUTHBANK, FEDERALBNK, KARURVYSYA, CUB, DCBBANK | Federal Bank (14:30), South Indian Bank (49:53) |
| `carbon_battery_materials` | HSCL, PCBL, GOACARBON | Himadri Speciality Chemical (48:07) |
| `luxury_consumption` | THELEELA, EIHOTEL, INDHOTEL, LOTUSDEV | Three ways to win in consumption (25:57) |

Existing groups received these members: `api` SOLARA, `chemicals_specialty` PAUSHAKLTD, `hotels` THELEELA, `steel` JAYNECOIND, `engineering_rd` ONWARDTEC, `residential_developers` LOTUSDEV and `nbfc_lending` SGFIN and FEDFINA.
These additions change inherited member lists but not inherited IDs, names, parents, kinds or timestamps.

The review downloaded the [NSE equity securities](https://archives.nseindia.com/content/equities/EQUITY_L.csv) file again.
It had 2,609 lines and SHA-256 `a6a09d6f397db028efcb4191a8aa5f0475a5f47697c070a27d48fd0795092810`.
All 26 new symbols matched.
All are in EQ series except `LOTUSDEV`, which is in BE trade-to-trade series; the `luxury_consumption` notes record this.
Aurum PropTech (`AURUM`) was the only listed proptech company in the video, so no proptech basket was added.
ASML and TSMC are already covered by the US semiconductor groups.

Bounded HTTP GETs returned 200 for every retained new issuer source.
The Eimco Elecon page describes mining and construction equipment.
The other pages were checked for status only, not for full business-segment content.
The South Indian Bank, Karur Vysya Bank, Zee and Goa Carbon sites were blocked or did not resolve, so they are not retained sources.
