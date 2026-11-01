# US sector catalogue sources

Checked on 2026-10-20, using the machine date from `date -I`.
The catalogue uses this date for `observed_on`.

## Scope and limits

`screener/sector_rotation/sector_groups_us.yaml` contains 46 curated research baskets: 11 broad sectors, 32 industry baskets and 3 cross-sector themes.
It contains 218 member placements and 152 distinct common or ordinary share symbols.
Every basket has 4 to 6 members, a description, concrete drivers, concrete risks and at least one primary issuer or official classification source.
There are no ETFs, depositary receipts, partnership units or preferred shares in the selected baskets.
Equity REIT common shares and foreign-domiciled ordinary shares are included and labeled in the catalogue note.

The baskets are analyst selections, not official GICS classifications of individual issuers, official S&P 500 constituents or exhaustive industry membership lists.
The `industry` field is an application grouping level, not a claim that every basket matches one formal GICS industry.
Mixed baskets identify the differences in their descriptions or notes.
Cybersecurity is a selected software subset; rail includes an equipment supplier; digital infrastructure combines data-center and tower landlords.
Only one Alphabet share class is selected.
The bundled baskets avoid share-class punctuation, while loader tests cover provider-compatible dash forms such as `NYSE:BRK-B`.
The existing `tv_to_yf` function does not translate `BRK.B` or `BRK_B` to `BRK-B`, so the catalogue loader rejects those ambiguous forms.

Drivers and risks are qualitative research questions, not forecasts, reported issuer guidance or quantified findings from these pages.
Issuer sources support business scope, not every proposed member or each risk statement.
The official classification document supports the sector vocabulary and industry boundaries, not membership in these research baskets.
Listing-directory checks support the selected symbols and exchanges on the checked date only.
No live price history, corporate-action adjustment, historical membership or future listing status was verified.

Current members create survivorship and selection bias when used for historical charts.
The benchmark is `^GSPC`, labeled **S&P 500 price index**.
Adjusted stock returns can include dividends and other distributions, while this price benchmark excludes distributions.
REIT distributions make this mismatch especially relevant.
Equal-weight research baskets also differ from the benchmark's capitalization weights, company sizes and membership.
The S&P pages below blocked access, so this check did not independently revalidate the current index methodology or the provider's `^GSPC` mapping.

## Method

Fetched primary URLs with Python `requests`, redirects enabled, a browser user-agent and 25 to 40 second request timeouts.
Read the returned page text or extracted PDF text rather than treating every HTTP 200 response as successful evidence.
MSCI challenge pages and a Lam error page returned HTTP 200 but did not supply the requested evidence.
The March 2023 MSCI PDF returned a real PDF and was read with `pdftotext -layout`.
Its overview lists all 11 sectors; its structure and sub-industry sections cover the industry vocabulary used here.
This is a dated classification reference, not a claim that the 2023 document is the latest methodology in 2026.

### Listing checks

| Checked URL | Result and scope |
| --- | --- |
| https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt | HTTP 200; parsed symbol, security name, ETF and test-issue fields for Nasdaq listings. |
| https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt | HTTP 200; parsed ACT symbol, security name, exchange, ETF and test-issue fields for other listings. |
| https://www.nasdaqtrader.com/trader.aspx?id=symboldirdefs | HTTP 200; readable field definitions, including exchange codes and ETF flags. |

Both downloaded directory files ended with `File Creation Time: 1020202606:00`.
All 152 selected symbols matched their exchange prefix, had `ETF=N` and had `Test Issue=N`.
The check found no missing selected symbols or exchange mismatches.
Security names identify common shares, ordinary shares, or the equivalent capital stock.
Current directory entries require `NASDAQ:FISV`, `NYSE:XYZ`, `NASDAQ:WMT`, `NASDAQ:KMB` and `NASDAQ:LIN`; older ticker or exchange assumptions would be wrong for these files.
No basket uses the former `FI` or `SQ` symbols.
The symbol files are live and will change after this date.

SHA-256 hashes of the downloaded bytes:

```text
nasdaqlisted.txt  96f311d39d7b26516f70a088af6e03f7b68edcc04cccbd29276233a61b88fa9e
otherlisted.txt   07f1decb2250dd050fbf18c13dfecbd16652ba9ee46e5dfa72c9ea837fe9144a
GICS PDF          2a949a81b7e92f7005685160383d55ca31e5e8b906c3a868992e10aa77c0491f
```

### Sources used in the catalogue

All rows marked readable returned HTTP 200 with relevant content.
Redirect destinations are shown where the requested URL differed.
The source arrays in the YAML identify which baskets use each page.

| Checked URL | Result and evidence scope |
| --- | --- |
| https://www.msci.com/documents/1296102/11185224/GICS+Methodology+2023.pdf | Readable PDF, March 2023; 11-sector structure and sub-industry definitions, including payment processing within Financials and equity REITs within Real Estate. |
| https://www.nvidia.com/en-us/about-nvidia/ | Readable; accelerated computing and AI business scope. |
| https://www.jpmorganchase.com/about | Readable; consumer banking, commercial and investment banking, payments and asset management. |
| https://www.merck.com/company-overview/ | Readable; medicines, vaccines and research business. |
| https://www.lockheedmartin.com/en-us/who-we-are.html | Readable; aerospace and defense systems. |
| https://www.marriott.com/marriott/aboutmarriott.mi | Readable; hotel brands and travel services. |
| https://us.pg.com/who-we-are/ | Readable; household and personal product brands. |
| https://www.eogresources.com/ | HTTP 200, but returned page text gave limited business detail; listing symbol visible, upstream scope supported by classification vocabulary and research selection rather than a complete issuer review. |
| https://www.kindermorgan.com/ | Readable; energy infrastructure and natural gas pipelines. |
| https://nucor.com/ | Readable; steel manufacturing and recycling. |
| https://www.nexteraenergy.com/about-us.html | Readable redirect from https://www.nexteraenergy.com/company.html; utility history, power assets and energy business scope. |
| https://thewaltdisneycompany.com/about/ | Readable; entertainment businesses and brands. |
| https://investor.equinix.com/ | Readable; digital infrastructure, interconnection and `Nasdaq: EQIX` issuer identification. |
| https://www.realtyincome.com/who-we-are | Readable; real estate ownership and common stock distributions; some numeric fields remained client-side placeholders and were not used. |
| https://www.kla.com/company | Readable; semiconductor process control and yield management. |
| https://www.salesforce.com/company/ | Readable redirect to https://www.salesforce.com/company/our-story/?bc=OTH; CRM and enterprise software scope; YAML uses the clean story URL. |
| https://www.salesforce.com/company/our-story/ | HTTP 200 redirect to the same story URL with `?bc=OTH`; readable CRM business page. |
| https://www.paloaltonetworks.com/about-us | Initial request timed out after 35 seconds; retry timed out after 25 seconds; no issuer content claim. Cybersecurity also cites the readable GICS software reference. |
| https://www.progressive.com/about/ | Readable; vehicle, home and business insurance. |
| https://www.fiserv.com/en/about-fiserv.html | Readable; merchant payments, commerce and financial institution technology. |
| https://www.amgen.com/about | Readable; biotechnology and commercial medicines. |
| https://www.abbott.com/about-abbott.html | Readable redirect to https://www.abbott.com/en-us/about-abbott; diagnostics and medical technology scope. |
| https://www.danaher.com/our-businesses | Readable; biotechnology, diagnostics and life sciences businesses. |
| https://www.up.com/aboutup/index.htm | Readable redirect to https://www.up.com/about-us; freight railroad business. |
| https://www.wm.com/us/en/inside-wm | Readable redirect to https://www.wm.com/us/en/inside-wm/who-we-are; waste management services. |
| https://www.gm.com/company/about-us | Readable; vehicle brands and manufacturing business. |
| https://investors.lennar.com/ | HTTP 200; investor landing page and share listings; returned text did not provide a full homebuilding business description. |
| https://www.coca-colacompany.com/about-us | Readable; non-alcoholic beverages and distribution system. |
| https://corporate.walmart.com/about | Readable; retail operations and membership warehouse business. |
| https://www.slb.com/about | Readable redirect to https://www.slb.com/about/who-we-are; drilling, reservoir and energy technology scope. |
| https://www.verizon.com/about/our-company | Readable redirect to https://www.verizon.com/about; networks and communications business. |
| https://www.stagindustrial.com/ | Readable; industrial property acquisition, development, ownership and operation. |
| https://www.firstsolar.com/en/About-Us/Overview | Readable; thin-film CdTe photovoltaic modules and utility-scale solar supply. |

Four baskets use only the official classification reference: heavy machinery, electrical automation, chemicals and gases, and regulated electric utilities.
This is adequate evidence for the category definitions, not independent issuer-by-issuer classification confirmation.

### Other checks that did not establish the requested facts

These URLs were attempted during source selection but are not presented as successful evidence.
A failed check is not evidence that the issuer or document does not exist.

| Checked URL | Observed result |
| --- | --- |
| https://www.msci.com/our-solutions/indexes/gics | HTTP 200 with a reCAPTCHA challenge, not classification content. |
| https://www.msci.com/indexes/index-resources/gics | HTTP 200 with a challenge page. |
| https://www.msci.com/indexes/index-resources/gics/gics-sector-industry-definitions | HTTP 200 with a challenge page. |
| https://www.msci.com/documents/10199/0/Global-Industry-Classification-Standard-GICS-Methodology.pdf | HTTP 404. |
| https://www.msci.com/documents/1296102/11185224/GICS+Sector+definitions.pdf | HTTP 404. |
| https://www.spglobal.com/spdji/en/indices/equity/sp-500/ | HTTP 403. |
| https://www.spglobal.com/spdji/en/documents/methodologies/methodology-sp-us-indices.pdf | HTTP 403. |
| https://www.spglobal.com/spdji/en/documents/methodologies/methodology-gics.pdf | HTTP 403. |
| https://www.spglobal.com/spdji/en/documents/factsheets/fs-sp-500.pdf | HTTP 403. |
| https://www.spglobal.com/spdji/en/landing/investment-themes/us-equities/sp-500/ | HTTP 403. |
| https://www.spglobal.com/marketintelligence/en/solutions/global-industry-classification-standard | HTTP 403. |
| https://www.spglobal.com/spdji/en/documents/additional-material/gics-map-2023.xlsx | HTTP 403. |
| https://investor.nvidia.com/home/default.aspx | HTTP 403 challenge page. |
| https://investors.paloaltonetworks.com/ | HTTP 403. |
| https://investors.constellationenergy.com/ | HTTP 403. |
| https://ir.aboutamazon.com/overview/default.aspx | HTTP 403 challenge page. |
| https://investor.visa.com/ | HTTP 403 challenge page. |
| https://ir.crowdstrike.com/ | HTTP 403. |
| https://www.caterpillar.com/en/company.html | HTTP 403. |
| https://investors.caterpillar.com/overview/default.aspx | HTTP 403 challenge page. |
| https://www.prologis.com/about | HTTP 403 challenge page. |
| https://www.prologis.com/ | HTTP 403 challenge page. |
| https://www.lilly.com/who-we-are | Redirect to https://www.lilly.com/about, HTTP 403 challenge page. |
| https://corporate.exxonmobil.com/what-we-do | HTTP 403. |
| https://www.microsoft.com/en-us/Investor/earnings/FY-2026-Q4/press-release-webcast | HTTP 403; no earnings facts used. |
| https://www.cmegroup.com/company.html | HTTP 403, IP blocked. |
| https://investor.cmegroup.com/ | HTTP 403. |
| https://ir.theice.com/overview/default.aspx | HTTP 403 challenge page. |
| https://www.lamresearch.com/company/ | HTTP 403 challenge page. |
| https://newsroom.lamresearch.com/company-overview | Redirect to `/404`, HTTP 200 with an error page. |
| https://www.deere.com/en/our-company/ | HTTP 404. |
| https://www.pultegroup.com/about-us/default.aspx | HTTP 403 challenge page. |
| https://corporate.mcdonalds.com/corpmcd/our-company.html | HTTP 403. |
| https://www.chevron.com/who-we-are | HTTP 403. |
| https://www.chevron.com/investors | HTTP 403. |
| https://www.williams.com/company/ | HTTP 404. |
| https://corporate.dow.com/en-us/about.html | HTTP 403. |
| https://www.ball.com/company | HTTP 403. |
| https://www.airproducts.com/company | HTTP 403. |
| https://www.americantower.com/us/ | HTTP 403 challenge page. |
| https://investor.kbhome.com/overview/default.aspx | HTTP 403 challenge page. |
| https://www.unitedhealthgroup.com/who-we-are.html | HTTP 200 redirect to a people-and-culture page, not used as business evidence. |
| https://www.eaton.com/us/en-us/company/about-us.html | Request timed out after 25 seconds. |
| https://www.eaton.com/us/en-us/company.html | Request timed out after 25 seconds. |
| https://www.duke-energy.com/our-company/about-us | DNS resolution failed. |
| https://www.eogresources.com/about | Redirect to the trailing-slash URL, HTTP 403. |
| https://www.lennar.com/about | HTTP 403 challenge page. |
| https://www.nexteraenergy.com/about-us/our-company.html | HTTP 404. |

The offline test suite does not repeat these network requests.
It checks loader contracts, source URL safety, market-specific exchanges, duplicate provider symbols, provider fetch arguments and minimum detail requirements for every US basket.
