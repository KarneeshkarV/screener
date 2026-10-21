"""Offline India catalogue contracts, including the 2026-10-20 NSE symbol review."""

from __future__ import annotations

import re
from datetime import date
from importlib.resources import files
from urllib.parse import urlsplit

import pytest
import yaml

from screener.sector_rotation.catalogue import load_sector_catalogue

# Inherited video labels and timestamps, not claims about video constituents.
ORIGINAL_VIDEO_TIMESTAMPS = {
    "pharma": "3:00",
    "us_generics": "3:30",
    "diagnostics": "3:40",
    "cdmo": "4:20",
    "api": "4:00",
    "branded_pharma": "4:40",
    "niche_cdmo": "4:40",
    "generic_cdmo": "4:40",
    "capital_goods": "5:00",
    "machines": "5:00",
    "bearings": "5:00",
    "pumps": "5:00",
    "transformers": "5:00",
    "cables": "5:00",
    "cranes": "5:00",
    "switchgear": "5:00",
    "compressors": "5:00",
    "valves": "5:00",
    "motors": "5:00",
    "boilers": "5:00",
    "conveyors": "5:00",
    "shipbuilding": "8:50",
    "healthcare_ayurveda": "5:20",
    "energy_transmission": "5:20",
    "cnc_machines": "9:20",
    "cryogenics": "10:10",
    "heat_exchangers": "9:40",
    "pre_engineered_buildings": "10:40",
    "investment_holding": "10:10",
    "defence_aerospace": "10:10",
    "energy_finance": "11:50",
    "precision_engineering": "11:50",
    "castings": "12:20",
    "home_furnishings": "6:00",
    "sanitaryware_bath_fittings": "6:00",
    "chemicals_amines": "12:50",
    "warehousing": "13:00",
    "vfx_post_production": "13:30",
    "chemicals_fluorochemicals": "15:20",
    "chemicals_oleochemicals": "15:20",
    "packaging_bopp_films": "15:30",
    "it_education": "15:20",
    "paints": "15:20",
    "energy_renewables": "15:20",
    "batteries": "16:00",
    "shipping": "16:00",
    "cement": "16:00",
    "tea": "16:50",
    "engineered_goods": "16:50",
    "hospitals": "17:50",
    "broking": "17:40",
    "animal_feed": "17:40",
    "paper_packaging": "17:40",
    "industrial_gases": "17:50",
    "credit_rating": "17:50",
    "hotels": "17:50",
    "retail": "17:50",
    "quick_service_restaurants": "18:10",
    "food_platforms": "18:10",
    "fertilizers": "12:00",
    "edible_oils": "12:00",
    "chemicals_agrochemicals": "15:20",
    "chemicals_dyes": "15:20",
    "energy_refining": "15:20",
    "chemicals_specialty": "17:50",
}

# Symbol identity only, from the main-board and SME files in the source document.
# This fixed review set prevents an accidental invented ticker, not future delistings.
REVIEWED_NSE_SYMBOLS = frozenset(
    """
    AARTIIND AAVAS ABB ABSLAMC ACE ADANIENSOL ADANIGREEN ADANIPORTS AJANTPHARM
    ALKEM ALKYLAMINE AMBUJACEM ANDHRAPAP ANGELONE APEX APOLLOHOSP APTUS ARE&M
    ASAHISONG ASHOKLEY ASIANPAINT ATAM ATGL ATUL AUBANK AUROPHARMA AVANTIFEED
    AWL AXISBANK AZAD BAJAJ-AUTO BAJAJHLDNG BAJFINANCE BALAMINES BANKBARODA
    BASILIC BBL BDL BEL BEML BERGEPAINT BHARATFORG BHARTIARTL BHEL BIKAJI
    BLUESTARCO BODALCHEM BOSCHLTD BPCL BRIGADE BRITANNIA CANBK CARERATING CERA
    CESC CGPOWER CHALET CHAMBLFERT CHOLAFIN CIPLA COCHINSHIP COROMANDEL COSMOFIRST
    CRISIL CROMPTON CUMMINSIND DABUR DATAPATTNS DEEPAKNTR DEVYANI DIGIKORE DIVISLAB
    DLF DMART DRREDDY DTIL EICHERMOT EIHOTEL ELECON ELGIEQUIP EMAMILTD EPACKPEB
    EQUITASBNK ESTER ETERNAL EXIDEIND FEDERALBNK FINCABLES FINEORG FLUOROCHEM
    FORTIS GAIL GALAXYSURF GESHIP GICRE GODREJAGRO GODREJCP GODREJIND GODREJPROP
    GOKULAGRO GPPL GRANULES GRSE GSFC HAL HAVELLS HBLENGINE HCLTECH HDFCAMC
    HDFCBANK HDFCLIFE HEROMOTOCO HFCL HIMATSEIDE HINDALCO HINDPETRO HINDUNILVR
    HINDWAREAP HINDZINC HOMEFIRST ICICIBANK ICICIGI ICRA IDEA IGL IIFL IIFLCAPS
    INDHOTEL INDIGOPNTS INDOMIM INDOTECH INFY INGERRAND INOXINDIA INTELLECT
    INTERARCH IOC IREDA ITC ITI JAYSREETEA JINDALPOLY JINDALSTEL JKCEMENT JKPAPER
    JSWENERGY JSWINFRA JSWSTEEL JUBLFOOD JYOTICNC KAJARIACER KALYANKJIL KANSAINER
    KEI KICL KIMS KIRIINDUS KIRLOSBROS KIRLPNU KOTAKBANK KPIL KPITTECH KRN KSB
    LALPATHLAB LAURUSLABS LEMONTREE LICHSGFIN LICI LINDEINDIA LMW LT LTF LTM LTTS
    LUPIN M&M M&MFIN MACPOWER MAHLOG MAHSCOOTER MANAPPURAM MANKIND MARICO MARUTI
    MAXHEALTH MAZDOCK MCLEODRUSS METROPOLIS MFSL MGL MOTHERSON MOTILALOFS MTARTECH
    MUTHOOTFIN NAM-INDIA NATIONALUM NAVINFLUOR NESTLEIND NEULANDLAB NEWGEN NH NHPC
    NIACL NIITLTD NIITMTS NMDC NRBBEARING NTPC NTPCGREEN OBEROIRLTY OFSS OIL
    OLECTRA ONGC PARADEEP PATANJALI PENIND PERSISTENT PFC PFOCUS PHOENIXLTD PIIND
    PNB PNBHOUSING POLYCAB POLYPLEX POWERGRID PPLPHARMA PREMIERENE PRESTIGE
    RADICO RAPPID RBA RECLTD RELIANCE RRKABEL SAIL SAILIFE SANGHVIMOV SANSERA
    SAPPHIRE SBILIFE SBIN SCHAEFFLER SCI SENCO SGIL SHAKTIPUMP SHARDACROP
    SHOPERSTOP SHREECEM SHRIRAMFIN SIEMENS SJVN SKFINDIA SOBHA SONACOMS SRF
    STALLION STARHEALTH STEELCAS SUDARSCHEM SUMICHEM SUNPHARMA SWIGGY SYNGENE
    TARIL TATACOMM TATACONSUM TATAELXSI TATAPOWER TATASTEEL TATATECH TCI TCS
    TECHM TEJASNET THANGAMAYL THERMAX THYROCARE TIL TIMKEN TITAN TMCV TMPV
    TORNTPHARM TORNTPOWER TRENT TRF TRIDENT TRITONV TVSMOTOR TVSSCS UBL UFLEX
    UJJIVANSFB ULTRACEMCO UNIONBANK UNIPARTS UNITDSPR UNITEDTEA UNOMINDA UPL
    UTIAMC VBL VEDL VIJAYA VINATIORGA VMART VOLTAMP VOLTAS WAAREEENER WELSPUNLIV
    WESTLIFE WHIRLPOOL WIPRO WPIL WSTCSTPAPR ZENTEC
    """.split()
)


def _india_catalogue_data():
    """Read the bundled India YAML without making any network requests."""
    return yaml.safe_load(
        files("screener.sector_rotation")
        .joinpath("sector_groups.yaml")
        .read_text(encoding="utf-8")
    )


def test_india_catalogue_preserves_original_video_ids_and_timestamps():
    catalogue = load_sector_catalogue(market="india")
    groups = {group.id: group for group in catalogue.groups}
    assert len(ORIGINAL_VIDEO_TIMESTAMPS) == 65
    assert set(ORIGINAL_VIDEO_TIMESTAMPS) <= groups.keys()
    for group_id, timestamp in ORIGINAL_VIDEO_TIMESTAMPS.items():
        assert groups[group_id].video_at == timestamp
    assert catalogue.source == "https://www.youtube.com/watch?v=d-I6BG5F-Yg"


@pytest.mark.parametrize(
    "required_ids",
    [
        {"banks", "private_banks", "public_sector_banks", "small_finance_banks"},
        {"financials", "nbfc_lending", "housing_finance", "gold_lending"},
        {"asset_management", "life_insurance", "general_insurance"},
        {
            "information_technology",
            "it_services",
            "engineering_rd",
            "enterprise_software",
        },
        {"automobiles", "passenger_vehicles", "two_wheelers", "commercial_vehicles"},
        {"auto_components", "telecom", "telecom_equipment"},
        {"utilities", "power_generation", "gas_distribution"},
        {"metals", "steel", "nonferrous_metals"},
        {"real_estate", "residential_developers", "commercial_property"},
        {"consumer_staples", "packaged_foods", "personal_care", "beverages"},
        {"consumer_durables", "jewellery", "healthcare", "energy"},
    ],
)
def test_india_catalogue_expands_distinct_exposure_areas(required_ids):
    catalogue = load_sector_catalogue(market="india")
    groups = {group.id for group in catalogue.groups}
    assert len(groups - ORIGINAL_VIDEO_TIMESTAMPS.keys()) >= 25
    assert required_ids <= groups


@pytest.mark.parametrize(
    "group", _india_catalogue_data()["groups"], ids=lambda g: g["id"]
)
def test_india_catalogue_every_group_has_details_and_primary_sources(group):
    assert isinstance(group["description"], str) and group["description"].strip()
    for field in ("drivers", "risks", "sources"):
        assert isinstance(group[field], list) and group[field]
        assert all(isinstance(item, str) and item.strip() for item in group[field])
        assert len(group[field]) == len(set(group[field]))
    for source in group["sources"]:
        parsed = urlsplit(source)
        assert parsed.scheme in {"http", "https"} and parsed.hostname
        assert not parsed.username and not parsed.password
        assert parsed.hostname not in {"youtube.com", "www.youtube.com", "youtu.be"}
    assert any("nseindia.com/" in source for source in group["sources"])
    assert any("nseindia.com/" not in source for source in group["sources"])


@pytest.mark.parametrize(
    "group", _india_catalogue_data()["groups"], ids=lambda g: g["id"]
)
def test_india_catalogue_members_match_reviewed_nse_symbols(group):
    members = group["members"]
    assert members and len(members) == len(set(members))
    assert all(re.fullmatch(r"NSE:[A-Z0-9_&-]+", member) for member in members)
    assert {member.removeprefix("NSE:") for member in members} <= REVIEWED_NSE_SYMBOLS
    if group["id"] not in ORIGINAL_VIDEO_TIMESTAMPS:
        assert 3 <= len(members) <= 7
        assert "video_at" not in group
        assert "video" not in group["description"].lower()


def test_india_catalogue_retains_membership_and_return_limitations():
    catalogue = load_sector_catalogue(market="india")
    assert catalogue.observed_on == date(2026, 10, 20)
    assert catalogue.benchmark == "^CRSLDX"
    assert catalogue.membership_basis == "curated_research"
    note = catalogue.membership_note.lower()
    for limitation in (
        "not official indices",
        "survivorship bias",
        "pure-play",
        "adjusted stock returns include distributions",
        "price benchmark does not",
        "inherited",
        "no video provenance",
        "not liquidity",
    ):
        assert limitation in note


def test_india_catalogue_uses_reviewed_corporate_action_symbols():
    groups = {group.id: group for group in load_sector_catalogue(market="india").groups}
    assert "NSE:LTM" in groups["information_technology"].members
    assert "NSE:TMPV" in groups["passenger_vehicles"].members
    assert "NSE:TMCV" in groups["commercial_vehicles"].members
    assert "NSE:OBEROIRLTY" in groups["real_estate"].members
    assert "NSE:BLUESTARCO" in groups["consumer_durables"].members
    all_members = {member for group in groups.values() for member in group.members}
    assert all_members.isdisjoint({"NSE:LTIM", "NSE:TATAMOTORS", "NSE:GUJGASLTD"})
    assert "BE trade-to-trade" in groups["telecom_equipment"].notes
    assert "SME" in groups["valves"].description
    assert "SME" in groups["vfx_post_production"].description


@pytest.mark.parametrize(
    "group_ids",
    [
        ("banks", "private_banks", "public_sector_banks", "small_finance_banks"),
        (
            "financials",
            "nbfc_lending",
            "housing_finance",
            "gold_lending",
            "asset_management",
            "life_insurance",
            "general_insurance",
        ),
        (
            "information_technology",
            "it_services",
            "engineering_rd",
            "enterprise_software",
        ),
        (
            "automobiles",
            "passenger_vehicles",
            "two_wheelers",
            "commercial_vehicles",
            "auto_components",
        ),
        ("telecom", "telecom_equipment"),
        ("utilities", "power_generation", "gas_distribution"),
        ("metals", "steel", "nonferrous_metals"),
        ("real_estate", "residential_developers", "commercial_property"),
        ("consumer_staples", "packaged_foods", "personal_care", "beverages"),
        ("consumer_durables", "jewellery"),
        ("energy", "oil_gas_production"),
    ],
)
def test_added_india_sector_and_related_industries_share_parent_filter(group_ids):
    groups = {group.id: group for group in load_sector_catalogue().groups}
    assert len({groups[group_id].parent for group_id in group_ids}) == 1
