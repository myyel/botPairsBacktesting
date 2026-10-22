from __future__ import annotations

from datetime import date

import pytest

from pairs_bt.data.archive import (
    DATASETS,
    ArchiveFile,
    days_between,
    file_from_key,
    listing_url,
    months_between,
    parse_s3_listing,
    plan_files,
    symbol_prefix,
)


def test_kline_key_and_url() -> None:
    f = ArchiveFile("klines_1m", "BTCUSDT", "monthly", "2025-09")
    assert f.key == "data/futures/um/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2025-09.zip"
    assert f.checksum_key == f.key + ".CHECKSUM"
    assert f.url("https://data.binance.vision") == "https://data.binance.vision/" + f.key


@pytest.mark.parametrize(
    ("f", "key"),
    [
        (
            ArchiveFile("fundingRate", "ETHUSDT", "monthly", "2026-01"),
            "data/futures/um/monthly/fundingRate/ETHUSDT/ETHUSDT-fundingRate-2026-01.zip",
        ),
        (
            ArchiveFile("aggTrades", "ETHUSDT", "daily", "2026-01-02"),
            "data/futures/um/daily/aggTrades/ETHUSDT/ETHUSDT-aggTrades-2026-01-02.zip",
        ),
        (
            ArchiveFile("klines_5m", "SOLUSDT", "daily", "2026-01-02"),
            "data/futures/um/daily/klines/SOLUSDT/5m/SOLUSDT-5m-2026-01-02.zip",
        ),
    ],
)
def test_keys_roundtrip(f: ArchiveFile, key: str) -> None:
    assert f.key == key
    assert file_from_key(key) == f


def test_file_from_key_rejects_garbage() -> None:
    assert file_from_key("data/futures/um/monthly/klines/BTCUSDT/1m/README.txt") is None
    assert file_from_key("data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2025-09.zip") is None
    assert file_from_key("data/futures/um/monthly/klines/BTCUSDT/3m/BTCUSDT-3m-2025-09.zip") is None


def test_daily_funding_not_allowed() -> None:
    with pytest.raises(ValueError, match="daily"):
        ArchiveFile("fundingRate", "BTCUSDT", "daily", "2025-09-01")


def test_file_covered_days() -> None:
    m = ArchiveFile("klines_1m", "BTCUSDT", "monthly", "2024-02")
    assert (m.first_day, m.last_day) == (date(2024, 2, 1), date(2024, 2, 29))
    d = ArchiveFile("klines_1m", "BTCUSDT", "daily", "2024-02-10")
    assert d.first_day == d.last_day == date(2024, 2, 10)


def test_symbol_prefix() -> None:
    assert (
        symbol_prefix("klines_5m", "BTCUSDT", "daily") == "data/futures/um/daily/klines/BTCUSDT/5m/"
    )
    assert (
        symbol_prefix("fundingRate", "BTCUSDT", "monthly")
        == "data/futures/um/monthly/fundingRate/BTCUSDT/"
    )


def test_calendar_helpers() -> None:
    assert months_between(date(2025, 11, 15), date(2026, 2, 1)) == [
        "2025-11",
        "2025-12",
        "2026-01",
        "2026-02",
    ]
    assert days_between(date(2025, 12, 30), date(2026, 1, 2)) == [
        "2025-12-30",
        "2025-12-31",
        "2026-01-01",
        "2026-01-02",
    ]
    assert months_between(date(2026, 1, 1), date(2026, 1, 1)) == ["2026-01"]


def test_plan_prefers_monthly_and_falls_back_to_daily() -> None:
    files = plan_files(
        "klines_1m",
        "NEWUSDT",
        date(2026, 8, 1),
        date(2026, 9, 3),
        monthly_available={"2026-08"},
        daily_available={"2026-08-31", "2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04"},
    )
    assert [(f.period, f.label) for f in files] == [
        ("monthly", "2026-08"),
        ("daily", "2026-09-01"),
        ("daily", "2026-09-02"),
        ("daily", "2026-09-03"),
    ]


def test_plan_funding_monthly_only() -> None:
    files = plan_files(
        "fundingRate",
        "BTCUSDT",
        date(2026, 8, 1),
        date(2026, 9, 30),
        monthly_available={"2026-08"},
        daily_available={"2026-09-01"},
    )
    assert [f.label for f in files] == ["2026-08"]


def test_plan_empty_when_nothing_available() -> None:
    assert plan_files("klines_1m", "X", date(2026, 1, 1), date(2026, 1, 5), set(), set()) == []


def test_dataset_specs() -> None:
    assert DATASETS["klines_1m"].interval_ms == 60_000
    assert DATASETS["klines_5m"].interval_ms == 300_000
    assert DATASETS["fundingRate"].interval_ms is None
    assert not DATASETS["fundingRate"].daily_available


_XML_PAGE = b"""<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><Name>data.binance.vision</Name>
<Prefix>data/futures/um/monthly/klines/</Prefix><Marker></Marker>
<NextMarker>data/futures/um/monthly/klines/WMTUSDT/</NextMarker><MaxKeys>1000</MaxKeys>
<Delimiter>/</Delimiter><IsTruncated>true</IsTruncated>
<CommonPrefixes><Prefix>data/futures/um/monthly/klines/0GUSDT/</Prefix></CommonPrefixes>
<CommonPrefixes><Prefix>data/futures/um/monthly/klines/BTCUSDT/</Prefix></CommonPrefixes>
<Contents><Key>data/futures/um/monthly/klines/x.zip</Key><Size>10</Size></Contents>
</ListBucketResult>"""


def test_parse_s3_listing() -> None:
    page = parse_s3_listing(_XML_PAGE)
    assert page.prefixes == (
        "data/futures/um/monthly/klines/0GUSDT/",
        "data/futures/um/monthly/klines/BTCUSDT/",
    )
    assert page.keys == ("data/futures/um/monthly/klines/x.zip",)
    assert page.is_truncated
    assert page.next_marker == "data/futures/um/monthly/klines/WMTUSDT/"


def test_parse_s3_listing_truncated_without_next_marker_uses_last_key() -> None:
    xml = (
        b'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        b"<IsTruncated>true</IsTruncated><Contents><Key>a/1</Key></Contents>"
        b"<Contents><Key>a/2</Key></Contents></ListBucketResult>"
    )
    page = parse_s3_listing(xml)
    assert page.next_marker == "a/2"


def test_listing_url_encodes_params() -> None:
    url = listing_url("https://s3.example", "data/futures/um/", marker="a/b/", delimiter=True)
    assert url == "https://s3.example?prefix=data%2Ffutures%2Fum%2F&marker=a%2Fb%2F&delimiter=%2F"
    assert listing_url("https://s3.example", "p/", marker="", delimiter=False) == (
        "https://s3.example?prefix=p%2F"
    )


def test_plan_partial_month_prefers_daily() -> None:
    files = plan_files(
        "aggTrades",
        "BTCUSDT",
        date(2025, 9, 1),
        date(2025, 9, 2),
        monthly_available={"2025-09"},
        daily_available={"2025-09-01", "2025-09-02", "2025-09-03"},
    )
    assert [(f.period, f.label) for f in files] == [
        ("daily", "2025-09-01"),
        ("daily", "2025-09-02"),
    ]


def test_plan_partial_month_without_daily_uses_monthly() -> None:
    files = plan_files(
        "fundingRate", "BTCUSDT", date(2025, 9, 10), date(2025, 9, 20), {"2025-09"}, set()
    )
    assert [(f.period, f.label) for f in files] == [("monthly", "2025-09")]
