"""Sahte data.binance.vision arşivi + fapi exchangeInfo (ağsız testler için).

Gerçek formatlar (2025-09 probe):
- klines: `open_time,open,high,low,close,volume,close_time,quote_volume,count,
  taker_buy_volume,taker_buy_quote_volume,ignore` (eski dosyalarda header yok)
- fundingRate: `calc_time,funding_interval_hours,last_funding_rate`
- aggTrades: `agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker`
- `.CHECKSUM`: `<sha256 hex>  <dosya adı>`
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from urllib.parse import parse_qs, urlparse

import httpx
import numpy as np

KLINE_HEADER = (
    "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
    "taker_buy_volume,taker_buy_quote_volume,ignore"
)
FUNDING_HEADER = "calc_time,funding_interval_hours,last_funding_rate"
AGG_HEADER = "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker"

KlineRow = tuple[int, float, float, float, float, float, float, int, float, float]

ARCHIVE_HOST = "data.binance.vision"
S3_HOST = "s3-ap-northeast-1.amazonaws.com"
FAPI_HOST = "fapi.binance.com"


def day_ms(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp() * 1000)


def make_klines(
    rng: np.random.Generator,
    start_ms: int,
    n: int,
    interval_ms: int = 60_000,
    price0: float = 100.0,
    tick: float = 0.01,
) -> list[KlineRow]:
    """Tutarlı (high>=max(o,c), low<=min(o,c)) tick grid'li sentetik kline satırları.

    Dönen tuple: (open_time, o, h, l, c, volume, quote_volume, count, taker_buy_vol, taker_quote).
    """
    rows = []
    price = price0
    for i in range(n):
        o = round(price / tick) * tick
        c = round(o * float(np.exp(0.001 * rng.standard_normal())) / tick) * tick
        h = max(o, c) + tick * int(rng.integers(0, 3))
        lo = min(o, c) - tick * int(rng.integers(0, 3))
        vol = round(float(rng.uniform(1, 100)), 3)
        qv = round(vol * (o + c) / 2, 5)
        cnt = int(rng.integers(1, 500))
        tb = round(vol * float(rng.uniform(0, 1)), 3)
        rows.append((start_ms + i * interval_ms, o, h, lo, c, vol, qv, cnt, tb, round(tb * c, 5)))
        price = c
    return rows


def _fmt(x: float, tick: float) -> str:
    decimals = max(0, -int(np.floor(np.log10(tick))))
    return f"{x:.{decimals}f}"


def klines_csv(
    rows: Sequence[KlineRow],
    interval_ms: int = 60_000,
    *,
    header: bool = True,
    ts_unit_us: bool = False,
    tick: float = 0.01,
) -> bytes:
    mult = 1000 if ts_unit_us else 1
    lines = [KLINE_HEADER] if header else []
    for ot, o, h, lo, c, v, qv, cnt, tb, tq in rows:
        ct = ot + interval_ms - 1
        lines.append(
            f"{ot * mult},{_fmt(o, tick)},{_fmt(h, tick)},{_fmt(lo, tick)},{_fmt(c, tick)},"
            f"{v},{ct * mult},{qv},{cnt},{tb},{tq},0"
        )
    return ("\n".join(lines) + "\n").encode()


def resample_rows(
    rows: Sequence[KlineRow],
    factor: int,
) -> list[KlineRow]:
    out = []
    for i in range(0, len(rows) - len(rows) % factor, factor):
        chunk = rows[i : i + factor]
        out.append(
            (
                chunk[0][0],
                chunk[0][1],
                max(r[2] for r in chunk),
                min(r[3] for r in chunk),
                chunk[-1][4],
                round(sum(r[5] for r in chunk), 3),
                round(sum(r[6] for r in chunk), 5),
                sum(r[7] for r in chunk),
                round(sum(r[8] for r in chunk), 3),
                round(sum(r[9] for r in chunk), 5),
            )
        )
    return out


def funding_csv(rows: Sequence[tuple[int, int, float]], *, header: bool = True) -> bytes:
    lines = [FUNDING_HEADER] if header else []
    lines += [f"{t},{h},{r:.8f}" for t, h, r in rows]
    return ("\n".join(lines) + "\n").encode()


def agg_trades_csv(
    rows: Sequence[tuple[int, float, float, int, bool]], *, header: bool = True
) -> bytes:
    """rows: (transact_time_ms, price, qty, agg_id, is_buyer_maker)."""
    lines = [AGG_HEADER] if header else []
    for t, p, q, aid, m in rows:
        lines.append(f"{aid},{p},{q},{aid * 10},{aid * 10 + 1},{t},{str(m).lower()}")
    return ("\n".join(lines) + "\n").encode()


def zip_bytes(inner_name: str, content: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # sabit tarih => deterministik zip baytları
        info = zipfile.ZipInfo(inner_name, date_time=(2020, 1, 1, 0, 0, 0))
        zf.writestr(info, content)
    return buf.getvalue()


def checksum_text(data: bytes, filename: str) -> bytes:
    return f"{hashlib.sha256(data).hexdigest()}  {filename}\n".encode()


def exchange_info_json(symbols: Sequence[dict[str, object]], server_time_ms: int) -> bytes:
    return json.dumps(
        {"timezone": "UTC", "serverTime": server_time_ms, "symbols": list(symbols)}
    ).encode()


def ei_symbol(
    symbol: str,
    *,
    status: str = "TRADING",
    contract_type: str = "PERPETUAL",
    quote: str = "USDT",
    tick: str = "0.01",
    step: str = "0.001",
    min_qty: str = "0.001",
    min_notional: str = "5",
    onboard_ms: int = 1_500_000_000_000,
    delivery_ms: int = 4_133_404_800_000,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "pair": symbol,
        "contractType": contract_type,
        "deliveryDate": delivery_ms,
        "onboardDate": onboard_ms,
        "status": status,
        "baseAsset": symbol.removesuffix(quote),
        "quoteAsset": quote,
        "filters": [
            {
                "filterType": "PRICE_FILTER",
                "minPrice": "0.01",
                "maxPrice": "1000000",
                "tickSize": tick,
            },
            {"filterType": "LOT_SIZE", "minQty": min_qty, "maxQty": "1000", "stepSize": step},
            {"filterType": "MIN_NOTIONAL", "notional": min_notional},
        ],
    }


@dataclass
class FakeArchive:
    """Anahtar -> bayt eşlemesi; S3 listing (sayfalı) + dosya + exchangeInfo servis eder.

    `fail_once`: ilk istekte 503 dönecek anahtarlar (retry testi).
    `corrupt`: CHECKSUM'ı bilinçli olarak yanlış servis edilen anahtarlar.
    """

    files: dict[str, bytes] = field(default_factory=dict)
    exchange_info: bytes = b'{"serverTime": 0, "symbols": []}'
    page_size: int = 1000
    fail_once: set[str] = field(default_factory=set)
    corrupt: set[str] = field(default_factory=set)
    requests: list[str] = field(default_factory=list)

    def add_zip(self, key: str, csv: bytes, *, with_checksum: bool = True) -> bytes:
        name = key.rsplit("/", 1)[-1]
        data = zip_bytes(name.removesuffix(".zip") + ".csv", csv)
        self.files[key] = data
        if with_checksum:
            self.files[key + ".CHECKSUM"] = checksum_text(data, name)
        return data

    # ------------------------------------------------------------ http
    def handler(self, request: httpx.Request) -> httpx.Response:
        url = urlparse(str(request.url))
        self.requests.append(str(request.url))
        if url.hostname == FAPI_HOST:
            return httpx.Response(200, content=self.exchange_info)
        if url.hostname == S3_HOST:
            return self._listing(parse_qs(url.query))
        key = url.path.lstrip("/")
        if key in self.fail_once:
            self.fail_once.discard(key)
            return httpx.Response(503)
        if key.endswith(".CHECKSUM") and key.removesuffix(".CHECKSUM") in self.corrupt:
            return httpx.Response(200, content=b"0" * 64 + b"  x.zip\n")
        if key not in self.files:
            return httpx.Response(404)
        return httpx.Response(200, content=self.files[key])

    def _listing(self, q: dict[str, list[str]]) -> httpx.Response:
        prefix = q.get("prefix", [""])[0]
        marker = q.get("marker", [""])[0]
        delimiter = q.get("delimiter", [""])[0]
        entries: list[tuple[str, bool]] = []  # (anahtar/prefix, is_prefix)
        seen: set[str] = set()
        for key in sorted(self.files):
            if not key.startswith(prefix):
                continue
            rest = key[len(prefix) :]
            if delimiter and delimiter in rest:
                cp = prefix + rest.split(delimiter, 1)[0] + delimiter
                if cp not in seen:
                    seen.add(cp)
                    entries.append((cp, True))
            else:
                entries.append((key, False))
        entries = [e for e in entries if e[0] > marker]
        page, truncated = entries[: self.page_size], len(entries) > self.page_size
        parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">',
            f"<Prefix>{prefix}</Prefix><Marker>{marker}</Marker>",
            f"<IsTruncated>{'true' if truncated else 'false'}</IsTruncated>",
        ]
        if truncated:
            parts.append(f"<NextMarker>{page[-1][0]}</NextMarker>")
        for k, is_prefix in page:
            if is_prefix:
                parts.append(f"<CommonPrefixes><Prefix>{k}</Prefix></CommonPrefixes>")
            else:
                parts.append(
                    f"<Contents><Key>{k}</Key><Size>{len(self.files[k])}</Size></Contents>"
                )
        parts.append("</ListBucketResult>")
        return httpx.Response(200, content="".join(parts).encode())

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))
