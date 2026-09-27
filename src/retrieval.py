"""Bounded-memory hashed postings over the full target corpus, no country filter.

Original TSV records remain the source of truth. Byte offsets support random
access; physical-line parsing is checked strictly during index construction.
The index is rebuilt if input sizes/mtimes or the version change.
"""
import csv
import hashlib
import json
import mmap
import re
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from rapidfuzz.fuzz import ratio, token_set_ratio
from .normalization import normalize_text, transliterate

VERSION = 1
SHARDS = 64
PAIR = np.dtype([("h", "<u8"), ("r", "<u4")])
STOP = frozenset("the and of for a an co company inc incorporated corp corporation llc ltd limited private pvt llp enterprises enterprise services service group international india sarl sas sa france dba".split())
ASTOP = frozenset("road rd street st avenue ave lane ln drive dr near no number unit floor fl plot dist district state building bldg city nagar sector block".split())


@lru_cache(maxsize=150000)
def h64(value):
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "little")


def parse_line(line):
    row = next(csv.reader([line.decode("utf-8-sig").rstrip("\r\n")], delimiter="\t", strict=True))
    if len(row) != 4:
        raise ValueError("Expected a single physical TSV record with four fields; multiline input needs a different offset reader")
    return tuple(row)


def prepared(row):
    eid, name, address, country = row
    n, a = normalize_text(name), normalize_text(address)
    nt, at = (n if n.isascii() else transliterate(n)), (a if a.isascii() else transliterate(a))
    return (eid, n, a, normalize_text(country), nt, at)


def blocking_keys(rec):
    _, n, a, c, nt, at = rec
    tokens = list(dict.fromkeys(t for t in nt.split() if t not in STOP and len(t) > 1))
    addr = list(dict.fromkeys(t for t in at.split() if t not in ASTOP and len(t) > 1))
    keys = set()
    def add(kind, value):
        if value:
            keys.add(h64(kind + ":" + value))
    add("exact", nt.replace(" ", ""))
    add("core", "".join(sorted(tokens)))
    # Order-invariant token postings; frequent postings are skipped at query time.
    for token in sorted(tokens, key=h64)[:3]:
        add("nt", token)
    for token in sorted({t[:5] for t in tokens if len(t) >= 5}, key=h64)[:2]:
        add("np", token)
    add("address", at.replace(" ", ""))
    # Complementary address keys recover names with spelling/script changes.
    numeric = re.findall(r"\b\d+\b", at)
    alpha = sorted((t for t in addr if not t.isdigit()), key=h64)[:3]
    if len(alpha) >= 2:
        add("ap", "|".join(sorted(alpha[:2])))
    for token in alpha[:2]:
        for number in numeric[:2]:
            add("an", number + "|" + token)
    return sorted(keys)


def fingerprint(paths):
    return {"version": VERSION, "files": [{"path": str(p.resolve()), "size": p.stat().st_size,
             "mtime_ns": p.stat().st_mtime_ns} for p in paths]}


def build_index(paths, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    meta = directory / "manifest.json"
    sig = fingerprint(paths)
    if meta.exists() and json.loads(meta.read_text()) == sig:
        print("Reusing index", directory, flush=True)
        return
    handles = [(directory / f"part{i}.bin").open("wb") for i in range(SHARDS)]
    buffers = [bytearray() for _ in handles]
    offsets = (directory / "offsets.bin").open("wb")
    source = (directory / "sources.bin").open("wb")
    import struct
    count, started = 0, time.time()
    try:
        for si, path in enumerate(paths):
            with path.open("rb") as stream:
                header = stream.readline().decode("utf-8-sig").strip().split("\t")
                assert header == ["entity_id", "business_name", "business_address", "country"]
                while True:
                    offset = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    row = parse_line(line)
                    offsets.write(struct.pack("<Q", offset))
                    source.write(bytes([si]))
                    for key in blocking_keys(prepared(row)):
                        buffers[key % SHARDS].extend(struct.pack("<QI", key, count))
                    count += 1
                    if count % 10000 == 0:
                        for h, b in zip(handles, buffers):
                            h.write(b)
                            b.clear()
                    if count % 250000 == 0:
                        print(f"Index {directory.name}: {count:,} rows, {time.time()-started:.0f}s", flush=True)
        for h, b in zip(handles, buffers):
            h.write(b)
    finally:
        offsets.close()
        source.close()
        for h in handles:
            h.close()
    for i in range(SHARDS):
        path = directory / f"part{i}.bin"
        data = np.fromfile(path, dtype=PAIR)
        data.sort(order=["h", "r"])
        data["h"].tofile(directory / f"keys{i}.bin")
        data["r"].tofile(directory / f"rows{i}.bin")
        del data
        path.unlink()
    meta.write_text(json.dumps(sig, indent=2))
    print(f"Index complete: {count:,} targets", flush=True)


def build_prepared(paths, directory):
    """Lossless cache of prepared() output: removes repeated normalization at inference."""
    import struct
    directory=Path(directory)
    meta=directory/'prepared_manifest.json'
    sig=fingerprint(paths)
    if meta.exists() and json.loads(meta.read_text())==sig:
        return
    count=0
    start=time.time()
    with (directory/'prepared.partial.bin').open('wb') as target, (directory/'prepared_offsets.partial.bin').open('wb') as offsets:
        for path in paths:
            with path.open('rb') as f:
                f.readline()
                for line in f:
                    eid,n,a,c,nt,at=prepared(parse_line(line))
                    offsets.write(struct.pack('<Q',target.tell()))
                    target.write(('\t'.join((eid,n,a,c,'=' if nt==n else nt,'=' if at==a else at))+'\n').encode('utf-8'))
                    count+=1
                    if count%500000==0:
                        print(f'Prepared cache: {count:,} targets, {time.time()-start:.0f}s',flush=True)
    (directory/'prepared.partial.bin').replace(directory/'prepared.bin')
    (directory/'prepared_offsets.partial.bin').replace(directory/'prepared_offsets.bin')
    meta.write_text(json.dumps(sig,indent=2))


class CandidateIndex:
    def __init__(self, paths, directory, max_posting=1200, max_candidates=24, balance_sources=False):
        self.paths, self.directory = paths, Path(directory)
        self.streams = [p.open("rb") for p in paths]
        self.raw_maps = [mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) for f in self.streams]
        self.offsets = np.asarray(np.memmap(self.directory / "offsets.bin", dtype="<u8", mode="r"))
        self.sources = np.asarray(np.memmap(self.directory / "sources.bin", dtype="u1", mode="r"))
        def mapped(name, dtype):
            path = self.directory / name
            return np.asarray(np.memmap(path, dtype=dtype, mode="r")) if path.stat().st_size else np.empty(0,dtype=dtype)
        self.keys = [mapped(f"keys{i}.bin", "<u8") for i in range(SHARDS)]
        self.rows = [mapped(f"rows{i}.bin", "<u4") for i in range(SHARDS)]
        self.max_posting, self.max_candidates = max_posting, max_candidates
        self.balance_sources=balance_sources
        self.prepared_map=None
        self.try_prepared()

    def try_prepared(self):
        if self.prepared_map is not None:
            return
        meta=self.directory/'prepared_manifest.json'
        if meta.exists() and json.loads(meta.read_text())==fingerprint(self.paths):
            self.prepared_stream=(self.directory/'prepared.bin').open('rb')
            self.prepared_map=mmap.mmap(self.prepared_stream.fileno(),0,access=mmap.ACCESS_READ)
            self.prepared_offsets=np.asarray(np.memmap(self.directory/'prepared_offsets.bin',dtype='<u8',mode='r'))
            assert len(self.prepared_offsets)==len(self.offsets)

    @lru_cache(maxsize=12000)
    def raw(self, rid):
        data = self.raw_maps[int(self.sources[rid])]
        start=int(self.offsets[rid])
        end=data.find(b'\n',start)
        return parse_line(data[start:end+1 if end>=0 else len(data)])

    @lru_cache(maxsize=12000)
    def record(self, rid):
        if self.prepared_map is not None:
            start=int(self.prepared_offsets[rid])
            end=int(self.prepared_offsets[rid+1]) if rid+1<len(self.prepared_offsets) else len(self.prepared_map)
            eid,n,a,c,nt,at=self.prepared_map[start:end-1].decode('utf-8').split('\t')
            return eid,n,a,c,n if nt=='=' else nt,a if at=='=' else at
        return prepared(self.raw(rid))

    def retrieve(self, left, mode="multi"):
        counts = defaultdict(int)
        keys = blocking_keys(left) if mode == "multi" else [h64("exact:" + left[4].replace(" ", ""))] if left[4] else []
        for key in keys:
            shard = key % SHARDS
            lo, hi = np.searchsorted(self.keys[shard], np.uint64(key), side="left"), np.searchsorted(self.keys[shard], np.uint64(key), side="right")
            if hi-lo > self.max_posting:
                continue
            for rid in self.rows[shard][lo:hi]:
                counts[int(rid)] += 1
        # Bound random I/O before fine ranking. Stable tie-breaking is deterministic.
        ordered = sorted(counts, key=lambda r: (-counts[r], r))
        if self.balance_sources:
            pool = [r for source in (0,1) for r in [x for x in ordered if int(self.sources[x])==source][:80]]
        else:
            pool = ordered[:160]
        def score(rid):
            right = self.record(rid)
            ns = ratio(left[4], right[4]) if left[4] and right[4] else 0
            ads = token_set_ratio(left[5], right[5]) if left[5] and right[5] else 0
            return (.58*ns + .42*ads + min(counts[rid], 5), -rid)
        return sorted(pool, key=score, reverse=True)[:self.max_candidates]

    def close(self):
        self.raw.cache_clear()
        self.record.cache_clear()
        for mapping in self.raw_maps:mapping.close()
        if self.prepared_map is not None:
            self.prepared_map.close()
            self.prepared_stream.close()
        for f in self.streams:
            f.close()
