"""Full-file streamed EDA, with disk-backed exact ID/record uniqueness checks."""
import csv
import hashlib
import json
import sqlite3
import time
from collections import Counter
from pathlib import Path
import numpy as np
from .normalization import detect_script


def length_summary(hist):
    total = sum(hist.values())
    values = sorted(hist)
    def quantile(q):
        rank = int((total-1)*q)
        acc = 0
        for value in values:
            acc += hist[value]
            if acc > rank:
                return value
    return {"min": min(hist), "max": max(hist), "mean":sum(k*v for k,v in hist.items())/total,
            "median":quantile(.5), "p95":quantile(.95)}


def run_eda(root):
    root = Path(root)
    out = root/"reports/eda_statistics.json"
    results = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    for split in ("train","test"):
        for source in (1,2,3):
            path = root/f"dataset/{split}/{split}_source{source}.tsv"
            key = f"{split}_source{source}"
            if key in results:
                continue
            counts, missing, scripts_n, scripts_a, lengths_n, lengths_a = [Counter() for _ in range(6)]
            dbpath = root/"cache/eda_unique.sqlite"
            con = sqlite3.connect(dbpath)
            con.execute("PRAGMA journal_mode=OFF")
            con.execute("PRAGMA synchronous=OFF")
            con.execute("PRAGMA cache_size=-32000")
            con.execute("DROP TABLE IF EXISTS records")
            con.execute("CREATE TABLE records (id TEXT PRIMARY KEY, digest BLOB UNIQUE)")
            total, duplicate_ids, duplicate_rows = 0,0,0
            batch = []
            start=time.time()
            with path.open(encoding="utf-8-sig",newline="") as f:
                for row in csv.DictReader(f,delimiter="\t"):
                    total += 1
                    counts[row["country"]] += 1
                    for field,value in row.items():
                        missing[field] += int(not value.strip())
                    n,a = row["business_name"],row["business_address"]
                    scripts_n[detect_script(n)] += 1
                    scripts_a[detect_script(a)] += 1
                    lengths_n[len(n)] += 1
                    lengths_a[len(a)] += 1
                    digest=hashlib.sha256(json.dumps(list(row.values()),ensure_ascii=False).encode()).digest()
                    batch.append((row["entity_id"],digest))
                    if len(batch)==20000:
                        before=con.total_changes
                        con.executemany("INSERT OR IGNORE INTO records VALUES (?,?)",batch)
                        duplicate_ids += len(batch)-(con.total_changes-before)
                        batch.clear()
                    if total%1000000==0:
                        print(f"EDA {key}: {total:,} ({time.time()-start:.0f}s)",flush=True)
            before=con.total_changes
            con.executemany("INSERT OR IGNORE INTO records VALUES (?,?)",batch)
            duplicate_ids += len(batch)-(con.total_changes-before)
            # A repeated full row necessarily has a repeated ID. Re-scan only if duplicates exist.
            if duplicate_ids:
                seen=set()
                with path.open(encoding="utf-8-sig",newline="") as f:
                    for row in csv.reader(f,delimiter="\t"):
                        digest=hashlib.sha256(json.dumps(row,ensure_ascii=False).encode()).digest()
                        duplicate_rows += digest in seen
                        seen.add(digest)
            unique=con.execute("SELECT COUNT(*) FROM records").fetchone()[0]
            con.close()
            results[key]={"rows":total,"unique_entity_ids":unique,"duplicate_entity_ids":duplicate_ids,
                "duplicate_full_rows":duplicate_rows,"missing":dict(missing),"countries":dict(counts),
                "name_scripts":dict(scripts_n),"address_scripts":dict(scripts_a),
                "name_lengths":length_summary(lengths_n),"address_lengths":length_summary(lengths_a)}
            out.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
            print(f"EDA completed {key}: {total:,}",flush=True)
    if "ground_truth" not in results:
        hist, sources = Counter(),Counter()
        rows=0
        with (root/"dataset/train/train_ground_truth.tsv").open(encoding="utf-8-sig",newline="") as f:
            for row in csv.DictReader(f,delimiter="\t"):
                ids=[x.strip() for x in row["matched_entity_ids"].split(",") if x.strip()]
                if len(set(ids))!=len(ids):
                    raise ValueError("Duplicate truth IDs")
                hist[len(ids)]+=1
                sources.update(x.split("-")[0] for x in ids)
                rows+=1
        results["ground_truth"]={"rows":rows,"match_count_distribution":dict(hist),"target_source_pairs":dict(sources),
            "zero_matches":hist[0],"one_match":hist[1],"multiple_matches":sum(v for k,v in hist.items() if k>1)}
        out.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
    for key,value in results.items():
        if key=='ground_truth':continue
        for field in ('name','address'):
            distribution=value[field+'_scripts']
            presence=Counter()
            for label,count in distribution.items():
                for script in label.split('+'):presence[script]+=count
            for required in ('Latin','Devanagari','Arabic','Cyrillic','CJK'):presence.setdefault(required,0)
            value[field+'_script_presence']=dict(presence)
            value['mixed_'+field+'_records']=sum(v for k,v in distribution.items() if '+' in k)
    out.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=["# Full dataset EDA", "", "All six source TSV files were scanned; original files were not modified.",
           "Script names describe Unicode writing systems, not inferred spoken languages. Mixed scripts contain `+`.",
           "", "| File | Rows | Unique IDs | Duplicate IDs | Duplicate rows |", "|---|---:|---:|---:|---:|"]
    for key,v in results.items():
        if key=="ground_truth":continue
        lines.append(f"| {key} | {v['rows']:,} | {v['unique_entity_ids']:,} | {v['duplicate_entity_ids']} | {v['duplicate_full_rows']} |")
    for key,v in results.items():
        lines.extend(["",f"## {key}","", "```json",json.dumps(v,ensure_ascii=False,indent=2),"```"])
    lines += ["", "Paired script variation and error examples are measured separately on the held-out sample in error_analysis.md."]
    (root/"reports/eda_report.md").write_text("\n".join(lines),encoding="utf-8")
    return results
