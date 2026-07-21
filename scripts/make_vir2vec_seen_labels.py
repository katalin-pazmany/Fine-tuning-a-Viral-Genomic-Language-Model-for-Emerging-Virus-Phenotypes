"""
Label each phenotype taxon as SEEN or UNSEEN by Vir2vec's pre-training corpus
============================================================================
Dissertation: Fine-tuning a Viral Genomic Language Model for Emerging Virus Phenotypes
Author: Katalin Pazmany

Vir2vec was pre-trained on a curated corpus of 565,747 genomes spanning ~295 viral
species. The exact accession lists are published in the Vir2vec GitHub repo
(simoRancati/Vir2vec, accessions_txt/{Train,Test,Validation}/*), split by source:
  * NCBI_virus   — entries are "<taxid>.<genome>"  -> gives the taxids Vir2vec saw
  * GISAID / HBVdb / LANL-HIV-DB — GenBank accessions -> exact genomes Vir2vec saw
  * BV-BRC       — GISAID EPI_ISL isolate ids (influenza / SARS), not matched here

A phenotype taxon is labelled SEEN if either
  (a) its NCBI species-level taxid matches the species of any Vir2vec NCBI_virus
      taxid (species mapping via the NCBI taxonomy dump, so a different strain of
      the same species still counts as seen — the conservative choice), or
  (b) any of its GenBank accessions appears in Vir2vec's accession lists.
Otherwise it is UNSEEN — genuinely absent from the model's pre-training, and thus a
fair out-of-distribution test for anticipating novel/emerging viruses.

This produces outputs/vir2vec_seen_labels.csv (virus_taxid, vir2vec_seen).

Prerequisites (network): downloads the Vir2vec accession lists from GitHub and the
NCBI taxonomy dump. Run once; the resulting CSV is what the analysis consumes.

Usage:
    conda activate viral-phenotype
    python scripts/make_vir2vec_seen_labels.py
"""

import csv
import io
import tarfile
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
OUTPUTS = BASE_DIR / "outputs"
CACHE = OUTPUTS / "accession_taxid_cache.csv"
LABELS = BASE_DIR / "data" / "phenotype_labels_with_strict_zoonotic.csv"
OUT = OUTPUTS / "vir2vec_seen_labels.csv"
WORK = OUTPUTS / "_vir2vec_corpus"      # cached downloads
WORK.mkdir(parents=True, exist_ok=True)

GH = "https://raw.githubusercontent.com/simoRancati/Vir2vec/main/accessions_txt"
SPLITS = {"Train": "train", "Test": "test", "Validation": "val"}
SOURCES = ["BV-BRC", "GISAID", "HBVdb", "LANL-HIV-DB", "NCBI_virus"]
TAXDUMP_URL = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz"


def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        print(f"  downloading {url}")
        urllib.request.urlretrieve(url, dest)
    return dest


# ── 1. Vir2vec accession lists ────────────────────────────────────────────────
print("Fetching Vir2vec accession lists...")
seen_strain_taxids, seen_accessions = set(), set()
for split, pfx in SPLITS.items():
    for src in SOURCES:
        dest = WORK / f"{pfx}__{src}.txt"
        try:
            fetch(f"{GH}/{split}/{pfx}__{src}.txt", dest)
        except Exception as e:
            print(f"  ! skip {split}/{src}: {e}")
            continue
        for line in dest.read_text().splitlines():
            tok = line.strip()
            if not tok:
                continue
            if src == "NCBI_virus":
                head = tok.split(".")[0]
                if head.isdigit():
                    seen_strain_taxids.add(int(head))
            elif src in ("GISAID", "HBVdb", "LANL-HIV-DB"):
                seen_accessions.add(tok.split(".")[0].upper())
print(f"  {len(seen_strain_taxids):,} NCBI taxids, {len(seen_accessions):,} accessions")

# ── 2. NCBI taxonomy: strain taxid -> species taxid ───────────────────────────
print("Fetching NCBI taxonomy dump...")
tarpath = fetch(TAXDUMP_URL, WORK / "taxdump.tar.gz")
parent, rank, merged = {}, {}, {}
with tarfile.open(tarpath) as tf:
    for name, store in (("nodes.dmp", "nodes"), ("merged.dmp", "merged")):
        f = io.TextIOWrapper(tf.extractfile(name))
        for line in f:
            p = [x.strip() for x in line.split("|")]
            if store == "nodes":
                parent[int(p[0])], rank[int(p[0])] = int(p[1]), p[2]
            else:
                merged[int(p[0])] = int(p[1])


def to_species(t):
    t = merged.get(t, t)
    for _ in range(40):
        if rank.get(t) == "species":
            return t
        if t not in parent or parent[t] == t:
            return None
        t = parent[t]
    return None


seen_species = {s for s in (to_species(t) for t in seen_strain_taxids) if s}
print(f"  {len(seen_species):,} species covered by Vir2vec")

# ── 3. Katalin's taxon -> accessions, then classify ───────────────────────────
taxid_accs = {}
for r in csv.DictReader(open(CACHE)):
    try:
        t = int(r["taxid"])
    except ValueError:
        continue
    taxid_accs.setdefault(t, set()).add(r["accession"].split(".")[0].upper())


def is_seen(tid: int) -> bool:
    sp = to_species(tid)
    if sp is not None and sp in seen_species:
        return True
    if tid in seen_strain_taxids:
        return True
    return any(a in seen_accessions for a in taxid_accs.get(tid, ()))


rows = list(csv.DictReader(open(LABELS)))
with open(OUT, "w", newline="") as fo:
    w = csv.writer(fo)
    w.writerow(["virus_taxid", "vir2vec_seen"])
    n_seen = 0
    for r in rows:
        s = int(is_seen(int(r["virus_taxid"])))
        n_seen += s
        w.writerow([r["virus_taxid"], s])

print(f"\nSaved {OUT}")
print(f"  {len(rows)} taxa: {n_seen} seen, {len(rows) - n_seen} unseen "
      f"({(len(rows) - n_seen) / len(rows) * 100:.0f}% unseen)")
