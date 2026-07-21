"""
fetch_ncbi_genomes.py
---------------------
Downloads complete viral genome sequences from NCBI Virus.
Saves results as a FASTA file in data/raw/.

Usage:
    python fetch_ncbi_genomes.py

Requirements:
    pip install biopython
"""

from Bio import Entrez, SeqIO
import time
import os

# ------------------------------------------------------------------ #
# CONFIGURATION — edit these before running
# ------------------------------------------------------------------ #

# Your email — NCBI requires this to use their API (they'll contact
# you if your script is causing problems, they won't spam you)
Entrez.email = os.environ.get("NCBI_EMAIL", "your.email@liverpool.ac.uk")
# NCBI API key is read from the environment — never hard-code it.
#   export NCBI_API_KEY="..."   (get one from your NCBI account settings)
Entrez.api_key = os.environ.get("NCBI_API_KEY")


OUTPUT_DIR    = "data/raw"
OUTPUT_FILE   = os.path.join(OUTPUT_DIR, "viral_genomes_raw.fasta")
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "download_progress.txt")  # tracks resume point
 
BATCH_SIZE = 200   # smaller batches = fewer dropped connections
 
SEARCH_QUERY = '"Viruses"[Organism] AND "complete genome"[Title] AND "RefSeq"[Filter]'
 
MAX_RETRIES  = 5   # how many times to retry a failed batch
RETRY_DELAY  = 10  # seconds to wait before retrying
 
# ------------------------------------------------------------------ #
# HELPERS
# ------------------------------------------------------------------ #
 
def save_progress(start):
    """Save the last successfully downloaded batch position."""
    with open(PROGRESS_FILE, "w") as f:
        f.write(str(start))
 
def load_progress():
    """Load resume point if a previous download was interrupted."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            start = int(f.read().strip())
        print(f"Resuming from sequence {start:,}")
        return start
    return 0
 
def fetch_batch_with_retry(web_env, query_key, start, batch_size):
    """Fetch a batch from NCBI, retrying up to MAX_RETRIES times on failure."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            time.sleep(0.4)  # be polite to NCBI
            handle = Entrez.efetch(
                db="nucleotide",
                rettype="fasta",
                retmode="text",
                retstart=start,
                retmax=batch_size,
                webenv=web_env,
                query_key=query_key
            )
            data = handle.read()
            handle.close()
            return data
 
        except Exception as e:
            print(f"  Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                print(f"  Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                raise RuntimeError(f"Failed after {MAX_RETRIES} attempts at position {start}.")
 
# ------------------------------------------------------------------ #
# MAIN FUNCTIONS
# ------------------------------------------------------------------ #
 
def search_ncbi(query):
    print(f"Searching NCBI for: {query}")
    handle = Entrez.esearch(db="nucleotide", term=query, usehistory="y")
    results = Entrez.read(handle)
    handle.close()
 
    total     = int(results["Count"])
    web_env   = results["WebEnv"]
    query_key = results["QueryKey"]
    print(f"Found {total:,} sequences")
    return total, web_env, query_key
 
 
def download_sequences(total, web_env, query_key):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
 
    # Resume from last saved position if interrupted
    resume_from = load_progress()
 
    # Append if resuming, write fresh if starting over
    mode = "a" if resume_from > 0 else "w"
 
    with open(OUTPUT_FILE, mode) as out:
        for start in range(resume_from, total, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total)
            print(f"Downloading sequences {start + 1}–{end} of {total:,}...")
 
            data = fetch_batch_with_retry(web_env, query_key, start, BATCH_SIZE)
            out.write(data)
            out.flush()          # write to disk immediately
            save_progress(end)   # save progress after every batch
 
    # Download complete — remove progress file
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
 
    print(f"\nDone! Saved to: {OUTPUT_FILE}")
 
 
def count_sequences(fasta_file):
    count = sum(1 for _ in SeqIO.parse(fasta_file, "fasta"))
    print(f"Total sequences in file: {count:,}")
    return count
 
 
# ------------------------------------------------------------------ #
# RUN
# ------------------------------------------------------------------ #
 
if __name__ == "__main__":
    print("=" * 50)
    print("NCBI Viral Genome Downloader")
    print("=" * 50)
 
    total, web_env, query_key = search_ncbi(SEARCH_QUERY)
 
    if total > 50000:
        print(f"WARNING: capping at 15,000 for now.")
        total = 15000
 
    download_sequences(total, web_env, query_key)
    count_sequences(OUTPUT_FILE)
 