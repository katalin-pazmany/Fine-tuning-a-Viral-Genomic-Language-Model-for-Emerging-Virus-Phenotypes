import pandas as pd
import numpy as np
import json
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

cache = pd.read_csv("outputs/accession_taxid_cache.csv")
acc_to_taxid = dict(zip(cache["accession"], cache["taxid"].astype(int)))
meta = pd.read_csv("outputs/refseq_metadata.csv")
meta["virus_taxid"] = meta["accession"].map(acc_to_taxid)
meta = meta.dropna(subset=["virus_taxid"])
meta["virus_taxid"] = meta["virus_taxid"].astype(int)

npz_data = np.load("outputs/refseq_embeddings.npz")
embs = npz_data["embeddings"]
accs = npz_data["accessions"]
acc_to_idx = {a:i for i,a in enumerate(accs)}

h2h_df = pd.read_excel("data/complete_h2h_labels.xlsx")
zoo_df = pd.read_excel("data/complete_zoonotic_labels.xlsx")
df = h2h_df.merge(zoo_df[["virus_taxid","zoonotic"]], on="virus_taxid")
df = df.dropna(subset=["virus_family"])
df["virus_family"] = df["virus_family"].astype(str)

matched = df[df["virus_taxid"].isin(set(meta["virus_taxid"]))]
taxid_embs = {}
for _, row in meta[meta["virus_taxid"].isin(matched["virus_taxid"])].iterrows():
    tid = row["virus_taxid"]
    idx = acc_to_idx.get(row["accession"])
    if idx is not None:
        taxid_embs.setdefault(tid, []).append(embs[idx])

taxids = list(taxid_embs.keys())
X = np.array([np.mean(taxid_embs[t], axis=0) for t in taxids])
labels = df.set_index("virus_taxid")[["human_to_human","zoonotic","virus_family"]]
taxids_valid = [t for t in taxids if t in labels.index]
X_valid = np.array([X[taxids.index(t)] for t in taxids_valid])
y_h2h = np.array([labels.loc[t,"human_to_human"] for t in taxids_valid])
y_zoo = np.array([labels.loc[t,"zoonotic"] for t in taxids_valid])
families = [str(labels.loc[t,"virus_family"]) for t in taxids_valid]

clf_h2h = Pipeline([("s",StandardScaler()),("c",SVC(kernel="rbf",probability=True,class_weight="balanced",random_state=42))])
clf_zoo = Pipeline([("s",StandardScaler()),("c",SVC(kernel="rbf",probability=True,class_weight="balanced",random_state=42))])
clf_h2h.fit(X_valid, y_h2h)
clf_zoo.fit(X_valid, y_zoo)

proba_h2h = clf_h2h.predict_proba(X_valid)[:,1]
proba_zoo = clf_zoo.predict_proba(X_valid)[:,1]

result = []
for fam in sorted(set(families)):
    mask = np.array([f==fam for f in families])
    if sum(mask) < 3: continue
    result.append({
        "family": fam,
        "n": int(sum(mask)),
        "h2h_true": round(float(np.mean(y_h2h[mask])),3),
        "zoo_true": round(float(np.mean(y_zoo[mask])),3),
        "h2h_pred": round(float(np.mean(proba_h2h[mask])),3),
        "zoo_pred": round(float(np.mean(proba_zoo[mask])),3),
    })

with open("outputs/network_data.json","w") as f:
    json.dump(result, f, indent=2)
print(f"Saved {len(result)} families to outputs/network_data.json")
