import json
import csv
import re
import random
from collections import defaultdict

WIKI_CHUNKS = "data/wiki/wiki_chunks.json"
WIKI_EVAL = "data/wiki/wiki_evaluation_set.csv"
JUR_TRAIN = "data/processed/train_pairs.json"
JUR_VAL = "data/processed/val_pairs.json"
OUT_DIR = "data/processed"

SEED = 42
PAIRS_PER_ARTICLE = 2
VAL_FRAC = 0.10
JUR_REPEAT = 3

BAD_TITLE = re.compile(
    r"census|\(song\)|\(album\)|\(film\)|\(TV series\)|\(band\)|\(singer\)"
    r"|\(rapper\)|\(surgeon\)|\(politician\)|Bilderberg|Conference|passport|^\d{4}",
    re.IGNORECASE)

REF_MARKS = ["ISBN", "Retrieved", "doi:", "doi.org", "(ed.)", "(eds.)", "Archived from"]

TEMPLATES = [
    "What is {}?",
    "How does {} work?",
    "Explain {}.",
    "Tell me about {}.",
]


def looks_like_references(text):
    hits = 0
    for m in REF_MARKS:
        hits += text.count(m)
    return hits >= 2


def title_ok(title):
    if BAD_TITLE.search(title):
        return False
    letters = sum(1 for ch in title if ch.isalpha())
    return letters >= 3


def make_pairs(by_article, rng):
    titles = sorted(by_article.keys())
    pool = []
    for t in titles:
        for c in by_article[t]:
            pool.append((t, c))

    pairs = []
    for title in titles:
        chunks = by_article[title]
        n = min(PAIRS_PER_ARTICLE, len(chunks))
        picked = rng.sample(chunks, n)
        for k in range(n):
            if k == 0:
                query = title
            else:
                query = rng.choice(TEMPLATES).format(title)
            while True:
                neg_title, neg = rng.choice(pool)
                if neg_title != title:
                    break
            pairs.append({
                "query": query,
                "positive_id": picked[k]["id"],
                "positive_text": picked[k]["content"],
                "negative_id": neg["id"],
                "negative_text": neg["content"],
            })
    return pairs


rng = random.Random(SEED)
eval_titles = set()
with open(WIKI_EVAL, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        eval_titles.add(row["query"].strip())
print("eval holdout articles:", len(eval_titles))

with open(WIKI_CHUNKS, encoding="utf-8") as f:
    raw = json.load(f)

by_article = defaultdict(list)
seen = set()
n_dupe = n_bad = n_refs = n_eval = 0
for c in raw:
    title = c["source_title"].strip()
    if title in eval_titles:
        n_eval += 1
        continue
    if not title_ok(title):
        n_bad += 1
        continue
    content = " ".join(c["content"].split())
    if looks_like_references(content):
        n_refs += 1
        continue
    key = content[:500]
    if key in seen:
        n_dupe += 1
        continue
    seen.add(key)
    by_article[title].append({"id": c["id"], "content": content})

print("chunks in:", len(raw))
print("dropped: dupes", n_dupe, "/ bad title", n_bad, "/ references", n_refs, "/ eval", n_eval)
print("kept:", sum(len(v) for v in by_article.values()), "chunks,", len(by_article), "articles")

titles = sorted(by_article.keys())
rng.shuffle(titles)
n_val = int(len(titles) * VAL_FRAC)
val_articles = {t: by_article[t] for t in titles[:n_val]}
train_articles = {t: by_article[t] for t in titles[n_val:]}
print("articles: train", len(train_articles), "val", len(val_articles))

wiki_train = make_pairs(train_articles, rng)
wiki_val = make_pairs(val_articles, rng)
print("wiki pairs: train", len(wiki_train), "val", len(wiki_val))

for p in wiki_train + wiki_val:
    assert p["positive_id"] != p["negative_id"]
    assert p["positive_text"] != p["negative_text"]
assert not set(train_articles) & set(val_articles)
assert not set(train_articles) & eval_titles
assert not set(val_articles) & eval_titles
print("sanity checks ok")

with open(OUT_DIR + "/wiki_train_pairs.json", "w", encoding="utf-8") as f:
    json.dump(wiki_train, f, indent=2, ensure_ascii=False)
with open(OUT_DIR + "/wiki_val_pairs.json", "w", encoding="utf-8") as f:
    json.dump(wiki_val, f, indent=2, ensure_ascii=False)

with open(JUR_TRAIN, encoding="utf-8") as f:
    jur_train = json.load(f)
with open(JUR_VAL, encoding="utf-8") as f:
    jur_val = json.load(f)

combined_train = wiki_train + jur_train * JUR_REPEAT
combined_val = wiki_val + jur_val
rng.shuffle(combined_train)
rng.shuffle(combined_val)

with open(OUT_DIR + "/train_pairs_combined.json", "w", encoding="utf-8") as f:
    json.dump(combined_train, f, indent=2, ensure_ascii=False)
with open(OUT_DIR + "/val_pairs_combined.json", "w", encoding="utf-8") as f:
    json.dump(combined_val, f, indent=2, ensure_ascii=False)

print("combined train:", len(combined_train), "( wiki", len(wiki_train),
      "+ jurafsky", len(jur_train), "x", JUR_REPEAT, ")")
print("combined val:", len(combined_val))
print("wrote 4 files to", OUT_DIR)
