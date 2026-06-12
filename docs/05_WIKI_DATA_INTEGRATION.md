# Wiki data integration notes (Ana)

How the wikipedia corpus gets into the training pipeline. Script:
src/prepare_wiki_pairs.py, run with `python3 -m src.prepare_wiki_pairs`
from the project root. Uses seed 42 so results are the same every run.

## Input

data/wiki/wiki_chunks.json comes from my scraping pipeline (separate repo,
see DATA_NOTES.txt there). 21698 passages of about 250 words from 4900
wikipedia articles (NLP, ML, linguistics, IR, security categories).
Each record: {id, content, word_count, source_title}.

## Cleaning

Found some problems in my data while integrating, the script drops:

1. duplicate chunks (580). Wikipedia has mirror articles with identical
   text (15.ai vs 15.dev for example). Bad for contrastive training because
   a sampled negative could be word for word the same as the positive.
2. off-topic titles (217 chunks). My keyword whitelist matches common words
   like "statistic" so things like censuses and songs slipped through.
   Also kills useless titles like ".ai" that make no sense as a query.
3. reference list chunks (682). Turns out my "== References ==" stripping
   never worked because the wikipedia-api library removes the == marks from
   page.text. So some chunks were just ISBN/doi citation lists. Detected
   by counting citation markers in the text.
4. the 500 articles from wiki_evaluation_set.csv (my holdout set), excluded
   from training completely.

Left after cleaning: 19719 chunks from 4235 articles.

## Pairs

Positive: query = article title, document = a chunk of that article (max 2
chunks per article). This title->body trick is the standard weak supervision
setup from the DPR and Contriever papers. For the second chunk of an article
the title gets rewritten as a question ("What is X?") so the query style is
closer to the question style queries on the jurafsky side.

Negative: random chunk from a different article. Same idea as the
"20 chunks apart" rule in the jurafsky pipeline, just adapted for wikipedia.

## Split

By article, 90/10. All chunks of an article go to train OR val, never both,
so val actually measures unseen topics. The script asserts there is no
overlap between train, val and the eval holdout.

## Merge

Only 570 jurafsky train pairs exist vs 6527 wiki pairs, and the final demo
searches the jurafsky book. So the jurafsky pairs are repeated 3x in the
combined file, otherwise the in-domain data gets drowned out (~21% of the
combined set is in-domain this way).

Output files in data/processed/:
- wiki_train_pairs.json (6527), wiki_val_pairs.json (745) - wiki only
- train_pairs_combined.json (8237), val_pairs_combined.json (809)

Schema is the same as train_pairs.json:
{query, positive_id, positive_text, negative_id, negative_text}
so InBatchDataset and TripletDataset work without any changes.

## Training on it

In 02_training.ipynb point the loader at the combined files:

    train_pairs = load_json('data/processed/train_pairs_combined.json')
    val_pairs   = load_json('data/processed/val_pairs_combined.json')

Heads up: 8237 pairs is about 14x more batches per epoch than before, so
lower EPOCHS (2 instead of 10), otherwise it takes hours on colab.
