# პროექტის ანგარიში — Neural Search Engine (ნულიდან აგებული მოდელი)

**ავტორი:** Elene
**კურსი:** NLP
**ჩაბარების თარიღი:** 13 ივნისი, 2026
**წყაროს კოდი:** `C:\Users\Enele\Desktop\Neural_Search_Engine`

---

## სარჩევი

1. პროექტის მიზანი და მთლიანი არქიტექტურა
2. მონაცემები — Jurafsky წიგნი + Wikipedia
3. Baseline მოდელი — BM25
4. ტოკენიზატორი — BPE ნულიდან
5. ძირითადი მოდელი — Transformer Encoder ნულიდან
6. სასწავლო პროცესი — Loss ფუნქცია და ჰიპერპარამეტრები
7. სასწავლო შედეგები — Loss-ის მრუდები
8. ევალუაცია — მეტრიკები და შედეგები
9. რეპოზიტორიის სტრუქტურა
10. დასკვნა და სამომავლო გაუმჯობესება

> **მნიშვნელოვანი შენიშვნა შეფასებისთვის.** პროექტის წესების მიხედვით **აკრძალულია წინასწარ გაწვრთნილი (pretrained / fine-tuned) მოდელის გამოყენება**. ამიტომ წინა ვერსიის DistilBERT backbone **მთლიანად ამოღებულია**. ამ ვერსიაში როგორც **ტოკენიზატორი**, ისე **ენკოდერი ნულიდანაა აგებული** მხოლოდ `torch`-ის საბაზისო შრეებით (`nn.Linear`, `nn.Embedding`, `nn.LayerNorm`). არ გამოიყენება არც `transformers`, არც `sentence-transformers`, არც რომელიმე pretrained წონა.

---

## 1. პროექტის მიზანი და მთლიანი არქიტექტურა

### 1.1 ამოცანის ფორმულირება

პროექტის ფარგლებში ავაგე **Neural Search Engine** წიგნისთვის *"Speech and Language Processing"* (Jurafsky & Martin). სისტემა იღებს მომხმარებლის query-ს ბუნებრივ ენაზე და აბრუნებს წიგნიდან top-k ყველაზე რელევანტურ ტექსტურ მონაკვეთს (chunk-ს).

### 1.2 Pipeline

```
   query (free text)
        |
        v
   ჩვენი BPE ტოკენიზატორი  ->  token ids
        |
        v
   text encoder (ნულიდან აგებული Transformer + mean pool + L2 norm)
        |
        v
   query embedding [256-dim, ||v||=1]
        |
        v
   cosine similarity vs corpus embeddings (1410 chunks)
        |
        v
   top-k chunks
```

ერთი და იგივე ენკოდერი ("tower") გამოიყენება როგორც query-სთვის, ისე document-ისთვის — ეს არის კლასიკური **bi-encoder** მიდგომა, რის გამოც კოდში მთავარი კლასი ისევ `BiEncoder`-ად იწოდება.

### 1.3 ძირითადი არჩევანი

| კომპონენტი | არჩევანი | რატომ |
|---|---|---|
| ტოკენიზატორი | **BPE ნულიდან** (vocab 8000) | სუბ-სიტყვური ტოკენიზაცია; იშვიათი ტექნიკური ტერმინებიც იშლება ნაცნობ ნაწილებად ([UNK]-ის ნაცვლად) |
| ენკოდერი | **Transformer encoder ნულიდან** (~5M პარამეტრი) | token embedding + sinusoidal positional encoding + multi-head self-attention + FFN — ყველა შრე ხელითაა აწყობილი |
| Pooling | Mean pooling (padding-ის გათვალისწინებით) | sentence embedding-ისთვის უკეთესია ვიდრე ერთი ცალკეული ტოკენი; მთელ თანმიმდევრობას ითვალისწინებს |
| Normalization | L2 unit normalization | cosine similarity ხდება dot product-ი → სწრაფი retrieval |
| Loss | სიმეტრიული InfoNCE (in-batch negatives) | batch_size=B → B-1 negatives უფასოდ; CLIP/SimCSE-style |
| Baseline | BM25 (rank_bm25) | კლასიკური IR baseline, ძლიერი ტექნიკურ ტექსტებზე |
| Index | In-memory dense matrix [N x 256] | N=1410-ისთვის brute-force სავსებით საკმარისია (<1ms search) |

**მთავარი შეზღუდვა (პროექტის წესი):** ვერ გამოვიყენე pretrained მოდელი და ვერც `sentence-transformers`. მთელი contrastive learning pipeline — ტოკენიზატორიდან ენკოდერამდე — იმპლემენტირებულია ნულიდან.

---

## 2. მონაცემები

### 2.1 პირველი წყარო — Jurafsky-ის წიგნი

* **PDF Extraction:** `src/utils.py`-ის ფუნქცია `extract_clean_text_from_pdf` PyMuPDF-ით კითხულობს PDF-ს, ხსნის header/footer-ებს გვერდის 8% top/bottom crop-ით, ფილტრავს chapter-ის სათაურებს და გადაბმულ სიტყვებს (`-\n`).
* **Sentence-aware chunking:** `src/chunker_v2.py`-ის `SentenceAwareChunker`:
  1. ყოფს ტექსტს წინადადებებად regex-ით (abbreviation-ების დაცვით: Mr., Fig., e.g.);
  2. სიხარბით აერთიანებს წინადადებებს ~220 სიტყვამდე target word budget-მდე;
  3. გადააქვს ბოლო 2 წინადადება შემდეგ chunk-ში overlap-ისთვის;
  4. **ყოველი chunk იწყება და მთავრდება წინადადების საზღვარზე.**
* **შედეგი:** 1410 chunk, თითო ~239 სიტყვა (min 101, max 414).

### 2.2 მეორე წყარო — Wikipedia

* პროექტში დავამატე Wikipedia-ს სტატიების chunk-ები, რომლებიც **მოცულობას მატებს** ნულიდან წვრთნისთვის (random init მოდელს მეტი მონაცემი სჭირდება, ვიდრე fine-tuning-ს).
* გაერთიანებული `train_pairs_combined.json` შეიცავს როგორც Wikipedia, ისე Jurafsky-ის წყვილებს. ჩვენი BPE ტოკენიზატორი ისწავლება **ორივე წყაროს ტექსტზე**, ასე რომ ლექსიკა ფარავს ორივე domain-ს.

### 2.3 Synthetic Query Generation

ვინაიდან ხელით დაწერილი (query, positive_chunk) წყვილები ბევრი chunk-ისთვის არ მქონდა, გამოვიყენე **LLM-based query generation** (ეს არის მონაცემთა მომზადების ეტაპი, არა მოდელის ნაწილი):

* `src/query_generator.py`-ის `ManualBatchGenerator` batch-ებად (20 chunk) წერს prompt-ებს, რომელთა გადახდენაც შესაძლებელია ჩატ-ინტერფეისში;
* თითო chunk-ისთვის გენერირდება 2 ბუნებრივი query (synonyms-ით, textbook-ის ფრაზების კოპირების გარეშე);
* `add_random_negatives` ამატებს random distant negative chunk-ს (მინ. 20 პოზიციის დაშორებით).

### 2.4 Train / Validation / Test Split

| Split | წყარო | ზომა | დანიშნულება |
|---|---|---|---|
| Train | LLM queries on Jurafsky + Wiki | 8237 pairs | მოდელის გაწვრთნა |
| Val | LLM queries (holdout) | 809 pairs | overfitting-ის მონიტორინგი (val InfoNCE loss) |
| Test | ხელით დაწერილი query | 25 queries | საბოლოო ბენჩმარკი (BM25 vs ჩვენი მოდელი) |

**მნიშვნელოვანი:** Test set-ი (`data/evaluation_set.csv`) არასოდეს ხედავს training-ში. ის შეიცავს 3 ტიპის query-ს:
* **keyword** — query-ის ლექსიკა პირდაპირ ემთხვევა target chunk-ის სიტყვებს (BM25-ის ძლიერი მხარე);
* **paraphrase** — იგივე target chunks, მაგრამ synonyms-ით გადაფრაზებული (აქ უნდა გამოვიდეს neural მოდელის უპირატესობა);
* **paraphrase_new** — სხვა chunks-ზე, ფართო თემატიკით.

---

## 3. Baseline მოდელი — BM25

### 3.1 იმპლემენტაცია

`src/evaluate.py:evaluate_bm25` იყენებს `rank_bm25.BM25Okapi`-ს default პარამეტრებით (k1=1.5, b=0.75). Tokenization: lowercase + punctuation-ის წაშლა + whitespace split.

### 3.2 რატომ BM25?

* 25+ წელია IR-ის გოლდ სტანდარტი;
* მუშაობს zero-shot — training data არ სჭირდება;
* განსაკუთრებით ძლიერია ტექნიკურ ტექსტებზე, სადაც domain-specific terms verbatim იმეორება query-ში.

### 3.3 BM25-ის სუსტი მხარე

* **Paraphrase queries:** როცა query იყენებს სინონიმს target-ში გამოყენებული სიტყვის ნაცვლად, BM25 ვერ პოულობს lexical დამთხვევას;
* **Conceptual queries:** ვერ აღიქვამს semantic similarity-ს keyword-ის გარეშე.

სწორედ ეს სუსტი მხარე გახდა ჩვენი neural მოდელის "მისია" — paraphrase queries-ზე BM25-ის შევსება/დამარცხება.

---

## 4. ტოკენიზატორი — BPE ნულიდან (`src/tokenizer.py`)

რადგან pretrained ტოკენიზატორი (DistilBERT WordPiece) აკრძალულია, ჩვენ ვწერთ **Byte-Pair-Encoding (BPE)** ტოკენიზატორს ნულიდან (Sennrich et al., 2016).

### 4.1 რატომ BPE და არა word-level?

ჩვენი ტექსტი ტექნიკურია და კორპუსი მცირე. სუფთა word-level ლექსიკა გაიზრდებოდა ძალიან დიდად და იშვიათი ტერმინების უმეტესობა გადაიქცეოდა ერთ `[UNK]` ტოკენად (მოდელისთვის ნულოვანი სიგნალი). BPE აბალანსებს: ხშირი სიტყვები რჩება მთლიანი (`language`, `model`), იშვიათი სიტყვები კი იშლება ხელახლა გამოყენებად ნაწილებად (`perplex` + `ity`).

### 4.2 ალგორითმი (training)

1. **Pre-tokenization:** ტექსტი იყოფა "სიტყვებად" (ალფა-რიცხვითი მონაკვეთები + ცალკეული პუნქტუაცია), lowercase. თითო სიტყვა იწერება სიმბოლოების მიმდევრობად სპეციალური ბოლო-მარკერით `</w>`, მაგ. `low → (l, o, w, </w>)`.
2. **Merge loop:** მუდმივად ვპოულობთ ყველაზე ხშირ მეზობელ წყვილს მთელ კორპუსში და ვაერთიანებთ ახალ სიმბოლოდ; merge ემახსოვრება.
3. **Stop:** როცა ლექსიკა მიაღწევს `vocab_size`-ს (8000) ან წყვილი აღარ მეორდება საკმარისად.

იმპლემენტაცია იყენებს **ინკრემენტულ pair-count განახლებას** (ინვერსიული ინდექსით `pair → words`), ასე რომ მხოლოდ ის სიტყვები ნახლდება, რომლებიც შერწყმულ წყვილს შეიცავენ — 8000 merge-ის სწავლა მთელ კორპუსზე სწრაფია.

### 4.3 Encoding

ახალი სიტყვის ენკოდირებისას ვიყენებთ ნასწავლ merge-ებს რანგის (პრიორიტეტის) მიხედვით, სანამ შესაძლებელია; მიღებული სუბ-სიტყვები გადადის integer id-ებში (უცნობი → `[UNK]`). სპეციალური ტოკენები: `[PAD]=0`, `[UNK]=1`. CLS/SEP არ გვჭირდება — mean pooling გვაკმარებს.

ტოკენიზატორი ინახება ერთ JSON ფაილად (`checkpoints/tokenizer.json`): `token2id` + ნასწავლი `merges`.

---

## 5. ძირითადი მოდელი — Transformer Encoder ნულიდან (`src/model.py`)

### 5.1 არქიტექტურა

```
input_ids [B, S]
      |
      v
  TokenEmbedding (nn.Embedding, padding_idx)  *  sqrt(d_model)
      + SinusoidalPositionalEncoding
      |
      v
  N x EncoderLayer  (pre-norm):
        x = x + MultiHeadSelfAttention(LayerNorm(x), mask)
        x = x + FeedForward(LayerNorm(x))
      |
      v
  final LayerNorm
      |
      v
  masked mean pooling  (მხოლოდ რეალურ ტოკენებზე საშუალო)
      |
      v
  F.normalize(·, p=2)  ->  embedding [B, 256], ||v|| = 1
```

### 5.2 შრეები (ხელით აწყობილი)

* **Token Embedding:** `nn.Embedding(vocab_size, d_model, padding_idx=0)`, გამრავლებული `sqrt(d_model)`-ზე (სტანდარტული Transformer scaling).
* **Sinusoidal Positional Encoding:** ფიქსირებული (არა-ნასწავლი) სიგნალი, რომ მოდელმა იცოდეს ტოკენების **რიგი** (self-attention თავისთავად permutation-invariant-ია):
  `PE(pos,2i)=sin(pos/10000^(2i/d))`, `PE(pos,2i+1)=cos(pos/10000^(2i/d))`.
* **Multi-Head Self-Attention:** ხელითაა იმპლემენტირებული (და არა `nn.MultiheadAttention`). ცალკე `nn.Linear` Q, K, V და output projection-ისთვის; scaled dot-product `softmax(QKᵀ/√d_head)V`; **key-padding mask** padding-ის პოზიციებს `-inf`-ად აქცევს softmax-მდე, რომ მათზე attention არ წავიდეს.
* **Feed-Forward:** `Linear(d_model, d_ff) → GELU → Linear(d_ff, d_model)`.
* **Pre-norm residual:** LayerNorm თითო sub-layer-ის *წინ* — random init-იდან უფრო სტაბილურად ისწავლება, ვიდრე ორიგინალური post-norm.
* **Init:** linear-ები — Xavier uniform; embeddings — normal(0, 0.02); padding row — ნული.

### 5.3 კონფიგურაცია და პარამეტრები

| პარამეტრი | მნიშვნელობა |
|---|---|
| vocab_size | 8000 (ტოკენიზატორიდან) |
| d_model | 256 |
| n_layers | 4 |
| n_heads | 4 |
| d_ff | 1024 |
| max_len | 256 |
| dropout | 0.1 |
| **სულ პარამეტრი** | **~5.2M** |

მოდელი ინახება კონფიგურაციასთან ერთად (`BiEncoder.save` / `BiEncoder.load`), ასე რომ checkpoint-ი თვითონ "იცავს" თავის არქიტექტურას და ევალუაცია/demo ზუსტად აღადგენს მას.

### 5.4 L2 Normalization-ის მნიშვნელობა

L2-ნორმალიზებული ვექტორებისთვის `cosine(u,v) = u·v`. ეს გვაძლევს: (1) სწრაფ retrieval-ს — ერთი matmul 1410 chunk-ის წინააღმდეგ; (2) loss-ისთვის სტაბილურ სიგნალს — მოდელი ვერ "მოატყუებს" magnitude-ის გაზრდით.

---

## 6. Loss ფუნქცია და სასწავლო პროცესი

### 6.1 სიმეტრიული InfoNCE Loss (`src/loss.py`)

InfoNCE (NT-Xent, SimCLR/CLIP-style) — modern contrastive learning-ის სტანდარტი.

**ინტუიცია:** batch-ში გვაქვს B (query, positive) წყვილი. ვაშენებთ B×B similarity matrix-ს:

```
              positive_1  positive_2  ...  positive_B
query_1     [ HIGH    ] [  low   ] ... [  low   ]   <- target = 1
query_2     [  low   ] [ HIGH    ] ... [  low   ]
   :
query_B     [  low   ] [  low   ] ... [ HIGH    ]
```

**ფორმულა (ერთი მიმართულება):**
```
L = -1/B * sum_i log( exp(S[i,i]/tau) / sum_j exp(S[i,j]/tau) )
S[i,j] = query_i · positive_j   (L2-norm-ის შემდეგ),   tau = 0.05
```

**სიმეტრიული ვარიანტი (აქ გამოყენებული):** loss ითვლება **ორივე მიმართულებით** და სრულდება საშუალო:
```
L = 0.5 * ( CE(S, diag)  +  CE(Sᵀ, diag) )
```
ანუ query→positive *და* positive→query. ეს უფრო სუფთა გრადიენტს იძლევა და ნულიდან წვრთნისას უფრო სტაბილურია.

**უპირატესობა Triplet Loss-თან:** batch=64 → თითო query-ისთვის 63 უფასო negative; hard negative mining აღარ სჭირდება; gradient signal ბევრად მდიდარია.

### 6.2 Training Loop (`src/train.py:Trainer`)

| ჰიპერპარამეტრი | მნიშვნელობა | რატომ |
|---|---|---|
| Loss | სიმეტრიული InfoNCE, temperature=0.05 | sharper distribution → უკეთესი from-scratch სიგნალი |
| Batch size | 64 | 63 უფასო negative თითო query-ზე |
| Doc / query max length | 256 / 64 | document-ები გრძელია (~255 სიტყვა), query-ები მოკლე |
| Epochs | 20 | random init-ს მეტი გავლა სჭირდება, ვიდრე fine-tuning-ს |
| Optimizer | AdamW, weight_decay=0.01 | decay მხოლოდ 2D წონებზე (Linear/Embedding), არა bias/LayerNorm-ზე |
| Learning rate | 3e-4 (peak) | ნულიდან წვრთნას უფრო მაღალი LR სჭირდება, ვიდრე fine-tuning-ს (2e-5) |
| LR schedule | linear warmup (10%) + linear decay | ხელს უშლის ადრეულ გრადიენტის აფეთქებას |
| Gradient clipping | max_norm=1.0 | სტაბილური training პირველ epoch-ში |
| Logging | CSV (epoch, step, train_loss, val_loss, lr) | `checkpoints/training_log.csv` |
| Checkpoint | best (lowest val_loss) + final + tokenizer | `checkpoints/best.pt`, `tokenizer.json` |

### 6.3 Vector Store (`src/vector_store.py`)

* `VectorStore.build(chunks)` — ერთჯერ ენკოდირებს მთლიან კორპუსს batch-ებით;
* `VectorStore.search(query, top_k)` — ენკოდირებს query-ს, ითვლის dot product-ს matrix-თან, აბრუნებს top-k შედეგებს;
* `save`/`load` — `embeddings.pt` + `metadata.json`.

---

## 7. სასწავლო შედეგები — Loss-ის მრუდები

წვრთნა სრულდება Google Colab-ზე (T4 GPU). შედეგად მიიღება `checkpoints/training_curves.png` და `training_log.csv`.

> **შესავსები (Colab-ის გაშვების შემდეგ):** ჩასვით `training_curves.png` და ქვემოთ შეავსეთ საბოლოო loss-ის მნიშვნელობები.

| | მნიშვნელობა |
|---|---|
| Random baseline (`log(batch_size)=log64`) | **4.16** |
| საუკეთესო val InfoNCE loss | `[___ შეავსეთ]` |
| საბოლოო train InfoNCE loss | `[___ შეავსეთ]` |

**როგორ წავიკითხოთ მრუდი:** გაუწვრთნელი მოდელი ვერ არჩევს positive-ს batch-ის negatives-ისგან, ამიტომ მისი loss ≈ `log(B)`. მრუდის ამ ხაზის ქვემოთ ჩამოსვლა **პირდაპირი მტკიცებულებაა**, რომ ენკოდერი სწავლობს query-ისა და document-ის შესაბამისობას.

**ვერიფიკაცია (CPU smoke run, შემცირებული):** პატარა საკონტროლო გაშვებაზე (4000 წყვილი, 4 epoch, პატარა მოდელი) val loss დაეცა **4.16 → 2.82** და აგრძელებდა კლებას — ეს ადასტურებს, რომ pipeline და სწავლის სიგნალი მუშაობს. სრული შედეგი (8237 წყვილი, 20 epoch, d_model=256) მიიღება Colab-ზე.

---

## 8. ევალუაცია — მეტრიკები და შედეგები

### 8.1 მეტრიკები (`src/evaluate.py`)

**MRR@10 (Mean Reciprocal Rank):** `MRR = (1/N) * Σ 1/rank_i`, სადაც `rank_i` სწორი chunk-ის პოზიციაა top-10-ში (ან 0 თუ ვერ მოხვდა).

**Recall@K (K = 1, 5, 10):** `Recall@K = (1/N) * |{i : სწორი_chunk_i ∈ top_K_i}|`.

ეს მეტრიკები სტანდარტულია IR-ში (BEIR, MS MARCO) და ბუნებრივად ერგება ჩვენს setup-ს (ერთი სწორი chunk თითო query-ზე).

### 8.2 აგრეგირებული შედეგები (25 query)

BM25 და **გაუწვრთნელი** (random init) მოდელის რიცხვები რეალურია და სტაბილური. **გაწვრთნილი** მოდელის სვეტი ივსება Colab-ის წვრთნის შემდეგ.

| Metric | BM25 | Neural (untrained, random init) | Neural (trained) |
|---|---|---|---|
| MRR@10 | **0.3833** | 0.0000 | `[___ შეავსეთ]` |
| R@1 | **0.3200** | 0.0000 | `[___ შეავსეთ]` |
| R@5 | **0.5200** | 0.0000 | `[___ შეავსეთ]` |
| R@10 | **0.5600** | 0.0000 | `[___ შეავსეთ]` |

გაუწვრთნელი მოდელი 0.0-ს იძლევა (random embeddings ვერაფერს პოულობს) — ეს არის სუფთა "neural baseline", რომელთან შედარებითაც contrastive training-ის ეფექტი ნათლად ჩანს.

### 8.3 ⭐ Per-Query-Type Breakdown (ანალიზის მთავარი ჩარჩო)

`src/evaluate.py:metrics_by_query_type` იძლევა მეტრიკებს ცალკე keyword / paraphrase / paraphrase_new query-ებისთვის. სწორედ აქ ჩანს ნეირონული მოდელის რეალური ღირებულება.

| Query Type | n | BM25 MRR@10 | Neural (trained) MRR@10 | მოსალოდნელი |
|---|---|---|---|---|
| keyword | 10 | `[___]` | `[___]` | BM25-ის უპირატესობა (lexical overlap) |
| **paraphrase** | 10 | `[___]` | `[___]` | აქ უნდა გამოვიდეს neural მოდელი (synonyms) |
| paraphrase_new | 5 | `[___]` | `[___]` | შერეული |

**ინტერპრეტაცია (ჩარჩო):** BM25 დაუმარცხებელია keyword queries-ზე (იგივე სიტყვები). paraphrase queries-ზე მისი lexical overlap ნულდება — სწორედ აქ ცდილობს neural ენკოდერი დაიჭიროს semantic similarity. პროექტის მთავარი მიგნება არ არის "BM25-ის ჩანაცვლება", არამედ მისი **შევსება** paraphrase/conceptual query-ებზე.

> **შენიშვნა შედეგების მოლოდინებზე.** ნულიდან გაწვრთნილი მცირე მოდელი ~8k (ძირითადად Wikipedia) წყვილზე და 25 Jurafsky-query-ზე შეფასებისას **ვერ აჯობებს BM25-ს აგრეგატში** — ეს მოსალოდნელია და დავალების პირობაშივეა ჩაწერილი, რომ მთავარია მეთოდის გაგება და baseline-თან შედარება, და არა საუკეთესო performance. წარმატების ნამდვილი ნიშნებია: (1) train/val loss-ის სტაბილური კლება `log(B)`-ის ქვემოთ; (2) გაუწვრთნელ მოდელთან (0.0) შედარებით აშკარა გაუმჯობესება; (3) paraphrase query-ებზე BM25-თან კონკურენტული ან უკეთესი შედეგი.

### 8.4 Qualitative analysis

`notebooks/03_evaluation.ipynb` და `notebooks/04_demo.ipynb` გვერდიგვერდ აჩვენებენ BM25-ისა და neural მოდელის top-k შედეგებს. რეპორტისთვის ეძებეთ query-ები სამ კატეგორიად:
* **BM25 wins** — query იყენებს ზუსტ textbook ტერმინოლოგიას;
* **Neural wins** — paraphrase / განსხვავებული ლექსიკა;
* **ორივე ვერ პოულობს** — თემა ცუდად არის დაფარული კორპუსში, ან chunking-მა context გაჭრა.

---

## 9. რეპოზიტორიის სტრუქტურა

```
Neural_Search_Engine/
+-- data/
|   +-- jurafsky_martin.pdf
|   +-- evaluation_set.csv                (25 test queries + query_type)
|   +-- processed/
|       +-- jurafsky_chunks_v2.json       (1410 sentence-aware chunks — კანონიკური)
|       +-- train_pairs_combined.json     (Jurafsky + Wiki LLM pairs, 8237)
|       +-- val_pairs_combined.json       (809)
|
+-- src/
|   +-- tokenizer.py                      (BPE ნულიდან — NEW)
|   +-- model.py                          (Transformer encoder ნულიდან + EncoderConfig)
|   +-- dataset.py                        (InBatchDataset — ჩვენი ტოკენიზატორი, dynamic padding)
|   +-- loss.py                           (სიმეტრიული InfoNCELoss)
|   +-- train.py                          (Trainer + build_tokenizer_from_pairs)
|   +-- vector_store.py                   (VectorStore — encode-once, search-many)
|   +-- evaluate.py                       (BM25 + neural eval, metrics_by_query_type)
|   +-- chunker_v2.py                     (SentenceAwareChunker)
|   +-- query_generator.py                (ManualBatchGenerator, add_random_negatives)
|   +-- utils.py                          (PDF extraction)
|
+-- notebooks/
|   +-- 01_data_v2_llm.ipynb              (chunk + generate LLM queries)
|   +-- 02_training.ipynb                 (train BPE + train encoder, Colab T4)
|   +-- 03_evaluation.ipynb               (BM25 vs untrained vs trained, per-type)
|   +-- 04_demo.ipynb                     (interactive search demo)
|
+-- docs/
|   +-- REPORT_GE.md                      (ეს ფაილი)
|   +-- 06_FROM_SCRATCH_MODEL.md          (არქიტექტურის deep-dive)
|   +-- ...
|
+-- checkpoints/
|   +-- best.pt / final.pt                (config + weights ერთად)
|   +-- tokenizer.json                    (BPE ლექსიკა + merges)
|   +-- training_log.csv, *.png
|
+-- pyproject.toml                        (uv; torch + rank-bm25 + matplotlib, NO transformers)
```

### 9.1 დამოკიდებულებები

`pyproject.toml`-დან ამოღებულია `transformers`, `tokenizers`, `safetensors` — ნულიდან აგებულ მოდელს **მხოლოდ `torch`** სჭირდება. BM25-ისთვის `rank-bm25`, გრაფიკებისთვის `matplotlib`.

---

## 10. დასკვნა და სამომავლო გაუმჯობესება

### 10.1 რა გავაკეთე

1. **ნულიდან ავაგე ტოკენიზატორი (BPE).** არანაირი pretrained vocab — merge-ები ნასწავლია ჩვენი კორპუსიდან.
2. **ნულიდან ავაგე Transformer ენკოდერი.** token embedding + sinusoidal positional encoding + multi-head self-attention + FFN + pre-norm residuals — ყველა შრე ხელითაა, მხოლოდ `torch.nn` primitives-ით.
3. **ნულიდან დავწერე contrastive learning.** სიმეტრიული InfoNCE in-batch negatives-ით, AdamW + warmup, CSV logging, best checkpoint.
4. **შევინარჩუნე მკაცრი ევალუაციის ჩარჩო.** BM25 baseline + MRR/Recall + per-query-type breakdown + qualitative comparison.

### 10.2 პროექტის შეზღუდვები

* **მცირე მოდელი + მცირე dataset.** ნულიდან გაწვრთნილი ~5M მოდელი ~8k წყვილზე ვერ მიაღწევს დიდ, pretrained მოდელების ხარისხს — ეს მოსალოდნელია.
* **Domain mismatch.** training ძირითადად Wikipedia-დან მოდის, test კი Jurafsky-დან — ეს ამცირებს test performance-ს.
* **მცირე test set (25 query).** ერთი query-ის რანკის ცვლილება ~0.04 MRR shift-ს იწვევს — დიდი variance.
* **Random negatives.** hard negatives უფრო ძლიერ სიგნალს მისცემდა.

### 10.3 სამომავლო გაუმჯობესება

* **Hard negative mining:** თითო query-ისთვის BM25-ის top-k-დან არასწორი chunks როგორც negatives.
* **მეტი epoch / დიდი მოდელი:** მეტი layer/d_model, თუ GPU-ს რესურსი იძლევა.
* **Learned positional embeddings:** sinusoidal-ის ნაცვლად ნასწავლი pozიციური embeddings (შესადარებლად).
* **Domain-balanced data:** მეტი Jurafsky წყვილი, რომ training-test შესაბამისობა გაიზარდოს.
* **Hybrid retrieval:** BM25 top-50 + neural rerank — წარმოებაში სტანდარტი ("best of both worlds").

### 10.4 დასკვნა

ეს პროექტი აჩვენებს, რომ **pretrained მოდელის გარეშე** შემიძლია ავაგო სრული neural search pipeline ნულიდან: საკუთარი BPE ტოკენიზატორი, საკუთარი Transformer ენკოდერი (embeddings, positional encoding, attention, FFN), საკუთარი contrastive loss, და მკაცრი ევალუაცია BM25 baseline-თან. მესმის თითოეული არჩევანის trade-off — რატომ BPE > word-level, რატომ mean pooling + L2 = სწრაფი retrieval, რატომ სიმეტრიული InfoNCE, და რატომ რჩება BM25 ძლიერი keyword query-ებზე.

---

## ბიბლიოგრაფია

* Jurafsky, D. & Martin, J. *Speech and Language Processing* (3rd ed., draft).
* Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS. — Transformer + sinusoidal positional encoding.
* Sennrich, R., Haddow, B. & Birch, A. (2016). *Neural Machine Translation of Rare Words with Subword Units.* ACL. — BPE.
* van den Oord, A., Li, Y. & Vinyals, O. (2018). *Representation Learning with Contrastive Predictive Coding.* — InfoNCE.
* Gao, T., Yao, X. & Chen, D. (2021). *SimCSE: Simple Contrastive Learning of Sentence Embeddings.* EMNLP. — temperature, in-batch negatives.
* Radford, A. et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision (CLIP).* — symmetric InfoNCE.
* Reimers, N. & Gurevych, I. (2019). *Sentence-BERT.* EMNLP. — bi-encoder + mean pooling (კონცეპტუალური მითითება).
* Robertson, S. & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.*

---

## დანართი — Reproducibility

```bash
# 1. Setup (uv; torch + rank-bm25 + matplotlib, NO pretrained libs)
uv sync

# 2. (ერთჯერ) მონაცემების მომზადება — chunking + LLM queries
jupyter notebook notebooks/01_data_v2_llm.ipynb

# 3. წვრთნა Colab T4-ზე: ჯერ BPE ტოკენიზატორი, შემდეგ ენკოდერი
jupyter notebook notebooks/02_training.ipynb

# 4. ევალუაცია — BM25 vs untrained vs trained
jupyter notebook notebooks/03_evaluation.ipynb

# 5. ინტერაქტიული demo
jupyter notebook notebooks/04_demo.ipynb
```

ლოკალურად (CPU) კოდის სისწორის შესამოწმებლად:
```bash
uv run python -m src.tokenizer     # BPE round-trip
uv run python -m src.model         # encoder shape/norm smoke test
uv run python -m src.loss          # InfoNCE ≈ log(B)
uv run python -m src.evaluate      # BM25 + (untrained) neural metrics
```

---

*ანგარიში დასრულდა.*
