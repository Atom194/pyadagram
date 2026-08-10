# pyadagram

`pyadagram` is a small Python extension module built with PyO3 on top of the
[`adagram`](https://github.com/ondra/adagram) Rust crate. It exposes model
loading, lexicon and vector access, sentence embeddings, nearest-neighbor
queries, and contextual disambiguation for Adaptive Skip-gram models from
Python.

## Build and import

### With cargo

Build the extension module with Cargo:

```bash
cargo build --release
```

The raw Cargo output is named `libadagram.so` on Linux, but the
Python module is imported as `adagram`. For `import adagram` to work, the
shared library must be available under the module name expected by Python. 

To make the import work, copy or link `./target/release/libadagram.so` to
`adagram.so` and place it in the directory with your script or on `PYTHONPATH`.

### With pip

Add the module using pip:

```bash
pip install git+https://github.com/ondra/pyadagram
```

## Python API

The compiled Python module is imported as `adagram` and provides one class:

```python
model = adagram.Model(model_path)
```

Methods:

- `model.id_range()` returns the number of word IDs in the model.
- `model.dim()` returns the true embedding dimension, without internal SIMD
  padding.
- `model.id2str(word_id)` returns the word for an ID. An invalid ID raises
  `IndexError`.
- `model.str2id(word)` returns the word ID or `None` when the word is not in
  the lexicon.
- `model.counts(word_id)` returns the raw learned count for every sense slot.
- `model.embedding(word_id, sense_id, *, normalize=False)` returns one sense
  embedding. Invalid IDs raise `IndexError`; an inactive sense raises
  `ValueError`.
- `model.embeddings_sent(words, *, weighted=True, normalize=False, window=None,
  min_prob=1e-3)` returns one contextual embedding per input token. The result
  contains a float list for each known word and `None` for each OOV word. With
  `weighted=True`, each vector is the posterior-weighted mean of its sense
  vectors; with `weighted=False`, it is the maximum probability sense vector.
- `model.nearest(word, senseno, num_neighbors=10, min_freq=5, min_prob=1e-3)`
  returns nearest neighbors for one sense as
  `[(neighbor_word, neighbor_sense, similarity), ...]`
- `model.nearest_all(word, num_neighbors=10, min_freq=5, min_prob=1e-3)`
  returns neighbors for all active senses as
  `[(sense_no, neighbors), ...]`
- `model.disamb(word, ctx, *, min_prob=1e-3)` returns
  `(sense_distribution, (num_senses, ctx_valid, ctx_oov))`

`embeddings` accepts a pre-tokenized sequence of strings. Its context window is
symmetric and excludes only the current token position. An explicit `window`
overrides the default; `window=0` uses the full sentence. When omitted, the
window is inferred from a `.wN.` component in the model path, with a fallback
of 4 tokens on each side. OOV context words are ignored, and a known word with
no known context uses its learned sense prior.

Unknown headwords passed to `nearest`, `nearest_all`, or `disamb` raise
`ValueError`. Model loading failures are reported as runtime errors from the
underlying Rust loader.

## Embedding example

```python
import adagram

model = adagram.Model("MODEL")

bank_id = model.str2id("bank-n")
if bank_id is not None:
    print(model.id2str(bank_id))
    print(model.counts(bank_id))
    bank_sense_0 = model.embedding(bank_id, 0, normalize=True)

words = ["the-x", "bank-n", "raised-v", "rates-n"]
soft_embeddings = model.embeddings_sent(words)
map_embeddings = model.embeddings_sent(words, weighted=False, normalize=True)

assert len(soft_embeddings) == len(words)
assert soft_embeddings[0] is None or len(soft_embeddings[0]) == model.dim()
```

## Example


```python
import adagram

model = adagram.Model("MODEL")
nns = model.nearest_all("bank-n", num_neighbors=3, min_freq=100)

nns[0]   # sense 0: central bank policy
# (0, [('rate-n', 15, 0.883), ('Boe-n', 3, 0.879)])

nns[2]   # sense 2: power bank
# (2, [('station-n', 4, 0.951), ('high-capacity-n', 1, 0.911)])

nns[4]   # sense 4: river bank
# (4, [('edge-n', 8, 0.922), ('eastern-j', 3, 0.908)])

sdist, (nsenses, ctx_valid, ctx_oov) = model.desamb(
    "bank-n",
    ["charge-v", "phone-n"],
)
[f"{i}:{p:.2f}" for i, p in enumerate(sdist) if p > 0.01]
# ['2:0.45', '8:0.30', '17:0.18']
```

