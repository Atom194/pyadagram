# pyadagram

`pyadagram` is a small Python extension module built with PyO3 on top of the
[`adagram`](https://github.com/ondra/adagram) Rust crate. It exposes model
loading, nearest-neighbor queries, and contextual disambiguation for Adaptive
Skip-gram models from Python.

## Build and import

Build the extension module with Cargo:

```bash
cargo build --release
```

The raw Cargo output is named `libpyadagram.so` on Linux, but the
Python module is imported as `adagram`. For `import adagram` to work, the
shared library must be available under the module name expected by Python. 

To make the import work, copy or link the built library in `./target/release/libadagram.so` to `adagram.so` and place it in the directory with your script or on `PYTHONPATH`.

## Python API

The compiled Python module is imported as `adagram` and provides one class:

```python
model = adagram.Model(model_path)
```

Methods:

- `model.nearest(word, senseno, num_neighbors=10, min_freq=5, min_prob=1e-3)`
  returns nearest neighbors for one sense as
  `[(neighbor_word, neighbor_sense, similarity), ...]`
- `model.nearest_all(word, num_neighbors=10, min_freq=5, min_prob=1e-3)`
  returns neighbors for all active senses as
  `[(sense_no, neighbors), ...]`
- `model.desamb(word, ctx)` returns
  `(sense_distribution, (num_senses, ctx_valid, ctx_oov))`

Unknown headwords raise `ValueError`. Model loading failures are reported as
runtime errors from the underlying Rust loader.

## Appendix example

The thesis appendix uses `pyadagram` like this:

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

`MODEL` should be replaced with a trained Adaptive Skip-gram model path created with the `learn` binary from the upstream `adagram` repository.
