use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError,PyRuntimeError};

#[pyclass]
struct Model {
    vm: adagram::adagram::VectorModel,     
    id2str: Vec<String>,
    str2id: std::collections::HashMap<String, u32>,
}

#[pymethods]
impl Model {
    #[new]
    fn py_new(modelpath: &str) -> PyResult<Self> {
        let (vm, id2str) = adagram::adagram::VectorModel::load_model(&modelpath)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        // build reverse lexicon mapping
        let mut str2id = std::collections::HashMap::<String, u32>::with_capacity(id2str.len());
        for (id, word) in id2str.iter().enumerate() {
            str2id.insert(word.to_string(), id as u32);
        }

        Ok(Model { vm, id2str, str2id })
    }

    #[pyo3(name="nearest", signature = (word, senseno, num_neighbors=10, min_freq=5))]
    fn py_nearest(&self, word: String, senseno: usize, num_neighbors: usize, min_freq: usize) -> PyResult<Vec<(String, u32, f32)>> {
        let head_id = match self.str2id.get(&word) {
            Some(id) => *id,
            None => { return Err(PyValueError::new_err(format!("not in model lexicon: {}", word))); },
        };

        let hv = adagram::nn::nearest(&self.vm, head_id as usize, senseno, num_neighbors, min_freq);
        Ok(hv.into_iter().map(
            |(i, j, sim)|
                (self.id2str[i as usize].clone(), j, sim)
            ).collect())
    }

    #[pyo3(name="desamb", signature = (word, ctx))]
    fn desamb(&self, word: String, ctx: Vec<String>) -> PyResult<(Vec<f64>, (u32, usize, usize))> {
        let head_id = match self.str2id.get(&word) {
            Some(id) => *id,
            None => { return Err(PyValueError::new_err(format!("not in model lexicon: {}", word))); },
        };

        let mut nvalid = 0;
        let mut ninvalid = 0;

        let mut z = self.vm.newz();

        let n_senses = self.vm.expected_pi(head_id, &mut z, 0.001, false);
        for ctxword in ctx {
            let ctx_id = match self.str2id.get(&ctxword) {
                Some(n) => { nvalid += 1; *n },
                None => { ninvalid += 1; continue; },
            };
            self.vm.var_update_z(head_id, ctx_id, &mut z);
        }
        adagram::common::exp_normalize(&mut z);

        Ok((z.to_vec(), (n_senses, nvalid, ninvalid)))
    }
}


/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule(name = "adagram")]
fn pyadagram(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Model>()?;
    Ok(())
}
