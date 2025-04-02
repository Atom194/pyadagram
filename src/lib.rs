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

    #[pyo3(signature = (word, senseno, num_neighbors=10, min_freq=5))]
    fn py_nn(&self, word: String, senseno: usize, num_neighbors: usize, min_freq: usize) -> PyResult<Vec<(String, u32, f32)>> {
        let head_id = match self.str2id.get(&word) {
            Some(id) => *id,
            None => { return Err(PyValueError::new_err(format!("not in model lexicon: {}", word))); },
        };

        /*
        if senseno >= vm.in_vecs.len_of(Axis(1)) {
            return Err(PyValueError::new_err(
                format!("invalid sense number {}, model has {} senses",
                    senseno, vm.in_vecs.len_of(Axis(1)) )));
        }*/

        let hv = adagram::nn::nearest(&self.vm, head_id as usize, senseno, num_neighbors, min_freq);
        for (i, j, sim) in hv.iter() {
            println!("{} {} {}", sim, self.id2str[*i as usize], j);
        }

        Ok(hv.iter().map(
            |(i, j, sim)|
                (self.id2str[*i as usize].clone(), *j, *sim)
            ).collect())
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
