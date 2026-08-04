//#[path = "../global_alloc.rs"]
mod global_alloc;

use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

#[pyclass]
struct Model {
    vm: adagram::adagram::VectorModel,
    id2str: Vec<String>,
    str2id: std::collections::HashMap<String, u32>,
    default_window: usize,
}

impl Model {
    fn checked_id(&self, id: i64) -> PyResult<usize> {
        usize::try_from(id)
            .ok()
            .filter(|&id| id < self.id2str.len())
            .ok_or_else(|| PyIndexError::new_err(format!("word id out of range: {id}")))
    }

    fn checked_sense(&self, sense: i64) -> PyResult<usize> {
        usize::try_from(sense)
            .ok()
            .filter(|&sense| sense < self.vm.nmeanings())
            .ok_or_else(|| PyIndexError::new_err(format!("sense id out of range: {sense}")))
    }

    fn vector(&self, id: usize, sense: usize) -> Vec<f32> {
        (0..self.vm.dim)
            .map(|dimension| self.vm.in_vecs[[id, sense, dimension]])
            .collect()
    }

    fn normalize(vector: &mut [f32]) {
        let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
        for value in vector {
            *value /= norm;
        }
    }

    fn sense_posterior<I>(&self, head_id: u32, context_ids: I, min_prob: f64) -> (Vec<f64>, u32)
    where
        I: IntoIterator<Item = u32>,
    {
        let mut posterior = self.vm.newz();
        let n_senses = self
            .vm
            .expected_pi(head_id, &mut posterior, min_prob, false);

        for context_id in context_ids {
            self.vm.var_update_z(head_id, context_id, &mut posterior);
        }
        adagram::common::exp_normalize(&mut posterior);

        (posterior.to_vec(), n_senses)
    }
}

#[pymethods]
impl Model {
    #[new]
    fn py_new(modelpath: &str) -> PyResult<Self> {
        let (vm, id2str) = adagram::adagram::VectorModel::load_model(modelpath)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let mut str2id = std::collections::HashMap::<String, u32>::with_capacity(id2str.len());
        for (id, word) in id2str.iter().enumerate() {
            str2id.insert(word.to_string(), id as u32);
        }

        let default_window = adagram::adagram::parse_window(None, modelpath).unwrap_or(4);

        Ok(Model {
            vm,
            id2str,
            str2id,
            default_window,
        })
    }

    fn id_range(&self) -> usize {
        self.id2str.len()
    }

    fn dim(&self) -> usize {
        self.vm.dim
    }

    #[pyo3(name = "id2str")]
    fn py_id2str(&self, id: i64) -> PyResult<String> {
        let id = self.checked_id(id)?;
        Ok(self.id2str[id].clone())
    }

    #[pyo3(name = "str2id")]
    fn py_str2id(&self, word: String) -> Option<u32> {
        self.str2id.get(&word).copied()
    }

    #[pyo3(name = "counts")]
    fn py_counts(&self, id: i64) -> PyResult<Vec<f32>> {
        let id = self.checked_id(id)?;
        Ok((0..self.vm.nmeanings())
            .map(|sense| self.vm.counts[[id, sense]])
            .collect())
    }

    #[pyo3(name = "embedding", signature = (id, sense, *, normalize=false))]
    fn py_embedding(&self, id: i64, sense: i64, normalize: bool) -> PyResult<Vec<f32>> {
        let id = self.checked_id(id)?;
        let sense = self.checked_sense(sense)?;
        let mut vector = self.vector(id, sense);

        if vector.iter().all(|&value| value == 0.0) {
            return Err(PyValueError::new_err(format!(
                "inactive sense: word id {id}, sense id {sense}"
            )));
        }
        if normalize {
            Self::normalize(&mut vector);
        }

        Ok(vector)
    }

    #[pyo3(
        name = "embeddings_sent",
        signature = (words, *, weighted=true, normalize=false, window=None, min_prob=1e-3)
    )]
    fn py_embeddings_sent(
        &self,
        words: Vec<String>,
        weighted: bool,
        normalize: bool,
        window: Option<usize>,
        min_prob: f64,
    ) -> Vec<Option<Vec<f32>>> {
        let ids: Vec<Option<u32>> = words
            .iter()
            .map(|word| self.str2id.get(word).copied())
            .collect();
        let window = window.unwrap_or(self.default_window);

        ids.iter()
            .enumerate()
            .map(|(target_position, head_id)| {
                let head_id = (*head_id)?;
                let (start, end) = if window == 0 {
                    (0, ids.len())
                } else {
                    (
                        target_position.saturating_sub(window),
                        target_position
                            .saturating_add(window)
                            .saturating_add(1)
                            .min(ids.len()),
                    )
                };
                let context_ids = (start..end)
                    .filter(|&position| position != target_position)
                    .filter_map(|position| ids[position]);
                let (posterior, _n_senses) = self.sense_posterior(head_id, context_ids, min_prob);

                let mut vector = if weighted {
                    let mut vector = vec![0.0f32; self.vm.dim];
                    for (sense, probability) in posterior.iter().enumerate() {
                        let probability = *probability as f32;
                        for (dimension, value) in vector.iter_mut().enumerate() {
                            *value +=
                                probability * self.vm.in_vecs[[head_id as usize, sense, dimension]];
                        }
                    }
                    vector
                } else {
                    let sense = posterior
                        .iter()
                        .enumerate()
                        .max_by(|(_, left), (_, right)| left.total_cmp(right))
                        .map(|(sense, _)| sense)
                        .unwrap();
                    self.vector(head_id as usize, sense)
                };

                if normalize {
                    Self::normalize(&mut vector);
                }
                Some(vector)
            })
            .collect()
    }

    #[pyo3(
        name = "nearest",
        signature = (word, senseno, num_neighbors=10, min_freq=5, min_prob=1e-3)
    )]
    fn py_nearest(
        &self,
        word: String,
        senseno: usize,
        num_neighbors: usize,
        min_freq: usize,
        min_prob: f64,
    ) -> PyResult<Vec<(String, u32, f32)>> {
        let head_id = match self.str2id.get(&word) {
            Some(id) => *id,
            None => {
                return Err(PyValueError::new_err(format!(
                    "not in model lexicon: {word}"
                )));
            }
        };

        let hv = adagram::nn::nearest_mmul(
            &self.vm,
            head_id as usize,
            num_neighbors,
            min_freq,
            min_prob,
        )
        .into_iter()
        .find_map(|(sn, hv)| if sn == senseno { Some(hv) } else { None })
        .unwrap_or_default();
        Ok(hv
            .into_iter()
            .map(|(i, j, sim)| (self.id2str[i as usize].clone(), j, sim))
            .collect())
    }

    #[pyo3(name = "disamb", signature = (word, ctx, *, min_prob=1e-3))]
    fn disamb(
        &self,
        word: String,
        ctx: Vec<String>,
        min_prob: f64,
    ) -> PyResult<(Vec<f64>, (u32, usize, usize))> {
        let head_id = match self.str2id.get(&word) {
            Some(id) => *id,
            None => {
                return Err(PyValueError::new_err(format!(
                    "not in model lexicon: {word}"
                )));
            }
        };

        let mut nvalid = 0;
        let mut ninvalid = 0;
        let context_ids = ctx.into_iter().filter_map(|context_word| {
            self.str2id.get(&context_word).copied().map_or_else(
                || {
                    ninvalid += 1;
                    None
                },
                |context_id| {
                    nvalid += 1;
                    Some(context_id)
                },
            )
        });
        let (posterior, n_senses) = self.sense_posterior(head_id, context_ids, min_prob);

        Ok((posterior, (n_senses, nvalid, ninvalid)))
    }

    #[pyo3(
        name = "nearest_all",
        signature = (word, num_neighbors=10, min_freq=5, min_prob=1e-3)
    )]
    fn py_nearest_all(
        &self,
        word: String,
        num_neighbors: usize,
        min_freq: usize,
        min_prob: f64,
    ) -> PyResult<Vec<(usize, Vec<(String, u32, f32)>)>> {
        let head_id = match self.str2id.get(&word) {
            Some(id) => *id,
            None => {
                return Err(PyValueError::new_err(format!(
                    "not in model lexicon: {word}"
                )));
            }
        };

        let hvs = adagram::nn::nearest_mmul(
            &self.vm,
            head_id as usize,
            num_neighbors,
            min_freq,
            min_prob,
        );
        Ok(hvs
            .into_iter()
            .map(|(sn, hv)| {
                (
                    sn,
                    hv.into_iter()
                        .map(|(i, j, sim)| (self.id2str[i as usize].clone(), j, sim))
                        .collect(),
                )
            })
            .collect())
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
