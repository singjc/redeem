pub mod input;
pub mod plot;
pub mod trainer;

use rand::seq::SliceRandom;
use rand::thread_rng;
use redeem_properties::utils::data_handling::PeptideData;

pub fn sample_peptides(peptides: &[PeptideData], n: usize) -> Vec<PeptideData> {
    let mut rng = thread_rng();
    let sample_size = n.min(peptides.len());
    peptides
        .choose_multiple(&mut rng, sample_size)
        .cloned()
        .collect()
}

/// Sample `n` indices from a dataset of length `len`.
/// Returns a Vec<usize> with unique indices in random order (length = min(n, len)).
pub fn sample_indices(len: usize, n: usize) -> Vec<usize> {
    let mut rng = thread_rng();
    let sample_size = n.min(len);
    // build a Vec<usize> [0..len) and choose multiple indices
    let mut idxs: Vec<usize> = (0..len).collect();
    idxs.shuffle(&mut rng);
    idxs.truncate(sample_size);
    idxs
}
