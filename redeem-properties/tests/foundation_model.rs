use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use redeem_properties::foundation::{
    FoundationConfig, PeptideFoundationEncoder, PeptideFoundationMultiTaskModel,
    PeptideGraphFeaturizer, PeptidoformInput, PrecursorContextBatch,
};

#[test]
fn foundation_encoder_returns_global_and_residue_embeddings() {
    let device = Device::Cpu;
    let mut config = FoundationConfig::default();
    config.max_sequence_len = 16;
    config.transformer_layers = 2;
    let featurizer = PeptideGraphFeaturizer::new(config.clone()).unwrap();
    let batch = featurizer
        .featurize(&[PeptidoformInput::unmodified("PEPTIDEK")], &device)
        .unwrap();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let encoder = PeptideFoundationEncoder::new(config.clone(), vb).unwrap();
    let output = encoder.forward_t(&batch, false).unwrap();

    assert_eq!(output.peptide_embedding.dims(), &[1, config.model_dim]);
    assert_eq!(
        output.residue_embeddings.dims(),
        &[1, config.max_sequence_len, config.model_dim]
    );
}

#[test]
fn multi_task_heads_have_expected_shapes() {
    let device = Device::Cpu;
    let mut config = FoundationConfig::default();
    config.max_sequence_len = 12;
    config.transformer_layers = 1;
    let featurizer = PeptideGraphFeaturizer::new(config.clone()).unwrap();
    let batch = featurizer
        .featurize(&[PeptidoformInput::unmodified("PEPTIDEK")], &device)
        .unwrap();
    let context = PrecursorContextBatch {
        charge: Tensor::from_vec(vec![2.0f32], 1, &device).unwrap(),
        nce: Tensor::from_vec(vec![27.0f32], 1, &device).unwrap(),
        instrument_ids: Tensor::from_vec(vec![0u32], 1, &device).unwrap(),
    };
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = PeptideFoundationMultiTaskModel::new(config.clone(), vb).unwrap();
    let output = model.forward_t(&batch, &context, false).unwrap();

    assert_eq!(output.rt.dims(), &[1, 1]);
    assert_eq!(output.ccs.dims(), &[1, 1]);
    assert_eq!(
        output.ms2.dims(),
        &[1, config.max_sequence_len - 1, config.ms2_fragment_channels]
    );
}
