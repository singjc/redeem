use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use redeem_properties::foundation::{
    FoundationConfig, PeptideFoundationEncoder, PeptideFoundationMultiTaskModel,
    PeptideGraphFeaturizer, PeptidoformInput, PrecursorContextBatch,
};

#[test]
fn foundation_encoder_returns_global_and_residue_embeddings() {
    let device = Device::Cpu;
    let config = FoundationConfig {
        max_sequence_len: 16,
        transformer_layers: 2,
        ..FoundationConfig::default()
    };
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
    let config = FoundationConfig {
        max_sequence_len: 12,
        transformer_layers: 1,
        ..FoundationConfig::default()
    };
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

#[test]
fn batched_default_length_attention_handles_contiguous_qkv() {
    let device = Device::Cpu;
    // One layer is sufficient to exercise the exact Q/K/V layout used by the
    // default 64-residue, four-head model while keeping the regression test
    // inexpensive. With batch size two this produces Q/K/V tensors shaped
    // `[2, 4, 64, 48]`, matching the layout that exposed the CPU matmul bug.
    let config = FoundationConfig {
        transformer_layers: 1,
        ..FoundationConfig::default()
    };

    let featurizer = PeptideGraphFeaturizer::new(config.clone()).unwrap();
    let batch = featurizer
        .featurize(
            &[
                PeptidoformInput::unmodified("PEPTIDEK"),
                PeptidoformInput::unmodified("AGHCEWQMKYR"),
            ],
            &device,
        )
        .unwrap();

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let encoder = PeptideFoundationEncoder::new(config.clone(), vb).unwrap();
    let output = encoder.forward_t(&batch, false).unwrap();

    assert_eq!(output.peptide_embedding.dims(), &[2, config.model_dim]);
    assert_eq!(
        output.residue_embeddings.dims(),
        &[2, config.max_sequence_len, config.model_dim]
    );
}
