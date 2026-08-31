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
        charge_present: Tensor::from_vec(vec![1.0f32], 1, &device).unwrap(),
        nce: Tensor::from_vec(vec![27.0f32], 1, &device).unwrap(),
        nce_present: Tensor::from_vec(vec![1.0f32], 1, &device).unwrap(),
        instrument_ids: Tensor::from_vec(vec![0u32], 1, &device).unwrap(),
        instrument_present: Tensor::from_vec(vec![0.0f32], 1, &device).unwrap(),
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

#[test]
fn all_unknown_acquisition_context_is_a_valid_forward_path() {
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
    let context = PrecursorContextBatch::unknown(1, &device).unwrap();
    assert_eq!(context.charge_present.to_vec1::<f32>().unwrap(), vec![0.0]);
    assert_eq!(context.nce_present.to_vec1::<f32>().unwrap(), vec![0.0]);
    assert_eq!(
        context.instrument_present.to_vec1::<f32>().unwrap(),
        vec![0.0]
    );

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = PeptideFoundationMultiTaskModel::new(config.clone(), vb).unwrap();
    let output = model.forward_t(&batch, &context, false).unwrap();
    assert_eq!(output.rt.dims(), &[1, 1]);
    assert_eq!(output.ccs.dims(), &[1, 1]);
    assert_eq!(output.ms2.dims(), &[1, config.max_sequence_len - 1, 8]);
}

#[test]
fn rt_encoder_gradient_gate_preserves_forward_values() {
    let device = Device::Cpu;
    let config = FoundationConfig {
        max_sequence_len: 12,
        transformer_layers: 1,
        dropout: 0.0,
        ..FoundationConfig::default()
    };
    let featurizer = PeptideGraphFeaturizer::new(config.clone()).unwrap();
    let batch = featurizer
        .featurize(&[PeptidoformInput::unmodified("PEPTIDEK")], &device)
        .unwrap();
    let context = PrecursorContextBatch::unknown(1, &device).unwrap();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = PeptideFoundationMultiTaskModel::new(config, vb).unwrap();

    let full = model
        .forward_t_with_rt_encoder_gradient_scale(&batch, &context, false, 1.0)
        .unwrap();
    let gated = model
        .forward_t_with_rt_encoder_gradient_scale(&batch, &context, false, 0.5)
        .unwrap();

    let full_rt = full.rt.to_vec2::<f32>().unwrap();
    let gated_rt = gated.rt.to_vec2::<f32>().unwrap();
    assert_eq!(full_rt.len(), gated_rt.len());
    for (full_row, gated_row) in full_rt.iter().zip(gated_rt.iter()) {
        for (left, right) in full_row.iter().zip(gated_row.iter()) {
            assert!((left - right).abs() <= 1e-6);
        }
    }
}

#[test]
fn rt_encoder_gradient_gate_scales_encoder_but_not_rt_head_gradients() {
    fn squared_norm(tensor: &Tensor) -> f64 {
        f64::from(
            tensor
                .sqr()
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap(),
        )
    }

    let device = Device::Cpu;
    let config = FoundationConfig {
        max_sequence_len: 12,
        transformer_layers: 1,
        dropout: 0.0,
        ..FoundationConfig::default()
    };
    let featurizer = PeptideGraphFeaturizer::new(config.clone()).unwrap();
    let batch = featurizer
        .featurize(&[PeptidoformInput::unmodified("PEPTIDEK")], &device)
        .unwrap();
    let context = PrecursorContextBatch::unknown(1, &device).unwrap();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = PeptideFoundationMultiTaskModel::new(config, vb).unwrap();

    let full = model
        .forward_t_with_rt_encoder_gradient_scale(&batch, &context, false, 1.0)
        .unwrap();
    let half = model
        .forward_t_with_rt_encoder_gradient_scale(&batch, &context, false, 0.5)
        .unwrap();
    let gradients_full = full.rt.sum_all().unwrap().backward().unwrap();
    let gradients_half = half.rt.sum_all().unwrap().backward().unwrap();

    let data = varmap.data().lock().unwrap();
    let encoder = data
        .get("encoder.output_norm.weight")
        .cloned()
        .expect("foundation encoder output-norm weight");
    let rt_head = data
        .get("heads.rt.weight")
        .cloned()
        .expect("foundation RT-head weight");
    drop(data);

    let encoder_full = gradients_full
        .get(&encoder)
        .map(squared_norm)
        .unwrap_or(0.0)
        .sqrt();
    let encoder_half = gradients_half
        .get(&encoder)
        .map(squared_norm)
        .unwrap_or(0.0)
        .sqrt();
    let head_full = gradients_full
        .get(&rt_head)
        .map(squared_norm)
        .unwrap_or(0.0)
        .sqrt();
    let head_half = gradients_half
        .get(&rt_head)
        .map(squared_norm)
        .unwrap_or(0.0)
        .sqrt();

    assert!(encoder_full > 0.0);
    assert!(head_full > 0.0);
    assert!((encoder_half / encoder_full - 0.5).abs() < 1e-3);
    assert!((head_half / head_full - 1.0).abs() < 1e-3);
}
