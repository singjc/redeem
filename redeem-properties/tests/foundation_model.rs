use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, VarBuilder, VarMap};
use redeem_properties::foundation::model::gradient_scaled_identity;
use redeem_properties::foundation::{
    FoundationCcsPhysicsBaselineConfig, FoundationConfig, PeptideFoundationEncoder,
    PeptideFoundationMultiTaskModel, PeptideGraphFeaturizer, PeptidoformInput,
    PrecursorContextBatch,
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
        precursor_mz: Tensor::from_vec(vec![650.0f32], 1, &device).unwrap(),
        precursor_mz_present: Tensor::from_vec(vec![1.0f32], 1, &device).unwrap(),
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
fn normalized_regression_heads_start_from_zero_prediction() {
    let device = Device::Cpu;
    let config = FoundationConfig {
        max_sequence_len: 12,
        transformer_layers: 1,
        dropout: 0.0,
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
    let context = PrecursorContextBatch {
        charge: Tensor::from_vec(vec![2.0f32, 4.0], 2, &device).unwrap(),
        charge_present: Tensor::from_vec(vec![1.0f32, 1.0], 2, &device).unwrap(),
        precursor_mz: Tensor::from_vec(vec![650.0f32, 800.0], 2, &device).unwrap(),
        precursor_mz_present: Tensor::from_vec(vec![1.0f32, 1.0], 2, &device).unwrap(),
        nce: Tensor::from_vec(vec![25.0f32, 35.0], 2, &device).unwrap(),
        nce_present: Tensor::from_vec(vec![1.0f32, 1.0], 2, &device).unwrap(),
        instrument_ids: Tensor::from_vec(vec![0u32, 0], 2, &device).unwrap(),
        instrument_present: Tensor::from_vec(vec![0.0f32, 0.0], 2, &device).unwrap(),
    };

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = PeptideFoundationMultiTaskModel::new(config, vb).unwrap();
    let output = model.forward_t(&batch, &context, false).unwrap();

    for row in output.rt.to_vec2::<f32>().unwrap() {
        assert_eq!(row, vec![0.0]);
    }
    for row in output.ccs.to_vec2::<f32>().unwrap() {
        assert_eq!(row, vec![0.0]);
    }
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
    assert_eq!(
        context.precursor_mz_present.to_vec1::<f32>().unwrap(),
        vec![0.0]
    );
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

fn assert_gradient_scale_primitive(scale: f64) {
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

    // Use an explicit tracked variable as the "encoder output" so Candle's
    // GradStore can be queried directly without relying on model parameter names.
    let input_varmap = VarMap::new();
    let input = {
        let vb = VarBuilder::from_varmap(&input_varmap, DType::F32, &device);
        vb.get_with_hints((1, 4), "embedding", candle_nn::Init::Const(1.0))
            .unwrap()
    };

    // A real trainable head verifies that the gate changes only the upstream
    // gradient while leaving head-parameter gradients untouched.
    let head_varmap = VarMap::new();
    let head = {
        let vb = VarBuilder::from_varmap(&head_varmap, DType::F32, &device);
        candle_nn::linear(4, 1, vb).unwrap()
    };

    let full_features = gradient_scaled_identity(&input, 1.0).unwrap();
    let gated_features = gradient_scaled_identity(&input, scale).unwrap();
    let full_output = head.forward(&full_features).unwrap();
    let gated_output = head.forward(&gated_features).unwrap();

    // Gradient scaling is an identity in the forward pass.
    let full_values = full_output.to_vec2::<f32>().unwrap();
    let gated_values = gated_output.to_vec2::<f32>().unwrap();
    assert_eq!(full_values.len(), gated_values.len());
    for (full_row, gated_row) in full_values.iter().zip(gated_values.iter()) {
        for (left, right) in full_row.iter().zip(gated_row.iter()) {
            assert!((left - right).abs() <= 1e-6);
        }
    }

    let gradients_full = full_output.sum_all().unwrap().backward().unwrap();
    let gradients_gated = gated_output.sum_all().unwrap().backward().unwrap();

    let input_full = gradients_full
        .get(&input)
        .expect("full path must produce an upstream gradient");
    let input_gated = gradients_gated
        .get(&input)
        .expect("gated path must produce an upstream gradient");
    let weight_full = gradients_full
        .get(head.weight())
        .expect("full path must produce a head-weight gradient");
    let weight_gated = gradients_gated
        .get(head.weight())
        .expect("gated path must produce a head-weight gradient");

    let input_full_norm = squared_norm(input_full).sqrt();
    let input_gated_norm = squared_norm(input_gated).sqrt();
    let weight_full_norm = squared_norm(weight_full).sqrt();
    let weight_gated_norm = squared_norm(weight_gated).sqrt();

    assert!(input_full_norm > 0.0);
    assert!(weight_full_norm > 0.0);
    assert!((input_gated_norm / input_full_norm - scale).abs() < 1e-6);
    assert!((weight_gated_norm / weight_full_norm - 1.0).abs() < 1e-6);

    if let Some(bias) = head.bias() {
        let bias_full = gradients_full
            .get(bias)
            .expect("full path must produce a head-bias gradient");
        let bias_gated = gradients_gated
            .get(bias)
            .expect("gated path must produce a head-bias gradient");
        let bias_full_norm = squared_norm(bias_full).sqrt();
        let bias_gated_norm = squared_norm(bias_gated).sqrt();
        assert!(bias_full_norm > 0.0);
        assert!((bias_gated_norm / bias_full_norm - 1.0).abs() < 1e-6);
    }
}

#[test]
fn rt_encoder_gradient_gate_scales_encoder_but_not_rt_head_gradients() {
    assert_gradient_scale_primitive(0.5);
}

#[test]
fn ccs_encoder_gradient_gate_preserves_forward_values() {
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
        .forward_t_with_shared_gradient_scales(&batch, &context, false, 1.0, 1.0)
        .unwrap();
    let gated = model
        .forward_t_with_shared_gradient_scales(&batch, &context, false, 1.0, 0.5)
        .unwrap();

    let full_ccs = full.ccs.to_vec2::<f32>().unwrap();
    let gated_ccs = gated.ccs.to_vec2::<f32>().unwrap();
    assert_eq!(full_ccs.len(), gated_ccs.len());
    for (full_row, gated_row) in full_ccs.iter().zip(gated_ccs.iter()) {
        for (left, right) in full_row.iter().zip(gated_row.iter()) {
            assert!((left - right).abs() <= 1e-6);
        }
    }
}

#[test]
fn ccs_encoder_gradient_gate_scales_encoder_but_not_ccs_head_gradients() {
    assert_gradient_scale_primitive(0.5);
}

#[test]
fn ccs_physics_baseline_initializes_a_native_physics_prior_plus_zero_residual() {
    let device = Device::Cpu;
    let config = FoundationConfig {
        max_sequence_len: 12,
        transformer_layers: 1,
        dropout: 0.0,
        ccs_physics_baseline: Some(FoundationCcsPhysicsBaselineConfig {
            // Make the expected native baseline easy to compute:
            // 100 + 40 * (charge/4) + 30 * (mz/1000) + 60 * (len/30).
            coefficients_native: [100.0, 40.0, 0.0, 30.0, 0.0, 60.0, 0.0, 0.0],
            target_mean_native: 200.0,
            target_std_native: 50.0,
        }),
        ..FoundationConfig::default()
    };
    let featurizer = PeptideGraphFeaturizer::new(config.clone()).unwrap();
    let batch = featurizer
        .featurize(
            &[
                PeptidoformInput::unmodified("PEPTIDEK"),    // len = 8
                PeptidoformInput::unmodified("AGHCEWQMKYR"), // len = 11
            ],
            &device,
        )
        .unwrap();
    let context = PrecursorContextBatch {
        charge: Tensor::from_vec(vec![2.0f32, 4.0], 2, &device).unwrap(),
        charge_present: Tensor::from_vec(vec![1.0f32, 1.0], 2, &device).unwrap(),
        precursor_mz: Tensor::from_vec(vec![500.0f32, 1000.0], 2, &device).unwrap(),
        precursor_mz_present: Tensor::from_vec(vec![1.0f32, 1.0], 2, &device).unwrap(),
        nce: Tensor::zeros(2, DType::F32, &device).unwrap(),
        nce_present: Tensor::zeros(2, DType::F32, &device).unwrap(),
        instrument_ids: Tensor::zeros(2, DType::U32, &device).unwrap(),
        instrument_present: Tensor::zeros(2, DType::F32, &device).unwrap(),
    };

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = PeptideFoundationMultiTaskModel::new(config, vb).unwrap();
    let output = model.forward_t(&batch, &context, false).unwrap();
    let values = output.ccs.squeeze(1).unwrap().to_vec1::<f32>().unwrap();

    let native0 = 100.0 + 40.0 * (2.0 / 4.0) + 30.0 * (500.0 / 1000.0) + 60.0 * (8.0 / 30.0);
    let native1 = 100.0 + 40.0 * (4.0 / 4.0) + 30.0 * (1000.0 / 1000.0) + 60.0 * (11.0 / 30.0);
    let expected0 = (native0 - 200.0) / 50.0;
    let expected1 = (native1 - 200.0) / 50.0;

    assert!((values[0] - expected0).abs() < 1e-5);
    assert!((values[1] - expected1).abs() < 1e-5);
}

#[test]
fn neutral_mass_charge_ccs_context_uses_precursor_mz_without_changing_head_width() {
    use redeem_properties::foundation::FoundationCcsContextMode;

    let device = Device::Cpu;
    let config = FoundationConfig {
        max_sequence_len: 12,
        transformer_layers: 1,
        dropout: 0.0,
        ccs_context_mode: FoundationCcsContextMode::NeutralMassCharge,
        ..FoundationConfig::default()
    };
    let featurizer = PeptideGraphFeaturizer::new(config.clone()).unwrap();
    let batch = featurizer
        .featurize(
            &[
                PeptidoformInput::unmodified("PEPTIDEK"),
                PeptidoformInput::unmodified("PEPTIDEK"),
            ],
            &device,
        )
        .unwrap();
    let context = PrecursorContextBatch {
        charge: Tensor::from_vec(vec![2.0f32, 2.0], 2, &device).unwrap(),
        charge_present: Tensor::from_vec(vec![1.0f32, 1.0], 2, &device).unwrap(),
        precursor_mz: Tensor::from_vec(vec![500.0f32, 1000.0], 2, &device).unwrap(),
        precursor_mz_present: Tensor::from_vec(vec![1.0f32, 1.0], 2, &device).unwrap(),
        nce: Tensor::zeros(2, DType::F32, &device).unwrap(),
        nce_present: Tensor::zeros(2, DType::F32, &device).unwrap(),
        instrument_ids: Tensor::zeros(2, DType::U32, &device).unwrap(),
        instrument_present: Tensor::zeros(2, DType::F32, &device).unwrap(),
    };

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = PeptideFoundationMultiTaskModel::new(config.clone(), vb).unwrap();

    // Isolate the first scalar CCS-context channel. The head width stays
    // `model_dim + 2`, so historical model-only initialization snapshots remain
    // shape-compatible across the two context modes.
    let ccs_weight = {
        let data = varmap.data().lock().unwrap();
        data.get("heads.ccs.weight").unwrap().clone()
    };
    let mut weights = vec![0.0f32; config.model_dim + 2];
    weights[config.model_dim] = 1.0;
    ccs_weight
        .set(&Tensor::from_vec(weights, (1, config.model_dim + 2), &device).unwrap())
        .unwrap();

    let output = model.forward_t(&batch, &context, false).unwrap();
    let values = output.ccs.squeeze(1).unwrap().to_vec1::<f32>().unwrap();
    assert!((values[0] - 1.0 / 3.0).abs() < 1e-5);
    assert!((values[1] - 2.0 / 3.0).abs() < 1e-5);
}
