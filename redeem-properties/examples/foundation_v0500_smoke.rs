use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use redeem_properties::foundation::{
    FoundationModification, PeptideFoundationV0500Config, PeptideFoundationV0500Model,
    PeptideGraphFeaturizer, PeptidoformInput, PrecursorContextBatch,
    FOUNDATION_MULTIMODAL_ARCHITECTURE_V0500,
};

fn main() -> Result<()> {
    let mode = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "local".to_string());
    let (config, device) = match mode.as_str() {
        "local" => (PeptideFoundationV0500Config::local_smoke(), Device::Cpu),
        "a100" => {
            #[cfg(feature = "cuda")]
            {
                (
                    PeptideFoundationV0500Config::a100_smoke(),
                    Device::new_cuda(0)?,
                )
            }
            #[cfg(not(feature = "cuda"))]
            {
                bail!("a100 smoke requires redeem-properties feature 'cuda'")
            }
        }
        other => bail!("usage: foundation_v0500_smoke [local|a100], got {other:?}"),
    };
    config.validate()?;

    let featurizer = PeptideGraphFeaturizer::new(config.featurizer_config())?;
    let mut modified = PeptidoformInput::unmodified("PEPTIDER");
    modified
        .modifications
        .push(FoundationModification::mass_delta(3, 79.9663));
    let batch = featurizer.featurize(
        &[
            modified,
            PeptidoformInput::unmodified("ACDEFGHIK"),
            PeptidoformInput::unmodified("MKWVTF"),
        ],
        &device,
    )?;

    let context = PrecursorContextBatch {
        charge: Tensor::new(&[2.0f32, 3.0, 2.0], &device)?,
        charge_present: Tensor::ones(3, DType::F32, &device)?,
        precursor_mz: Tensor::new(&[500.25f32, 610.4, 430.2], &device)?,
        precursor_mz_present: Tensor::ones(3, DType::F32, &device)?,
        nce: Tensor::new(&[30.0f32, 28.0, 32.0], &device)?,
        nce_present: Tensor::ones(3, DType::F32, &device)?,
        instrument_ids: Tensor::from_vec(vec![1u32, 2, 1], 3, &device)?.to_dtype(DType::U32)?,
        instrument_present: Tensor::ones(3, DType::F32, &device)?,
    };

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = PeptideFoundationV0500Model::new(config.clone(), vb)?;
    let output = model.forward_t(&batch, &context, true)?;

    let loss = (((output.rt.sqr()?.mean_all()? + output.mobility_native.sqr()?.mean_all()?)?
        + output.ms2.sqr()?.mean_all()?)?
        + output.pair_interaction_logits.sqr()?.mean_all()?)?;
    let gradients = loss.backward()?;
    let data = varmap.data().lock().unwrap();
    let parameter_count: usize = data
        .values()
        .map(|value| value.as_tensor().elem_count())
        .sum();
    let required_gradients = [
        "student_v050.chemistry.atom_input.weight",
        "student_v050.interaction.0.attention.query.weight",
        "student_v050.interaction.0.pair_update_left.weight",
        "student_v050.task.embedding.weight",
        "student_v050.heads.rt.output.weight",
        "student_v050.heads.mobility.output.weight",
        "student_v050.heads.ms2.output.weight",
    ];
    let mut gradient_checks = Vec::new();
    for name in required_gradients {
        let variable = data
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("missing smoke parameter {name}"))?;
        let gradient = gradients
            .get(variable)
            .ok_or_else(|| anyhow::anyhow!("missing smoke gradient {name}"))?;
        let norm2 = gradient.sqr()?.sum_all()?.to_scalar::<f32>()?;
        if !norm2.is_finite() {
            bail!("non-finite gradient for {name}");
        }
        gradient_checks.push((name, norm2.sqrt()));
    }

    println!("architecture={FOUNDATION_MULTIMODAL_ARCHITECTURE_V0500}");
    println!("mode={mode}");
    println!("device={device:?}");
    println!("graph_hidden_dim={}", config.graph_hidden_dim);
    println!("graph_layers={}", config.graph_layers);
    println!("residue_dim={}", config.residue_dim);
    println!("pair_dim={}", config.pair_dim);
    println!("interaction_blocks={}", config.interaction_blocks);
    println!("attention_heads={}", config.num_attention_heads);
    println!("feed_forward_dim={}", config.feed_forward_dim);
    println!("parameter_count={parameter_count}");
    println!("rt_shape={:?}", output.rt.shape());
    println!("mobility_shape={:?}", output.mobility_native.shape());
    println!("ms2_shape={:?}", output.ms2.shape());
    println!(
        "pair_shape={:?}",
        output.representation.pair_embeddings.shape()
    );
    println!("loss={:.8}", loss.to_scalar::<f32>()?);
    for (name, norm) in gradient_checks {
        println!("gradient_norm\t{name}\t{norm:.8}");
    }
    println!("teacher_source=external_frozen_v0350");
    println!("holdout_consumed=NO");
    println!("historical_validation_consumed=NO");
    println!("historical_test_consumed=NO");
    Ok(())
}
