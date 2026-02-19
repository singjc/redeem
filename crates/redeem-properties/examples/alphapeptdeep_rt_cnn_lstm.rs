use anyhow::{Context, Result};
use candle_core::Device;
use redeem_properties::{
    models::{
        model_interface::{ModelInterface, PredictionResult},
        rt_cnn_lstm_model::RTCNNLSTMModel,
    },
    utils::{
        data_handling::{PeptideData, TargetNormalization},
        peptdeep_utils::{
            get_modification_indices, get_modification_string, load_modifications,
            remove_mass_shift,
        },
    },
};
use std::path::PathBuf;

fn run_prediction(model: &mut RTCNNLSTMModel, batch_data: &[PeptideData]) -> Result<()> {
    let batch = redeem_properties::utils::data_handling::PeptideBatchData::from(batch_data);

    let predictions = model.predict(
        &batch.naked_sequence,
        &batch.mods,
        &batch.mod_sites,
        if batch.charges.iter().all(|c| c.is_some()) {
            Some(batch.charges.iter().map(|c| c.unwrap()).collect())
        } else {
            None
        },
        None,
        if batch.instruments.iter().all(|i| i.is_some()) {
            Some(
                batch
                    .instruments
                    .iter()
                    .map(|opt| opt.as_ref().map(|a| a.clone()))
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        },
    )?;

    if let PredictionResult::RTResult(rt_preds) = predictions {
        let total_error: f32 = rt_preds
            .iter()
            .zip(batch.retention_times.iter())
            .map(|(pred, obs)| (pred - obs.unwrap_or_default()).abs())
            .sum();

        for ((seq, pred), obs) in batch
            .naked_sequence
            .iter()
            .zip(rt_preds.iter())
            .zip(batch.retention_times.iter())
        {
            println!(
                "Peptide: {}, Predicted RT: {:.6}, Observed RT: {:.6}",
                std::str::from_utf8(seq).unwrap_or(""),
                pred,
                obs.unwrap_or_default()
            );
        }

        println!(
            "Mean Absolute Error: {:.6}",
            total_error / rt_preds.len() as f32
        );
    }

    Ok(())
}

fn main() -> Result<()> {
    env_logger::init();

    let model_path =
        PathBuf::from("/home/singjc/Documents/github/redeem/rt_fine_tuned.safetensors");
    let constants_path = PathBuf::from("/home/singjc/Documents/github/redeem/crates/redeem-properties/data/models/alphapeptdeep/generic/rt.pth.model_const.yaml");
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    println!("Device: {:?}", device);

    let mut model = RTCNNLSTMModel::new(&model_path, Some(&constants_path), 0, 8, 4, true, device)
        .context("Failed to create RTCNNLSTMModel")?;

    let modifications = load_modifications().context("Failed to load modifications")?;

    let training_data: Vec<PeptideData> = vec![
        "AKPLMELIER",
        "TEM[+15.9949]VTISDASQR",
        "AGKFPSLLTHNENMVAK",
        "LSELDDRADALQAGASQFETSAAK",
        "FLLQDTVELR",
        "SVTEQGAELSNEER",
        "EHALLAYTLGVK",
        "TVQSLEIDLDSM[+15.9949]R",
        "VVSQYSSLLSPMSVNAVM[+15.9949]K",
        "TFLALINQVFPAEEDSKK",
    ]
    .into_iter()
    .enumerate()
    .map(|(i, seq)| {
        let naked = remove_mass_shift(seq);
        let mods = get_modification_string(seq, &modifications);
        let sites = get_modification_indices(seq);
        PeptideData::new(
            seq,
            &naked,
            &mods,
            &sites,
            None,
            None,
            None,
            None,
            Some(i as f32 / 10.0),
            None,
            None,
            None,
        )
    })
    .collect();

    let test_peptides = vec![
        ("QPYAVSELAGHQTSAESWGTGR", "", "", 0.4328955),
        ("GMSVSDLADKLSTDDLNSLIAHAHR", "Oxidation@M", "1", 0.6536107),
        (
            "TVQHHVLFTDNMVLICR",
            "Oxidation@M;Carbamidomethyl@C",
            "11;15",
            0.7811949,
        ),
        ("EAELDVNEELDKK", "", "", 0.2934583),
        ("YTPVQQGPVGVNVTYGGDPIPK", "", "", 0.5863009),
    ];

    let prediction_data: Vec<PeptideData> = test_peptides
        .into_iter()
        .map(|(seq, mods, sites, rt)| {
            let naked = remove_mass_shift(seq);
            PeptideData::new(
                seq,
                &naked,
                mods,
                sites,
                None,
                None,
                None,
                None,
                Some(rt),
                None,
                None,
                None,
            )
        })
        .collect();

    run_prediction(&mut model, &prediction_data)?;

    model.fine_tune(
        &training_data,
        modifications,
        10,
        0.001,
        5,
        TargetNormalization::None,
        None,
    )?;

    run_prediction(&mut model, &prediction_data)?;

    model.save("alphapeptdeep_rt_cnn_lstm_finetuned.safetensors")?;

    Ok(())
}
