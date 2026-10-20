//! Development-Validation-only spectrum perturbation; no model training.
//! Preserves recipient peptide, precursor context, labels, and causal targets.
//! Only observed product-ion peak input is replaced with a different DEV64 record.
use anyhow::{Context, Result, bail};
use candle_core::Device;
use redeem_properties::foundation::{
    FoundationModel, FoundationPartition, FoundationSpectrum, FoundationTrainingRecord,
    ObservedSpectrumPeak, load_foundation_records_from_run_partition,
};
use std::{collections::BTreeMap, env, fs::OpenOptions, io::{BufWriter, Write}, path::Path};

const DEV_COUNT: usize = 64;

type Record = FoundationTrainingRecord;

fn observed(record: &Record) -> Result<FoundationSpectrum> {
    let spectrum = FoundationSpectrum::from_training_record(record)
        .context("record has no observed product-ion spectrum")?;
    if spectrum.peaks.is_empty() { bail!("empty observed spectrum"); }
    Ok(spectrum)
}

fn swap_observed_peaks(recipient: &Record, donor: &Record) -> Result<Record> {
    let donor_spectrum = observed(donor)?;
    let recipient_spectrum = observed(recipient)?;
    if donor_spectrum == recipient_spectrum {
        bail!("matched and donor peak lists are identical; invalid perturbation");
    }
    let mut swapped = recipient.clone();
    swapped.observed_spectrum_peaks = donor_spectrum.peaks.iter().map(|p| ObservedSpectrumPeak {
        mz: p.mz, intensity: p.intensity,
    }).collect();
    // `FoundationSpectrum::from_training_record` preferentially selects raw
    // observed peaks. The original forward-label fragments remain untouched.
    if observed(&swapped)? != donor_spectrum { bail!("donor spectrum not substituted exactly"); }
    let mut restored = swapped.clone();
    restored.observed_spectrum_peaks = recipient.observed_spectrum_peaks.clone();
    if &restored != recipient { bail!("swap changed non-peak fields, including precursor/target"); }
    Ok(swapped)
}

// Deterministic one-to-one donor permutation. Within charge strata, rotate
// records sorted by precursor m/z. Singleton charge strata are rotated with
// each other; a lone singleton exchanges donor assignments with another stratum.
// No train/test/holdout records participate.
fn donors(records: &[Record]) -> Result<Vec<usize>> {
    if records.len() != DEV_COUNT { bail!("exactly {DEV_COUNT} DEV records required"); }
    let spectra = records.iter().map(observed).collect::<Result<Vec<_>>>()?;
    let mut groups: BTreeMap<Option<i32>, Vec<usize>> = BTreeMap::new();
    for (i, r) in records.iter().enumerate() {
        groups.entry(r.context.charge).or_default().push(i);
    }
    let mut map = vec![usize::MAX; records.len()];
    let mut singleton = Vec::new();
    for mut group in groups.into_values() {
        group.sort_by(|&a, &b| {
            records[a].context.precursor_mz.unwrap_or(f32::INFINITY)
                .total_cmp(&records[b].context.precursor_mz.unwrap_or(f32::INFINITY))
                .then_with(|| a.cmp(&b))
        });
        if group.len() == 1 { singleton.extend(group); continue; }
        for pos in 0..group.len() {
            map[group[pos]] = group[(pos + 1) % group.len()];
        }
    }
    if singleton.len() > 1 {
        for pos in 0..singleton.len() {
            map[singleton[pos]] = singleton[(pos + 1) % singleton.len()];
        }
    } else if let Some(&s) = singleton.first() {
        // All other strata have at least two elements. Exchange two donor
        // assignments, preserving a permutation with no self-pairs.
        let t = map.iter().position(|&d| d != usize::MAX && d != s)
            .context("no valid cross-charge donor for singleton")?;
        map[s] = map[t];
        map[t] = s;
    }
    // Repair rare repeated-peptide or repeated-peak pairs by exchanging donor
    // assignments. Prefer exchanges that preserve charge compatibility.
    let valid = |i: usize, j: usize| -> bool {
        i != j && records[i].peptidoform != records[j].peptidoform && spectra[i] != spectra[j]
    };
    for i in 0..records.len() {
        if valid(i, map[i]) { continue; }
        let candidate = (0..records.len())
            .filter(|&k| k != i && valid(i, map[k]) && valid(k, map[i]))
            .min_by_key(|&k| {
                usize::from(records[i].context.charge != records[map[k]].context.charge) +
                usize::from(records[k].context.charge != records[map[i]].context.charge)
            })
            .context("cannot construct nonidentical-peptidoform/spectrum DEV donor pairing")?;
        map.swap(i, candidate);
    }
    let mut unique = map.clone();
    unique.sort_unstable();
    if unique != (0..records.len()).collect::<Vec<_>>() ||
       map.iter().enumerate().any(|(i, &j)| i == j || j >= records.len()) {
        bail!("donor assignment is not a strict derangement");
    }
    for (i, &j) in map.iter().enumerate() {
        if records[i].peptidoform == records[j].peptidoform {
            bail!("donor has identical peptidoform identity at recipient {i}; invalid control");
        }
        swap_observed_peaks(&records[i], &records[j])?;
    }
    Ok(map)
}

fn peak_signature(s: &FoundationSpectrum) -> u64 {
    // FNV-1a diagnostic fingerprint of raw peak float bits; not cryptographic.
    let mut h: u64 = 0xcbf29ce484222325;
    for p in &s.peaks {
        for byte in p.mz.to_bits().to_le_bytes().into_iter()
            .chain(p.intensity.to_bits().to_le_bytes()) {
            h = (h ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
    }
    h
}

fn cleaned(s: &str) -> String { s.replace('\t', " ").replace('\n', " ").replace('\r', " ") }
fn float_text(v: Option<f32>) -> String {
    v.map(|x| format!("{x:.9}")).unwrap_or_default()
}
fn run(run_yaml: &Path, checkpoint: &Path, output: &Path) -> Result<()> {
    if output.exists() { bail!("refusing to overwrite {}", output.display()); }
    let mut records = load_foundation_records_from_run_partition(run_yaml, FoundationPartition::Validation)?;
    records.retain(|r| FoundationSpectrum::from_training_record(r).is_some());
    records.truncate(DEV_COUNT);
    if records.len() != DEV_COUNT { bail!("expected {DEV_COUNT} spectrum-bearing DEV records"); }
    let donor_map = donors(&records)?;
    let device = Device::new_cuda(0).context("A100 CUDA required; no CPU fallback")?;
    let model = FoundationModel::load(checkpoint, device)?;
    // Create output exclusively after all input checks succeeded. Errors leave
    // a partial file for inspection; reruns refuse to replace it.
    let file = OpenOptions::new().write(true).create_new(true).open(output)?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "record_index\tdonor_index\tsequence\tdonor_sequence\tcharge\tdonor_charge\tprecursor_mz\tdonor_precursor_mz\tmatched_peaks\tswapped_peaks\tmatched_peak_sig\tdonor_peak_sig\tswapped_peak_sig\tmatched_nll\tswapped_nll\tdelta_nll\tmatched_generated\tswapped_generated")?;
    let mut positive = 0usize;
    for (i, recipient) in records.iter().enumerate() {
        let j = donor_map[i];
        let donor = &records[j];
        let swapped = swap_observed_peaks(recipient, donor)?;
        let matched_nll = model.inverse_mean_nll(recipient)?.context("matched NLL unavailable")?;
        let swapped_nll = model.inverse_mean_nll(&swapped)?.context("swapped NLL unavailable")?;
        if !matched_nll.is_finite() || !swapped_nll.is_finite() {
            bail!("nonfinite inverse NLL at DEV record {i}");
        }
        let matched_generated = model.generate_peptide(recipient)?.sequence;
        let swapped_generated = model.generate_peptide(&swapped)?.sequence;
        let delta = swapped_nll - matched_nll;
        if delta > 0.0 { positive += 1; }
        writeln!(writer, "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.9}\t{:.9}\t{:.9}\t{}\t{}",
            i, j, cleaned(&recipient.peptidoform.sequence), cleaned(&donor.peptidoform.sequence),
            recipient.context.charge.map(|v| v.to_string()).unwrap_or_default(),
            donor.context.charge.map(|v| v.to_string()).unwrap_or_default(),
            float_text(recipient.context.precursor_mz), float_text(donor.context.precursor_mz),
            observed(recipient)?.peaks.len(), observed(&swapped)?.peaks.len(),
            peak_signature(&observed(recipient)?), peak_signature(&observed(donor)?),
            peak_signature(&observed(&swapped)?),
            matched_nll, swapped_nll, delta, cleaned(&matched_generated), cleaned(&swapped_generated))?;
    }
    writer.flush()?;
    println!("REDEEM_PAIRED_SPECTRUM_SWAP=PASS records={DEV_COUNT} positive_delta={positive}");
    println!("output={}", output.display());
    Ok(())
}

fn self_test() -> Result<()> {
    use redeem_properties::foundation::{PeptidoformInput, TrainingContext, RetentionTimeLabels};
    let mut r = Record {
        peptidoform: PeptidoformInput { sequence: "PEPTIDE".into(), modifications: vec![] },
        retention_time: RetentionTimeLabels::default(), ccs: None,
        fragments: vec![], observed_spectrum_peaks: vec![ObservedSpectrumPeak { mz: 123.0, intensity: 1.0 }],
        context: TrainingContext { charge: Some(2), precursor_mz: Some(512.1), ..Default::default() },
        run_id: Some("DEV-only".into()),
    };
    let mut donor = r.clone();
    donor.context.charge = Some(3);
    donor.context.precursor_mz = Some(724.5);
    donor.peptidoform.sequence = "SEQUENCE".into();
    donor.observed_spectrum_peaks = vec![ObservedSpectrumPeak { mz: 456.0, intensity: 4.0 }];
    let changed = swap_observed_peaks(&r, &donor)?;
    if changed.context != r.context || changed.peptidoform != r.peptidoform ||
       changed.fragments != r.fragments || observed(&changed)? != observed(&donor)? {
        bail!("spectrum swap invariant self-test failed");
    }
    if swap_observed_peaks(&r, &r).is_ok() { bail!("identical-spectrum refusal test failed"); }
    let mut records = Vec::new();
    for i in 0..DEV_COUNT {
        r.context.charge = Some(if i < 32 { 2 } else if i < 63 { 3 } else { 4 });
        r.context.precursor_mz = Some(400.0 + i as f32);
        r.observed_spectrum_peaks[0].mz = 100.0 + i as f32;
        r.peptidoform.sequence = format!("PEPTIDE{i}");
        records.push(r.clone());
    }
    // Even if source records repeat a peptide identity, no recipient may
    // receive an identical-peptidoform donor.
    records[1].peptidoform = records[0].peptidoform.clone();
    let pairing = donors(&records)?;
    for (i,j) in pairing.into_iter().enumerate() {
        if i == j || records[i].peptidoform == records[j].peptidoform {
            bail!("invalid DEV donor at {i}");
        }
    }
    println!("REDEEM_PAIRED_SPECTRUM_SWAP_SELFTEST=PASS");
    Ok(())
}

fn main() -> Result<()> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.len() == 1 && args[0] == "--self-test" { return self_test(); }
    if args.len() != 3 { bail!("usage: foundation_paired_spectrum_swap RUN.yaml CHECKPOINT_DIR NEW_OUTPUT.tsv | --self-test"); }
    // No options for TRAIN or protected partitions. All runs select DEV64.
    run(Path::new(&args[0]), Path::new(&args[1]), Path::new(&args[2]))
}
