use std::sync::Arc;

#[derive(Debug, Clone, Copy)]
pub enum TargetNormalization {
    ZScore(f32, f32), // mean, std
    MinMax(f32, f32), // min, max
    None,
}

impl TargetNormalization {
    pub fn from_str(norm: Option<String>) -> Self {
        match norm.as_deref() {
            Some("z_score") => TargetNormalization::ZScore(0.0, 0.0),
            Some("min_max") => TargetNormalization::MinMax(0.0, 0.0),
            _ => TargetNormalization::None,
        }
    }
}

#[derive(Clone)]
pub struct PeptideData {
    pub modified_sequence: Arc<[u8]>, // e.g., "(UniMod:1)M(UniMod:35)AAAATMAAAAR"
    pub naked_sequence: Arc<[u8]>,    // e.g., "MAAAATMAAAAR"
    pub mods: Arc<[u8]>,              // e.g., "Any_N-term;Oxidation@M"
    pub mod_sites: Arc<[u8]>,         // e.g., "0;1"
    pub charge: Option<i32>,
    pub precursor_mass: Option<f32>,
    pub nce: Option<i32>,
    pub instrument: Option<Arc<[u8]>>,
    pub retention_time: Option<f32>,
    pub ion_mobility: Option<f32>,
    pub ccs: Option<f32>,
    pub ms2_intensities: Option<Vec<Vec<f32>>>,
}

impl PeptideData {
    pub fn new(
        modified_sequence: &str,
        naked_sequence: &str,
        mods: &str,
        mod_sites: &str,
        charge: Option<i32>,
        precursor_mass: Option<f32>,
        nce: Option<i32>,
        instrument: Option<&str>,
        retention_time: Option<f32>,
        ion_mobility: Option<f32>,
        ccs: Option<f32>,
        ms2_intensities: Option<Vec<Vec<f32>>>,
    ) -> Self {
        Self {
            modified_sequence: Arc::from(modified_sequence.as_bytes().to_vec().into_boxed_slice()),
            naked_sequence: Arc::from(naked_sequence.as_bytes().to_vec().into_boxed_slice()),
            mods: Arc::from(mods.as_bytes().to_vec().into_boxed_slice()),
            mod_sites: Arc::from(mod_sites.as_bytes().to_vec().into_boxed_slice()),
            charge,
            precursor_mass,
            nce,
            instrument: instrument.map(|s| Arc::from(s.as_bytes().to_vec().into_boxed_slice())),
            retention_time,
            ion_mobility,
            ccs,
            ms2_intensities,
        }
    }

    pub fn modified_sequence_str(&self) -> &str {
        std::str::from_utf8(&self.modified_sequence).unwrap_or("")
    }

    pub fn naked_sequence_str(&self) -> &str {
        std::str::from_utf8(&self.naked_sequence).unwrap_or("")
    }

    pub fn mods_str(&self) -> &str {
        std::str::from_utf8(&self.mods).unwrap_or("")
    }

    pub fn mod_sites_str(&self) -> &str {
        std::str::from_utf8(&self.mod_sites).unwrap_or("")
    }

    pub fn instrument_str(&self) -> Option<&str> {
        self.instrument
            .as_ref()
            .map(|v| std::str::from_utf8(v).unwrap_or(""))
    }
}

pub struct PeptideBatchData {
    pub naked_sequence: Vec<Arc<[u8]>>,
    pub mods: Vec<Arc<[u8]>>,
    pub mod_sites: Vec<Arc<[u8]>>,
    pub charges: Vec<Option<i32>>,
    pub precursor_masses: Vec<Option<f32>>,
    pub nces: Vec<Option<i32>>,
    pub instruments: Vec<Option<Arc<[u8]>>>,
    pub retention_times: Vec<Option<f32>>,
    pub ion_mobilities: Vec<Option<f32>>,
    pub ccs: Vec<Option<f32>>,
    pub ms2_intensities: Vec<Option<Vec<Vec<f32>>>>,
}

impl From<&[PeptideData]> for PeptideBatchData {
    fn from(slice: &[PeptideData]) -> Self {
        Self {
            naked_sequence: slice
                .iter()
                .map(|p| Arc::clone(&p.naked_sequence))
                .collect(),
            mods: slice.iter().map(|p| Arc::clone(&p.mods)).collect(),
            mod_sites: slice.iter().map(|p| Arc::clone(&p.mod_sites)).collect(),
            charges: slice.iter().map(|p| p.charge).collect(),
            precursor_masses: slice.iter().map(|p| p.precursor_mass).collect(),
            nces: slice.iter().map(|p| p.nce).collect(),
            instruments: slice.iter().map(|p| p.instrument.clone()).collect(),
            retention_times: slice.iter().map(|p| p.retention_time).collect(),
            ion_mobilities: slice.iter().map(|p| p.ion_mobility).collect(),
            ccs: slice.iter().map(|p| p.ccs).collect(),
            ms2_intensities: slice.iter().map(|p| p.ms2_intensities.clone()).collect(),
        }
    }
}

impl PeptideBatchData {
    pub fn naked_sequence_strs(&self) -> Vec<&str> {
        self.naked_sequence
            .iter()
            .map(|s| std::str::from_utf8(s).unwrap_or(""))
            .collect()
    }

    pub fn mods_strs(&self) -> Vec<&str> {
        self.mods
            .iter()
            .map(|s| std::str::from_utf8(s).unwrap_or(""))
            .collect()
    }

    pub fn mod_sites_strs(&self) -> Vec<&str> {
        self.mod_sites
            .iter()
            .map(|s| std::str::from_utf8(s).unwrap_or(""))
            .collect()
    }

    pub fn instrument_strs(&self) -> Vec<Option<&str>> {
        self.instruments
            .iter()
            .map(|opt| opt.as_ref().map(|s| std::str::from_utf8(s).unwrap_or("")))
            .collect()
    }
}
