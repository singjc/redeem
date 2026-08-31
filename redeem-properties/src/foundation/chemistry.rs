//! Lightweight molecular-graph chemistry used by the peptide foundation model.
//!
//! The foundation chemistry layer models residue-local heavy-atom graphs.
//! High-frequency canonical PTMs can apply explicit local graph transformations,
//! while unknown or unsupported modification/site combinations retain a
//! pseudo-mass node fallback. Hydrogens remain implicit to keep graph sizes
//! compact for message passing.

/// Number of raw features emitted for each atom.
pub const ATOM_FEATURE_DIM: usize = 12;

/// Signed elemental-composition delta for a canonical modification.
///
/// Negative counts are required for transformations such as deamidation.  The
/// isotope-specific fields preserve the chemistry of isobaric labels rather
/// than collapsing them into an unexplained mass shift.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ElementalComposition {
    /// Carbon-12 atoms.
    pub carbon: i16,
    /// Carbon-13 atoms.
    pub carbon_13: i16,
    /// Hydrogen-1 atoms.
    pub hydrogen: i16,
    /// Nitrogen-14 atoms.
    pub nitrogen: i16,
    /// Nitrogen-15 atoms.
    pub nitrogen_15: i16,
    /// Oxygen-16 atoms.
    pub oxygen: i16,
    /// Sulfur atoms.
    pub sulfur: i16,
    /// Phosphorus atoms.
    pub phosphorus: i16,
}

/// Coarse attachment scope used when deciding whether a canonical PTM can be
/// represented by an exact local heavy-atom graph transformation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModificationAttachmentSite {
    /// Modification is attached to the residue side chain/backbone atom.
    Residue,
    /// Modification is attached to the peptide N terminus.
    NTerm,
    /// Modification is attached to the peptide C terminus.
    CTerm,
}

/// Canonical PTM transformations for which the current residue-local graph has
/// an explicit heavy-atom topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExactGraphModification {
    /// Cysteine S-carbamidomethylation: `S-CH2-C(=O)-NH2`.
    CarbamidomethylCysteine,
    /// Methionine sulfoxide represented by an oxygen attached to sulfur.
    MethionineOxidation,
    /// Asparagine/glutamine deamidation represented by amide N -> O.
    AsparagineGlutamineDeamidation,
    /// Peptide N-terminal acetylation: `N-C(=O)-CH3`.
    NTerminalAcetylation,
    /// Lysine epsilon-amino acetylation: `N-C(=O)-CH3`.
    LysineAcetylation,
}

/// Resolve whether a canonical UniMod annotation can be represented exactly by
/// the current local heavy-atom graph.
///
/// Unsupported sites deliberately return `None` so callers retain the generic
/// pseudo-mass fallback rather than inventing chemistry.
pub fn exact_graph_modification(
    unimod_id: u32,
    residue: char,
    site: ModificationAttachmentSite,
) -> Option<ExactGraphModification> {
    match (unimod_id, residue, site) {
        (4, 'C', ModificationAttachmentSite::Residue) => {
            Some(ExactGraphModification::CarbamidomethylCysteine)
        }
        (35, 'M', ModificationAttachmentSite::Residue) => {
            Some(ExactGraphModification::MethionineOxidation)
        }
        (7, 'N' | 'Q', ModificationAttachmentSite::Residue) => {
            Some(ExactGraphModification::AsparagineGlutamineDeamidation)
        }
        (1, _, ModificationAttachmentSite::NTerm) => {
            Some(ExactGraphModification::NTerminalAcetylation)
        }
        (1, 'K', ModificationAttachmentSite::Residue) => {
            Some(ExactGraphModification::LysineAcetylation)
        }
        _ => None,
    }
}

/// Canonical metadata retained for a supported UniMod modification.
///
/// This registry is intentionally small and explicit. It provides stable
/// modification identity and elemental composition alongside graph templates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FoundationModificationDefinition {
    /// UniMod accession.
    pub unimod_id: u32,
    /// Canonical short name.
    pub name: &'static str,
    /// Monoisotopic mass delta.
    pub mass_delta: f32,
    /// Signed elemental-composition delta.
    pub composition: ElementalComposition,
}

/// Return canonical metadata for UniMod accessions supported by the foundation
/// loader.
///
/// The compositions mirror UniMod definitions for the currently accepted
/// accessions.  TMT entries retain explicit heavy-isotope atom counts.
pub fn common_unimod_definition(id: u32) -> Option<FoundationModificationDefinition> {
    let (name, mass_delta, composition) = match id {
        1 => (
            "Acetyl",
            42.010_565,
            ElementalComposition {
                carbon: 2,
                hydrogen: 2,
                oxygen: 1,
                ..ElementalComposition::default()
            },
        ),
        4 => (
            "Carbamidomethyl",
            57.021_465,
            ElementalComposition {
                carbon: 2,
                hydrogen: 3,
                nitrogen: 1,
                oxygen: 1,
                ..ElementalComposition::default()
            },
        ),
        7 => (
            "Deamidated",
            0.984_016,
            ElementalComposition {
                hydrogen: -1,
                nitrogen: -1,
                oxygen: 1,
                ..ElementalComposition::default()
            },
        ),
        21 => (
            "Phospho",
            79.966_33,
            ElementalComposition {
                hydrogen: 1,
                oxygen: 3,
                phosphorus: 1,
                ..ElementalComposition::default()
            },
        ),
        35 => (
            "Oxidation",
            15.994_915,
            ElementalComposition {
                oxygen: 1,
                ..ElementalComposition::default()
            },
        ),
        737 => (
            "TMT6plex",
            229.162_93,
            ElementalComposition {
                carbon: 8,
                carbon_13: 4,
                hydrogen: 20,
                nitrogen: 1,
                nitrogen_15: 1,
                oxygen: 2,
                ..ElementalComposition::default()
            },
        ),
        2016 => (
            "TMTpro",
            304.207_15,
            ElementalComposition {
                carbon: 8,
                carbon_13: 7,
                hydrogen: 25,
                nitrogen: 1,
                nitrogen_15: 2,
                oxygen: 3,
                ..ElementalComposition::default()
            },
        ),
        _ => return None,
    };
    Some(FoundationModificationDefinition {
        unimod_id: id,
        name,
        mass_delta,
        composition,
    })
}

/// Chemical element represented in a residue graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Element {
    C,
    N,
    O,
    S,
    Se,
    P,
    /// Generic pseudo atom used when only a modification mass delta is known.
    Pseudo,
}

impl Element {
    fn atomic_number(self) -> f32 {
        match self {
            Self::C => 6.0,
            Self::N => 7.0,
            Self::O => 8.0,
            Self::S => 16.0,
            Self::Se => 34.0,
            Self::P => 15.0,
            Self::Pseudo => 0.0,
        }
    }

    fn atomic_mass(self) -> f32 {
        match self {
            Self::C => 12.011,
            Self::N => 14.007,
            Self::O => 15.999,
            Self::S => 32.06,
            Self::Se => 78.971,
            Self::P => 30.974,
            Self::Pseudo => 0.0,
        }
    }

    fn electronegativity(self) -> f32 {
        match self {
            Self::C => 2.55,
            Self::N => 3.04,
            Self::O => 3.44,
            Self::S => 2.58,
            Self::Se => 2.55,
            Self::P => 2.19,
            Self::Pseudo => 0.0,
        }
    }

    fn typical_valence(self) -> f32 {
        match self {
            Self::C => 4.0,
            Self::N => 3.0,
            Self::O => 2.0,
            Self::S => 2.0,
            Self::Se => 2.0,
            Self::P => 5.0,
            Self::Pseudo => 1.0,
        }
    }
}

/// One atom in a residue-level molecular graph.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomNode {
    /// Chemical element.
    pub element: Element,
    /// Whether the atom belongs to the peptide backbone.
    pub is_backbone: bool,
    /// Whether the atom is aromatic in the residue template.
    pub is_aromatic: bool,
    /// Whether the atom can act as a coarse hydrogen-bond donor.
    pub is_donor: bool,
    /// Whether the atom can act as a coarse hydrogen-bond acceptor.
    pub is_acceptor: bool,
    /// Formal charge used by the canonical graph template.
    pub formal_charge: i8,
    /// Optional mass delta for a pseudo atom representing an unresolved PTM.
    pub pseudo_mass_delta: f32,
}

impl AtomNode {
    /// Convert this atom into the normalized feature vector consumed by Candle.
    pub fn features(&self, is_n_terminal: bool, is_c_terminal: bool) -> [f32; ATOM_FEATURE_DIM] {
        [
            self.element.atomic_number() / 34.0,
            self.element.atomic_mass() / 80.0,
            self.element.electronegativity() / 4.0,
            self.element.typical_valence() / 5.0,
            self.formal_charge as f32 / 2.0,
            if self.is_aromatic { 1.0 } else { 0.0 },
            if self.is_donor { 1.0 } else { 0.0 },
            if self.is_acceptor { 1.0 } else { 0.0 },
            if self.is_backbone { 1.0 } else { 0.0 },
            if self.is_backbone { 0.0 } else { 1.0 },
            if self.element == Element::Pseudo {
                (self.pseudo_mass_delta / 200.0).clamp(-2.0, 2.0)
            } else {
                0.0
            },
            if is_n_terminal {
                1.0
            } else if is_c_terminal {
                -1.0
            } else {
                0.0
            },
        ]
    }
}

/// One undirected bond in a residue-level molecular graph.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BondEdge {
    /// Index of the first atom.
    pub source: usize,
    /// Index of the second atom.
    pub target: usize,
    /// Continuous bond order (1, 1.5, 2, ...).
    pub order: f32,
}

/// Canonical heavy-atom graph for one amino-acid residue.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidueChemicalGraph {
    /// Amino-acid one-letter code.
    pub residue: char,
    /// Atoms retained for the residue.
    pub atoms: Vec<AtomNode>,
    /// Undirected bonds between atoms.
    pub bonds: Vec<BondEdge>,
}

impl ResidueChemicalGraph {
    fn base(residue: char) -> Self {
        let atoms = vec![
            atom(Element::N, true, false, true, true),
            atom(Element::C, true, false, false, false),
            atom(Element::C, true, false, false, false),
            atom(Element::O, true, false, false, true),
        ];
        let bonds = vec![bond(0, 1, 1.0), bond(1, 2, 1.0), bond(2, 3, 2.0)];
        Self {
            residue,
            atoms,
            bonds,
        }
    }

    fn add_atom(&mut self, element: Element, aromatic: bool, donor: bool, acceptor: bool) -> usize {
        let idx = self.atoms.len();
        self.atoms
            .push(atom(element, false, aromatic, donor, acceptor));
        idx
    }

    fn connect(&mut self, source: usize, target: usize, order: f32) {
        self.bonds.push(bond(source, target, order));
    }

    /// Apply one exact canonical PTM heavy-atom transformation.
    ///
    /// Returns `false` if the current residue graph does not contain the anchor
    /// atom expected by the transformation. Callers should then use the generic
    /// pseudo-mass fallback rather than silently dropping the PTM.
    pub fn apply_exact_modification(&mut self, modification: ExactGraphModification) -> bool {
        match modification {
            ExactGraphModification::CarbamidomethylCysteine => {
                let Some(sulfur) = self
                    .atoms
                    .iter()
                    .position(|atom| atom.element == Element::S && !atom.is_backbone)
                else {
                    return false;
                };
                let methylene = self.add_atom(Element::C, false, false, false);
                let carbonyl = self.add_atom(Element::C, false, false, false);
                let oxygen = self.add_atom(Element::O, false, false, true);
                let amide_n = self.add_atom(Element::N, false, true, false);
                self.connect(sulfur, methylene, 1.0);
                self.connect(methylene, carbonyl, 1.0);
                self.connect(carbonyl, oxygen, 2.0);
                self.connect(carbonyl, amide_n, 1.0);
                true
            }
            ExactGraphModification::MethionineOxidation => {
                let Some(sulfur) = self
                    .atoms
                    .iter()
                    .position(|atom| atom.element == Element::S && !atom.is_backbone)
                else {
                    return false;
                };
                let oxygen = self.add_atom(Element::O, false, false, true);
                self.connect(sulfur, oxygen, 1.0);
                true
            }
            ExactGraphModification::AsparagineGlutamineDeamidation => {
                let Some(amide_n) = self
                    .atoms
                    .iter()
                    .position(|atom| atom.element == Element::N && !atom.is_backbone)
                else {
                    return false;
                };
                let atom = &mut self.atoms[amide_n];
                atom.element = Element::O;
                atom.is_aromatic = false;
                atom.is_donor = false;
                atom.is_acceptor = true;
                atom.formal_charge = 0;
                atom.pseudo_mass_delta = 0.0;
                true
            }
            ExactGraphModification::NTerminalAcetylation => {
                if self.atoms.is_empty() || self.atoms[0].element != Element::N {
                    return false;
                }
                self.add_acetyl_to_nitrogen(0);
                true
            }
            ExactGraphModification::LysineAcetylation => {
                let Some(side_chain_n) =
                    self.atoms
                        .iter()
                        .enumerate()
                        .rev()
                        .find_map(|(index, atom)| {
                            (atom.element == Element::N && !atom.is_backbone).then_some(index)
                        })
                else {
                    return false;
                };
                self.add_acetyl_to_nitrogen(side_chain_n);
                true
            }
        }
    }

    fn add_acetyl_to_nitrogen(&mut self, nitrogen: usize) {
        // Acetylation converts an amine into an amide-like nitrogen.
        self.atoms[nitrogen].is_acceptor = false;
        let carbonyl = self.add_atom(Element::C, false, false, false);
        let oxygen = self.add_atom(Element::O, false, false, true);
        let methyl = self.add_atom(Element::C, false, false, false);
        self.connect(nitrogen, carbonyl, 1.0);
        self.connect(carbonyl, oxygen, 2.0);
        self.connect(carbonyl, methyl, 1.0);
    }

    /// Attach a pseudo atom to C-alpha for a modification whose detailed
    /// elemental structure is unavailable but whose mass delta is known.
    pub fn add_mass_delta_modification(&mut self, mass_delta: f32) {
        let idx = self.atoms.len();
        self.atoms.push(AtomNode {
            element: Element::Pseudo,
            is_backbone: false,
            is_aromatic: false,
            is_donor: false,
            is_acceptor: false,
            formal_charge: 0,
            pseudo_mass_delta: mass_delta,
        });
        self.connect(1, idx, 1.0);
    }
}

fn atom(
    element: Element,
    is_backbone: bool,
    is_aromatic: bool,
    is_donor: bool,
    is_acceptor: bool,
) -> AtomNode {
    AtomNode {
        element,
        is_backbone,
        is_aromatic,
        is_donor,
        is_acceptor,
        formal_charge: 0,
        pseudo_mass_delta: 0.0,
    }
}

fn bond(source: usize, target: usize, order: f32) -> BondEdge {
    BondEdge {
        source,
        target,
        order,
    }
}

/// Build the canonical heavy-atom graph for a standard amino acid.
///
/// The graph contains a common `N-CA-C(=O)` backbone and a side-chain
/// topology.  Peptide bonds are represented later by the sequence Transformer;
/// atom message passing is intentionally local to each residue in this first
/// hierarchical implementation.
pub fn residue_graph(residue: char) -> Option<ResidueChemicalGraph> {
    let mut g = ResidueChemicalGraph::base(residue);
    match residue {
        'G' => {}
        'A' => {
            let b = g.add_atom(Element::C, false, false, false);
            g.connect(1, b, 1.0);
        }
        'V' => {
            let b = g.add_atom(Element::C, false, false, false);
            let c1 = g.add_atom(Element::C, false, false, false);
            let c2 = g.add_atom(Element::C, false, false, false);
            g.connect(1, b, 1.0);
            g.connect(b, c1, 1.0);
            g.connect(b, c2, 1.0);
        }
        'L' => {
            let b = g.add_atom(Element::C, false, false, false);
            let c = g.add_atom(Element::C, false, false, false);
            let d1 = g.add_atom(Element::C, false, false, false);
            let d2 = g.add_atom(Element::C, false, false, false);
            g.connect(1, b, 1.0);
            g.connect(b, c, 1.0);
            g.connect(c, d1, 1.0);
            g.connect(c, d2, 1.0);
        }
        'I' => {
            let b = g.add_atom(Element::C, false, false, false);
            let c1 = g.add_atom(Element::C, false, false, false);
            let c2 = g.add_atom(Element::C, false, false, false);
            let d = g.add_atom(Element::C, false, false, false);
            g.connect(1, b, 1.0);
            g.connect(b, c1, 1.0);
            g.connect(b, c2, 1.0);
            g.connect(c1, d, 1.0);
        }
        'S' => {
            let b = g.add_atom(Element::C, false, false, false);
            let o = g.add_atom(Element::O, false, true, true);
            g.connect(1, b, 1.0);
            g.connect(b, o, 1.0);
        }
        'T' => {
            let b = g.add_atom(Element::C, false, false, false);
            let o = g.add_atom(Element::O, false, true, true);
            let c = g.add_atom(Element::C, false, false, false);
            g.connect(1, b, 1.0);
            g.connect(b, o, 1.0);
            g.connect(b, c, 1.0);
        }
        'C' => {
            let b = g.add_atom(Element::C, false, false, false);
            let s = g.add_atom(Element::S, false, true, true);
            g.connect(1, b, 1.0);
            g.connect(b, s, 1.0);
        }
        'M' => {
            let b = g.add_atom(Element::C, false, false, false);
            let c = g.add_atom(Element::C, false, false, false);
            let s = g.add_atom(Element::S, false, false, true);
            let e = g.add_atom(Element::C, false, false, false);
            g.connect(1, b, 1.0);
            g.connect(b, c, 1.0);
            g.connect(c, s, 1.0);
            g.connect(s, e, 1.0);
        }
        'D' => {
            let b = g.add_atom(Element::C, false, false, false);
            let c = g.add_atom(Element::C, false, false, false);
            let o1 = g.add_atom(Element::O, false, false, true);
            let o2 = g.add_atom(Element::O, false, false, true);
            g.connect(1, b, 1.0);
            g.connect(b, c, 1.0);
            g.connect(c, o1, 2.0);
            g.connect(c, o2, 1.0);
        }
        'E' => {
            let b = g.add_atom(Element::C, false, false, false);
            let c = g.add_atom(Element::C, false, false, false);
            let d = g.add_atom(Element::C, false, false, false);
            let o1 = g.add_atom(Element::O, false, false, true);
            let o2 = g.add_atom(Element::O, false, false, true);
            g.connect(1, b, 1.0);
            g.connect(b, c, 1.0);
            g.connect(c, d, 1.0);
            g.connect(d, o1, 2.0);
            g.connect(d, o2, 1.0);
        }
        'N' => {
            let b = g.add_atom(Element::C, false, false, false);
            let c = g.add_atom(Element::C, false, false, false);
            let o = g.add_atom(Element::O, false, false, true);
            let n = g.add_atom(Element::N, false, true, true);
            g.connect(1, b, 1.0);
            g.connect(b, c, 1.0);
            g.connect(c, o, 2.0);
            g.connect(c, n, 1.0);
        }
        'Q' => {
            let b = g.add_atom(Element::C, false, false, false);
            let c = g.add_atom(Element::C, false, false, false);
            let d = g.add_atom(Element::C, false, false, false);
            let o = g.add_atom(Element::O, false, false, true);
            let n = g.add_atom(Element::N, false, true, true);
            g.connect(1, b, 1.0);
            g.connect(b, c, 1.0);
            g.connect(c, d, 1.0);
            g.connect(d, o, 2.0);
            g.connect(d, n, 1.0);
        }
        'K' => {
            let mut prev = 1;
            for _ in 0..4 {
                let c = g.add_atom(Element::C, false, false, false);
                g.connect(prev, c, 1.0);
                prev = c;
            }
            let n = g.add_atom(Element::N, false, true, true);
            g.connect(prev, n, 1.0);
        }
        'R' => {
            let b = g.add_atom(Element::C, false, false, false);
            let c = g.add_atom(Element::C, false, false, false);
            let d = g.add_atom(Element::C, false, false, false);
            let n1 = g.add_atom(Element::N, false, true, true);
            let cz = g.add_atom(Element::C, false, false, false);
            let n2 = g.add_atom(Element::N, false, true, true);
            let n3 = g.add_atom(Element::N, false, true, true);
            for (a, bx) in [(1, b), (b, c), (c, d), (d, n1), (n1, cz)] {
                g.connect(a, bx, 1.0);
            }
            g.connect(cz, n2, 1.5);
            g.connect(cz, n3, 1.5);
        }
        'H' => {
            let b = g.add_atom(Element::C, false, false, false);
            let c = g.add_atom(Element::C, true, false, false);
            let n1 = g.add_atom(Element::N, true, true, true);
            let c2 = g.add_atom(Element::C, true, false, false);
            let n2 = g.add_atom(Element::N, true, true, true);
            let c3 = g.add_atom(Element::C, true, false, false);
            g.connect(1, b, 1.0);
            g.connect(b, c, 1.0);
            g.connect(c, n1, 1.5);
            g.connect(n1, c2, 1.5);
            g.connect(c2, n2, 1.5);
            g.connect(n2, c3, 1.5);
            g.connect(c3, c, 1.5);
        }
        'F' | 'Y' => {
            let b = g.add_atom(Element::C, false, false, false);
            let c1 = g.add_atom(Element::C, true, false, false);
            let c2 = g.add_atom(Element::C, true, false, false);
            let c3 = g.add_atom(Element::C, true, false, false);
            let c4 = g.add_atom(Element::C, true, false, false);
            let c5 = g.add_atom(Element::C, true, false, false);
            let c6 = g.add_atom(Element::C, true, false, false);
            g.connect(1, b, 1.0);
            g.connect(b, c1, 1.0);
            for (a, bx) in [(c1, c2), (c2, c3), (c3, c4), (c4, c5), (c5, c6), (c6, c1)] {
                g.connect(a, bx, 1.5);
            }
            if residue == 'Y' {
                let o = g.add_atom(Element::O, false, true, true);
                g.connect(c4, o, 1.0);
            }
        }
        'W' => {
            let b = g.add_atom(Element::C, false, false, false);
            let c1 = g.add_atom(Element::C, true, false, false);
            let c2 = g.add_atom(Element::C, true, false, false);
            let n = g.add_atom(Element::N, true, true, true);
            let c3 = g.add_atom(Element::C, true, false, false);
            let c4 = g.add_atom(Element::C, true, false, false);
            let c5 = g.add_atom(Element::C, true, false, false);
            let c6 = g.add_atom(Element::C, true, false, false);
            let c7 = g.add_atom(Element::C, true, false, false);
            g.connect(1, b, 1.0);
            g.connect(b, c1, 1.0);
            for (a, bx) in [
                (c1, c2),
                (c2, n),
                (n, c3),
                (c3, c4),
                (c4, c1),
                (c4, c5),
                (c5, c6),
                (c6, c7),
                (c7, c3),
            ] {
                g.connect(a, bx, 1.5);
            }
        }
        'P' => {
            let b = g.add_atom(Element::C, false, false, false);
            let c = g.add_atom(Element::C, false, false, false);
            let d = g.add_atom(Element::C, false, false, false);
            g.connect(1, b, 1.0);
            g.connect(b, c, 1.0);
            g.connect(c, d, 1.0);
            g.connect(d, 0, 1.0);
        }
        _ => return None,
    }
    Some(g)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_standard_residues_have_graphs() {
        for aa in "ACDEFGHIKLMNPQRSTVWY".chars() {
            let graph = residue_graph(aa).expect("standard residue graph");
            assert!(graph.atoms.len() >= 4);
            assert!(graph.bonds.len() >= 3);
        }
    }

    #[test]
    fn unresolved_modification_adds_pseudo_atom() {
        let mut graph = residue_graph('M').unwrap();
        let before = graph.atoms.len();
        graph.add_mass_delta_modification(15.9949);
        assert_eq!(graph.atoms.len(), before + 1);
        assert_eq!(graph.atoms.last().unwrap().element, Element::Pseudo);
    }

    #[test]
    fn canonical_unimod_registry_preserves_identity_and_composition() {
        let phospho = common_unimod_definition(21).unwrap();
        assert_eq!(phospho.name, "Phospho");
        assert_eq!(phospho.composition.phosphorus, 1);
        assert_eq!(phospho.composition.oxygen, 3);

        let tmtpro = common_unimod_definition(2016).unwrap();
        assert_eq!(tmtpro.composition.carbon_13, 7);
        assert_eq!(tmtpro.composition.nitrogen_15, 2);
    }
    #[test]
    fn exact_ptm_graphs_use_real_atoms_not_pseudo_nodes() {
        let mut cysteine = residue_graph('C').unwrap();
        let before_c = cysteine.atoms.len();
        assert!(cysteine.apply_exact_modification(ExactGraphModification::CarbamidomethylCysteine));
        assert_eq!(cysteine.atoms.len(), before_c + 4);
        assert!(cysteine
            .atoms
            .iter()
            .all(|atom| atom.element != Element::Pseudo));

        let mut methionine = residue_graph('M').unwrap();
        let before_m = methionine.atoms.len();
        assert!(methionine.apply_exact_modification(ExactGraphModification::MethionineOxidation));
        assert_eq!(methionine.atoms.len(), before_m + 1);
        assert_eq!(methionine.atoms.last().unwrap().element, Element::O);
    }

    #[test]
    fn deamidation_replaces_side_chain_nitrogen_with_oxygen() {
        let mut asparagine = residue_graph('N').unwrap();
        let before_n = asparagine
            .atoms
            .iter()
            .filter(|atom| atom.element == Element::N)
            .count();
        let before_o = asparagine
            .atoms
            .iter()
            .filter(|atom| atom.element == Element::O)
            .count();
        assert!(asparagine
            .apply_exact_modification(ExactGraphModification::AsparagineGlutamineDeamidation));
        assert_eq!(
            asparagine
                .atoms
                .iter()
                .filter(|atom| atom.element == Element::N)
                .count(),
            before_n - 1
        );
        assert_eq!(
            asparagine
                .atoms
                .iter()
                .filter(|atom| atom.element == Element::O)
                .count(),
            before_o + 1
        );
    }

    #[test]
    fn exact_modification_support_is_site_specific() {
        assert_eq!(
            exact_graph_modification(4, 'C', ModificationAttachmentSite::Residue),
            Some(ExactGraphModification::CarbamidomethylCysteine)
        );
        assert_eq!(
            exact_graph_modification(35, 'M', ModificationAttachmentSite::Residue),
            Some(ExactGraphModification::MethionineOxidation)
        );
        assert_eq!(
            exact_graph_modification(7, 'Q', ModificationAttachmentSite::Residue),
            Some(ExactGraphModification::AsparagineGlutamineDeamidation)
        );
        assert_eq!(
            exact_graph_modification(1, 'A', ModificationAttachmentSite::NTerm),
            Some(ExactGraphModification::NTerminalAcetylation)
        );
        assert_eq!(
            exact_graph_modification(1, 'K', ModificationAttachmentSite::Residue),
            Some(ExactGraphModification::LysineAcetylation)
        );
        assert_eq!(
            exact_graph_modification(35, 'W', ModificationAttachmentSite::Residue),
            None
        );
    }
}
