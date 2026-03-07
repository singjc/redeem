//! Integration tests for the pretrained model registry.

use std::str::FromStr;

use redeem_properties::pretrained::PretrainedModel;

// ---------------------------------------------------------------------------
// PretrainedModel parsing (FromStr)
// ---------------------------------------------------------------------------

#[test]
fn parse_redeem_rt_aliases() {
    for alias in &["redeem-rt", "redeem-rt-cnn-tf", "redeem-rt-cnn", "rt"] {
        let pm = PretrainedModel::from_str(alias)
            .unwrap_or_else(|e| panic!("failed to parse '{}': {}", alias, e));
        assert_eq!(pm, PretrainedModel::RedeemRtCnnTf, "alias = {}", alias);
    }
}

#[test]
fn parse_redeem_ccs_aliases() {
    for alias in &["redeem-ccs", "redeem-ccs-cnn-tf", "redeem-ccs-cnn", "ccs"] {
        let pm = PretrainedModel::from_str(alias)
            .unwrap_or_else(|e| panic!("failed to parse '{}': {}", alias, e));
        assert_eq!(pm, PretrainedModel::RedeemCcsCnnTf, "alias = {}", alias);
    }
}

#[test]
fn parse_alphapeptdeep_rt_aliases() {
    for alias in &[
        "peptdeep-rt",
        "alphapeptdeep-rt",
        "alphapeptdeep-rt-cnn-lstm",
    ] {
        let pm = PretrainedModel::from_str(alias)
            .unwrap_or_else(|e| panic!("failed to parse '{}': {}", alias, e));
        assert_eq!(
            pm,
            PretrainedModel::AlphapeptdeepRtCnnLstm,
            "alias = {}",
            alias
        );
    }
}

#[test]
fn parse_alphapeptdeep_ccs_aliases() {
    for alias in &[
        "peptdeep-ccs",
        "alphapeptdeep-ccs",
        "alphapeptdeep-ccs-cnn-lstm",
    ] {
        let pm = PretrainedModel::from_str(alias)
            .unwrap_or_else(|e| panic!("failed to parse '{}': {}", alias, e));
        assert_eq!(
            pm,
            PretrainedModel::AlphapeptdeepCcsCnnLstm,
            "alias = {}",
            alias
        );
    }
}

#[test]
fn parse_alphapeptdeep_ms2_aliases() {
    for alias in &[
        "peptdeep-ms2",
        "alphapeptdeep-ms2",
        "alphapeptdeep-ms2-bert",
        "ms2",
    ] {
        let pm = PretrainedModel::from_str(alias)
            .unwrap_or_else(|e| panic!("failed to parse '{}': {}", alias, e));
        assert_eq!(
            pm,
            PretrainedModel::AlphapeptdeepMs2Bert,
            "alias = {}",
            alias
        );
    }
}

#[test]
fn parse_unknown_model_errors() {
    let result = PretrainedModel::from_str("nonexistent_model_xyz");
    assert!(result.is_err());
}

#[test]
fn parse_is_case_insensitive() {
    let pm = PretrainedModel::from_str("REDEEM-RT").unwrap();
    assert_eq!(pm, PretrainedModel::RedeemRtCnnTf);

    let pm2 = PretrainedModel::from_str("Alphapeptdeep-CCS").unwrap();
    assert_eq!(pm2, PretrainedModel::AlphapeptdeepCcsCnnLstm);
}

// ---------------------------------------------------------------------------
// Display / arch round-trip
// ---------------------------------------------------------------------------

#[test]
fn display_returns_human_readable_name() {
    let pm = PretrainedModel::RedeemRtCnnTf;
    assert_eq!(pm.to_string(), "redeem-rt-cnn-tf");

    let pm2 = PretrainedModel::AlphapeptdeepMs2Bert;
    assert_eq!(pm2.to_string(), "alphapeptdeep-ms2-bert");
}

#[test]
fn arch_returns_expected_strings() {
    assert_eq!(PretrainedModel::RedeemRtCnnTf.arch(), "rt_cnn_tf");
    assert_eq!(PretrainedModel::RedeemCcsCnnTf.arch(), "ccs_cnn_tf");
    assert_eq!(
        PretrainedModel::AlphapeptdeepRtCnnLstm.arch(),
        "rt_cnn_lstm"
    );
    assert_eq!(
        PretrainedModel::AlphapeptdeepCcsCnnLstm.arch(),
        "ccs_cnn_lstm"
    );
    assert_eq!(PretrainedModel::AlphapeptdeepMs2Bert.arch(), "ms2_bert");
}

#[test]
fn all_variants_have_non_empty_arch() {
    let variants = vec![
        PretrainedModel::RedeemRtCnnTf,
        PretrainedModel::RedeemCcsCnnTf,
        PretrainedModel::AlphapeptdeepRtCnnLstm,
        PretrainedModel::AlphapeptdeepCcsCnnLstm,
        PretrainedModel::AlphapeptdeepMs2Bert,
    ];
    for v in variants {
        assert!(!v.arch().is_empty(), "arch() empty for {:?}", v);
        assert!(!v.to_string().is_empty(), "to_string() empty for {:?}", v);
    }
}
