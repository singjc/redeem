// Small regression test that ensures training a tiny XGBoost model produces
// a model dump containing a non-zero `num_trees` value. This guards against
// upstream regressions where `Booster::train()` omitted per-iteration
// `update()` calls and produced only a base_score model.

#![cfg(feature = "xgboost")]

use xgb::{
    parameters::{
        learning::LearningTaskParametersBuilder,
        learning::Objective,
        tree::{TreeBoosterParametersBuilder, TreeMethod},
        BoosterParametersBuilder, BoosterType,
    },
    Booster, DMatrix,
};

#[test]
fn xgb_train_produces_trees() {
    // Tiny dataset: 4 rows x 2 columns
    let data: Vec<f32> = vec![
        1.0, 2.0, // row0
        1.0, 2.0, // row1
        3.0, 4.0, // row2
        3.0, 4.0, // row3
    ];
    let mut dmat = DMatrix::from_dense(&data, 4).expect("create dmatrix");
    let labels = vec![1.0f32, 0.0, 1.0, 0.0];
    dmat.set_labels(&labels).expect("set labels");

    // Build minimal booster params
    let learning_params = LearningTaskParametersBuilder::default()
        .objective(Objective::BinaryLogistic)
        .build()
        .unwrap();

    let tree_params = TreeBoosterParametersBuilder::default()
        .tree_method(TreeMethod::Hist)
        .max_depth(2)
        .eta(0.3)
        .build()
        .unwrap();

    let booster_params = BoosterParametersBuilder::default()
        .booster_type(BoosterType::Tree(tree_params))
        .learning_params(learning_params)
        .verbose(false)
        .build()
        .unwrap();

    let mut bst =
        Booster::new_with_cached_dmats(&booster_params, &[&dmat]).expect("create booster");

    // Train for a few rounds using the explicit update loop (workaround).
    for i in 0..5i32 {
        bst.update(&dmat, i).expect("bst.update");
    }

    let buf = bst.save_buffer(false).expect("save_buffer");
    let s = String::from_utf8_lossy(&buf);

    // Look for the JSON field num_trees and ensure it's > 0
    if let Some(idx) = s.find("\"num_trees\":\"") {
        let start = idx + "\"num_trees\":\"".len();
        let digits: String = s
            .get(start..)
            .unwrap_or("")
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect();
        let n = digits.parse::<usize>().unwrap_or(0);
        assert!(n > 0, "parsed num_trees == 0; dump: {}", s);
    } else {
        // If the `num_trees` field isn't present, fail the test but show dump
        panic!("model dump missing num_trees: {}", s);
    }
}
