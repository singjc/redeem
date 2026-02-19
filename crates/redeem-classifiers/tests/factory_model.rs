use redeem_classifiers::config::{ModelConfig, ModelType};
use redeem_classifiers::math::{Array1, Array2};
use redeem_classifiers::models::factory;

#[test]
fn test_factory_builds_and_predicts() {
    // tiny dataset
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![
            1.0, 0.0, // class 1
            0.0, 1.0, // class -1
            1.0, 0.1, // class 1
            0.0, 0.9, // class -1
            1.1, 0.0, // class 1
            0.0, 1.2, // class -1
        ],
    )
    .expect("failed to create feature matrix");

    let y = Array1::from_vec(vec![1i32, -1i32, 1i32, -1i32, 1i32, -1i32]);
    // convert to 0/1 as some models expect
    let y01: Vec<i32> = y.iter().map(|&v| if v == 1 { 1 } else { 0 }).collect();

    let params = ModelConfig {
        learning_rate: 0.1,
        model_type: ModelType::GBDT {
            max_depth: 3,
            num_boost_round: 3,
            debug: false,
            training_optimization_level: 2,
            loss_type: "LogLikelyhood".to_string(),
        },
    };

    let mut model = factory::build_model(params);
    model.fit(&x, &y01, None, None);
    let probs = model.predict_proba(&x);
    assert_eq!(probs.len(), x.nrows());
}
