<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/singjc/redeem/raw/master/img/redeem_logo.png" alt="ReDeem_Logo" width="500">
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/singjc/redeem/raw/master/img/redeem_logo.png" alt="ReDeem_Logo" width="500">
    <img alt="ReDeem Logo" comment="Placeholder to transition between light color mode and dark color mode - this image is not directly used." src="https://github.com/singjc/redeem/raw/master/img/redeem_logo.png">
  </picture>
</p>

---

# ReDeeM: Repository for Deep Learning Models for Mass Spectrometry

ReDeeM is a Rust crate designed for implementing deep learning models specifically tailored for mass spectrometry data. The primary goal of this project is to facilitate the prediction of peptide properties and to develop classifier scoring models (TDA).

### Current Crates

The ReDeeM project consists of two primary crates:

1. **redeem-properties**: 
   - This crate focuses on deep learning models for peptide property prediction. It implements models for predicting retention time (RT), ion mobility (IM), and MS2 fragment intensities using the Candle library.
   
   - Current Models
  
    Model | Name | Architecture | Implemented
    --- | --- | --- | ---
    AlphaPept RT Model | `redeem_properties::RTCNNLSTMModel` | CNN-LSTM | :heavy_check_mark:
    AlphaPept MS2 Model | `redeem_properties::MS2BertModel` | Bert | :heavy_check_mark:
    AlphaPept IM Model | `redeem_properties::CCSCNNLSTMModel` | CNN-LSTM | :heavy_check_mark:

2. **redeem-classifiers**:
   - This crate is aimed at developing semi-supervised scoring classifier models. The goal is to create models for separating target peptides from decoys.
  
   - Current Models
  
    Model | Name | Architecture | Implemented
    --- | --- | --- | ---
    XGBoost Classifier | `redeem_classifiers::XGBoostClassifier` | XGBoost | :heavy_check_mark: