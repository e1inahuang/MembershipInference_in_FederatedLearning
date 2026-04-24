# MembershipInference_in_FederatedLearning

Membership Inference Attacks in Federated Learning
Investigating the privacy-utility trade-off in federated learning through membership inference attacks (MIA) on healthcare data.
Authors: Elina Yutong Huang, Isabel Agadagba, Cole Callahan
Course: Engineering Privacy, Carnegie Mellon University
Overview
Federated learning (FL) promises collaborative model training without sharing raw data — but how private is it really? This project puts that claim to the test by launching black-box membership inference attacks against FL models trained on medical data, measuring whether an adversary can determine if a specific patient record was used during training.
We find that a fine-tuned federated model can outperform its centralized counterpart in accuracy (75.97% vs. 70.13%) while simultaneously exhibiting higher vulnerability to membership inference — a result with real implications for healthcare ML deployments.
Key Findings

Fine-tuned FL > Centralized: Careful hyperparameter tuning (15 clients, balanced data, batch size 16, lr=0.01) pushed the federated model past centralized performance, challenging the assumption that privacy always costs utility.
FL is more vulnerable to MIA: The federated model showed an attacker advantage of 0.2497 vs. 0.1356 for the centralized model — FL's distributed nature alone is not sufficient privacy protection.
Data skew matters: Unbalanced client data introduced significant performance volatility, highlighting that real-world institutional data heterogeneity is a critical design consideration.
Tuning > Complexity: Performance gains came from fitting the right configuration, not from architectural changes.

Dataset
Pima Indians Diabetes Database — 768 samples, 8 clinical features (glucose, BMI, age, etc.), binary classification (diabetic vs. non-diabetic). Chosen for its health sensitivity and suitability for privacy studies.
Architecture
Federated Learning Pipeline

Built with TensorFlow Federated (TFF)
Keras sequential model with dense layers, ReLU, dropout, sigmoid output
Federated Averaging (tff.learning.build_federated_averaging_process)
Data partitioned across simulated clients (3 / 7 / 15 configurations tested)

Hyperparameter Sweep
24 configurations tested across:
ParameterValuesClient count3, 7, 15Data distributionBalanced, SkewedBatch size16, 32Learning rate0.001, 0.01
Best config: 15 clients, balanced, batch 16, lr=0.01 → 75.97% accuracy, AUC 0.8265
Membership Inference Attack

Library: tensorflow_privacy.privacy.privacy_tests.membership_inference_attack
Strategies: Threshold Attack, Logistic Regression
Input: Prediction probabilities + true labels from train/test splits
Originally explored IBM's Adversarial Robustness Toolbox (ART), switched to TF-Privacy due to TFF/TF version constraints

Results Summary
ModelAccuracyAttacker AdvantageAUC (MIA)Centralized70.13%0.13560.5070Federated (baseline)74.03%0.24970.4556Federated (fine-tuned)75.97%——
Fine-tuned MIA results pending due to TF/NumPy dependency conflicts in the privacy test suite.
Project Structure
├── MembershipInferenceFL.ipynb   # NIST Genomics PPFL red team exercise (ART-based MIA)
├── federated_learning.ipynb      # Main FL pipeline: training, tuning, MIA evaluation
├── attack_targets/               # Pre-trained target models (CNN, DPCNN variants)
├── utils.py                      # Data loading, model loading, attack model helpers
└── README.md
Setup
bashpip install tensorflow tensorflow-federated tensorflow-privacy
pip install numpy pandas matplotlib seaborn scikit-learn
The notebook runs in Google Colab with standard GPU runtime.
Tech Stack

TensorFlow / TensorFlow Federated (TFF)
TensorFlow Privacy (MIA module)
IBM Adversarial Robustness Toolbox (ART) — for NIST genomics exercise
PyTorch + torchvision (NIST attack models)
pandas, NumPy, matplotlib, seaborn

Limitations & Future Work

Complete MIA evaluation on the fine-tuned federated model (blocked by dependency conflicts)
Integrate differential privacy (DP-SGD) into the federated pipeline
Test on larger, more diverse medical datasets
Explore white-box attack strategies for deeper privacy analysis
Investigate personalized federated learning techniques

License
Academic project — Carnegie Mellon University, Spring 2025.
