# 🛡️ FedShield AI
### Privacy-Preserving Federated Learning for Medical Image Classification

FedShield AI is a federated learning research prototype that demonstrates how multiple hospitals can collaboratively train an AI model for medical image classification without sharing their raw patient images.

Built as a solo hackathon project, FedShield AI explores federated learning, differential privacy, optimization, and defenses against malicious model updates through an interactive Streamlit dashboard.

> **Disclaimer:** This is an educational research prototype using a public dataset and simulated hospitals. It is not a clinically validated medical system.

##  The Problem

Medical institutions hold valuable imaging datasets, but sharing patient data can raise privacy, security, and regulatory concerns.

Traditional centralized machine learning requires collecting training data in one location. Federated learning offers an alternative: institutions train models locally and share model updates rather than raw images.

However, federated learning introduces additional challenges:

- Different hospitals may have different data distributions.
- Model updates may reveal information about training records.
- Malicious participants may attempt to poison the global model.
- Federated optimization can be less stable than centralized training.

##  Our Solution

FedShield AI simulates a network of four hospitals collaborating to train a convolutional neural network (CNN) for medical image classification.

Each hospital trains locally on its own subset of the dataset. A federated server combines the resulting model parameters to improve the shared global model.

The project also investigates differential privacy, FedProx optimization, and robust aggregation against malicious updates.

##  Key Features

###  Federated Hospital Network
- Four simulated hospitals with non-IID data distributions.
- Local training without transferring raw images between hospitals.
- Global model aggregation using Federated Averaging (FedAvg).
- Comparison with local and centralized training.

###  Differential Privacy
- Differentially Private Stochastic Gradient Descent (DP-SGD).
- Gradient clipping and Gaussian noise.
- Privacy experiments using Opacus.
- Exploration of the privacy–accuracy trade-off.

###  Federated Optimization
- FedAvg for standard federated aggregation.
- FedProx to reduce excessive divergence during local training.
- Multi-round training and accuracy comparisons.

###  Model Poisoning Defense
- Simulation of a compromised hospital.
- Malicious model-update manipulation.
- Comparison of ordinary averaging with coordinate-wise median aggregation.

###  Interactive Dashboard
Built with Streamlit and Plotly, the dashboard provides:

1. **Hospital Network:** Hospital datasets and class distributions.
2. **Attack Lab:** Model poisoning and defense comparisons.
3. **Privacy Lab:** Differential privacy experiments and privacy metrics.
4. **Training Optimization:** FedAvg versus FedProx.
5. **Audit Report:** Exportable experimental summaries.

The dashboard displays saved experimental results rather than automatically retraining models whenever a control changes.

##  Dataset

**Dataset:** PathMNIST from the MedMNIST collection.

PathMNIST contains histopathology images grouped into nine tissue categories.

The experimental training setup uses 8,000 images distributed across four simulated hospitals.

| Hospital | Training Images |
|---|---:|
| Hospital 1 | 1,719 |
| Hospital 2 | 1,564 |
| Hospital 3 | 1,696 |
| Hospital 4 | 3,021 |
| **Total** | **8,000** |

The hospitals have different class distributions, simulating non-independent and identically distributed (non-IID) medical datasets.

##  System Architecture

```text
                  GLOBAL MODEL
                       |
              Federated Server
                       |
         +-------------+-------------+
         |             |             |
     Hospital 1    Hospital 2    Hospital 3
         |             |             |
     Local CNN     Local CNN     Local CNN
         |             |             |
         +-------------+-------------+
                       |
                   Hospital 4
                   Local CNN

       Model updates are aggregated
       Raw images remain local
```

**Training workflow:**

1. Initialize the global CNN.
2. Distribute the model to four hospitals.
3. Train the model independently at each hospital.
4. Collect the locally trained model parameters.
5. Aggregate the updates using FedAvg or another selected method.
6. Evaluate the updated global model.
7. Repeat for multiple communication rounds.

##  Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | CNN training and evaluation |
| MedMNIST | Medical imaging dataset |
| NumPy | Numerical operations |
| Opacus | Differential privacy experiments |
| Streamlit | Interactive dashboard |
| Plotly | Data visualization |
| Git and GitHub | Version control |

##  Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Replace the placeholder URL and folder name with your actual GitHub repository information.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install torch torchvision medmnist opacus numpy pandas streamlit plotly scikit-learn
```

### 4. Launch the Dashboard

If the Streamlit app is inside the `dashboard` folder:

```bash
python -m streamlit run dashboard/app.py
```

Open:

http://localhost:8501

The dashboard expects previously generated experiment files in the `results` directory. Some sections may be empty until the corresponding experiments have been run.

##  Experimental Results

The project explored local training, centralized training, federated learning, FedProx, differential privacy, and poisoning defenses.

An initial experiment reported:

| Training Approach | Accuracy |
|---|---:|
| Hospital 1 (Local) | 43.87% |
| Hospital 2 (Local) | 54.90% |
| Hospital 3 (Local) | 42.81% |
| Hospital 4 (Local) | 50.26% |
| Federated Learning (3 rounds) | 54.82% |
| Centralized Learning | 71.92% |

Additional federated experiments explored more communication rounds, producing improved results under different training configurations.

These results are from recorded development experiments, not a final controlled benchmark across identical settings.

##  Privacy and Security

**Differential Privacy:** Opacus is used to investigate how gradient clipping and Gaussian noise affect model accuracy and privacy guarantees.

**Model Poisoning:** A malicious hospital is simulated by manipulating its model update. The project compares standard aggregation against a coordinate-wise median defense.

Federated learning reduces the need to centralize raw images, but does not by itself guarantee that model updates reveal no private information.

##  Limitations

- Hospitals are simulated rather than real medical institutions.
- The project uses public medical imaging data.
- Differential privacy may significantly reduce classification accuracy.
- Robust aggregation cannot defend against every poisoning strategy.
- The prototype does not implement cryptographic secure aggregation.
- The system has not undergone clinical validation or real hospital deployment.

##  Future Improvements

- Expand testing to more hospitals and larger datasets.
- Investigate stronger privacy-preserving aggregation.
- Evaluate more advanced poisoning attacks.
- Improve accuracy under differential privacy.
- Add live experiment execution and monitoring.
- Test additional federated optimization algorithms.

##  Project Information

**Project:** FedShield AI  
**Category:** Artificial Intelligence / Machine Learning / Healthcare  
**Development:** Solo Hackathon Project  
**Dataset:** MedMNIST — PathMNIST

---

**FedShield AI — Collaborative medical AI without centralizing raw medical images.**