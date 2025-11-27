# Audio Source Separation with Time-Frequency Sequence Attention Res-U-Net (DCASE 2025)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

This repository contains an implementation that replicates the architecture described in **"TFSWA-ResUNet: music source separation with time–frequency sequence and shifted window attention-based ResUNet"**. 

Instead of music source separation, this implementation adapts the model for **Sound Event Separation** using a subset of the DCASE 2025 Task 4 dataset. The entire training, validation, and testing pipeline is contained within a single Jupyter notebook.

---

## 🎯 Features

### Architecture
- **Res-U-Net** with integrated Time-Frequency Sequence Attention (TF-SA) and Shifted Window Attention
- **Task**: Separating overlapping sound events in domestic environments
- **Input**: Magnitude spectrograms of mixed audio (32kHz sampling rate)
- **Output**: Estimated spectrograms of specific sound classes

---

## 📁 Project Structure

```
Audio-Separation-ResUNet-TF-Attention/
├── TF_SA_ResUNet.ipynb   # Main notebook containing model, training, and inference
└── README.md             # Project documentation
```

---

## 📊 Dataset

This project uses a custom subset of the **DCASE 2025 Task 4 dataset**, reduced to facilitate efficient training while maintaining task complexity.

### Dataset Statistics
- **Total Samples**: 10,000
- **Configuration**: 3 overlapping events per mixture
- **Classes**: 5 target sound classes
- **Sampling Rate**: 32kHz

### Access the Dataset
- 🤗 [Hugging Face](https://huggingface.co/datasets/Kiuyha/dcase-5class-3source-mixtures-32k)
- 📦 [Kaggle](https://www.kaggle.com/datasets/kiuyha/dcase-5class-3source-mixtures-32k)

---

## 🚀 Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/kiuyha/Audio-Separation-ResUNet-TF-Attention.git
cd Audio-Separation-ResUNet-TF-Attention
```

### 2. Open the Notebook
This project is designed to run in **Google Colab** or a local **Jupyter** environment. All necessary dependencies are installed directly within the notebook cells.

- Open `TF_SA_ResUNet.ipynb`
- Ensure you have a **GPU runtime** enabled for training

### 3. Dependencies
The code relies on standard deep learning and audio libraries:
- Python 3.8+
- PyTorch
- Librosa
- NumPy
- Matplotlib
- Soundfile

All dependencies are automatically installed when running the notebook cells.

---

## 🤖 Model Weights

Pre-trained model weights are hosted on Hugging Face:

🤗 **[Download Model Weights](https://huggingface.co/kiuyha/TF-SA-ResUNet-Model)**

### How to Load Weights
1. Download the `.pth` file from the link above
2. Place it in the root directory of the project (or upload it to your Colab session)
3. Run the inference cell in the notebook to load the state dictionary

---

## 📈 Evaluation

The model is evaluated using the DCASE metric:
CA-SDRi (Class-Aware Sound Signal-to-Distortion Ratio Improvement)

### Results

| Model Variant | CA-SDRi (dB) |
|---------------|--------------|
| ResUNet (Baseline) | 3.15857     |
| ResUNet + SpecAugment | 2.95301     |
| TF-SA-ResUNet  | 5.25322     |
| TF-SA-ResUNet + SpecAugment | 4.66175     |

---

## 📝 Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@article{kong2024tfswa,
  title={TFSWA-ResUNet: music source separation with time–frequency sequence and shifted window attention-based ResUNet},
  author={Kong, Q. and Cao, Y. and Liu, H. and Doi, K. and Iqbal, T.},
  journal={Complex \& Intelligent Systems},
  volume={10},
  pages={1--17},
  year={2024},
  publisher={Springer}
}
```

**Paper Link**: [TFSWA-ResUNet on Springer](https://link.springer.com/article/10.1186/s13634-025-01249-0)


## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.


## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the repository maintainer.


## 🙏 Acknowledgments

- DCASE 2025 Task 4 organizers for providing the dataset framework
- Original authors of the TFSWA-ResUNet architecture
- The open-source audio processing community
