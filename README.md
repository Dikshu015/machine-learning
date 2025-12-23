# Machine Learning Projects

This repository contains multiple machine learning projects and exercises covering structured and unstructured data. Some projects and concepts are based on the **Zero to Mastery AI & ML course by Andrei Neagoie**.

## Projects Included

* **Dog Vision Deep Learning**: Image classification of dogs using deep learning in google colab.
* **Bulldozer Price Prediction**: End-to-end regression project predicting bulldozer prices.
* **Heart Disease Classification**: Classification project predicting heart disease based on patient data.
* **Sample Project Exercises**: Exercises covering Python, NumPy, Pandas, Matplotlib, and Scikit-learn.

## Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/Dikshu015/machine-learning.git
cd machine-learning
```

2. **Create environment**

```bash
conda env create -f environment.yml
conda activate ml_course
```

3. **Install additional packages (if required)**

```bash
pip install -r requirements.txt
```

## Folder Structure

```
ML_course/
│
├─ DOG-VISION-DEEP-LEARNING/
│   └─ dog_vision.ipynb
│
├─ bulldozer-price-prediction-project/
│   └─ end-to-end-bulldozer-price-regression.ipynb
│
├─ heart-disease-project/
│   ├─ 90 - heart-disease.csv
│   └─ end-to-end-heart-disease-classification.ipynb
│
├─ sample_project/
│   ├─ exercises/
│   │   ├─ for_manufacturing/
│   │   ├─ matplotlib-exercises.ipynb
│   │   ├─ numpy-exercises.ipynb
│   │   └─ pandas-exercises.ipynb
│   ├─ images/
│   └─ study_material/
│       ├─ teacher-materials/
│       └─ other notebooks and resources
│
├─ environment.yml
└─ README.md
```

## Usage

Open any `.ipynb` file in Jupyter Notebook or VS Code. Run each cell sequentially to follow the project workflow.

* For structured data projects (Bulldozer, Heart Disease): load CSV, preprocess, train models, evaluate results.
* For unstructured data projects (Dog Vision): load images, preprocess, define model, train, and evaluate accuracy.

## Credits

* **Zero to Mastery AI & ML Course** by Andrei Neagoie: Used as a learning reference for exercises and project workflow.
* Open-source Python libraries: NumPy, Pandas, Matplotlib, Scikit-learn, TensorFlow, PyTorch.

## License

This repository is for learning purposes. Use for personal or educational projects only.

---
