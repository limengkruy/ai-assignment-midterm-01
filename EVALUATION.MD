# Model Evaluation Report  
**Date**: October 25, 2023  
**Team Members**:  
- Limeng KRUY  
- Oudom CHHOUN  

---

## Overview of Metrics 📊  
We evaluated multiple machine learning models using the following key metrics:  

- **Confusion Matrix**: TP, FP, FN, TN 🔢  
- **Classification Report**: Precision, Recall, F1-Score 🎯  
- **ROC-AUC Score**: Class separation ability 📈  
- **Accuracy, MSE, R² Score**: Overall performance 🏆  

---

## Top 3 Models 🌟  

### 1. Logistic Regression 🥇  
- **Accuracy**: 77.14%  
- **ROC-AUC**: 85.04%  
- **R² Score**: 8.56%  

### 2. ANN (10 Epochs) 🧠  
- **Accuracy**: 77.54%  
- **ROC-AUC**: 85.36%  
- **R² Score**: 10.16%  

### 3. SVM / Kernel SVM ⚙️  
- **Accuracy**: 76.34%  
- **ROC-AUC**: 84.21%  
- **R² Score**: 5.35%  

---

## Model-by-Model Performance 📋  

| Model               | Accuracy (%) | ROC-AUC (%) | R² Score (%) | Icon       |  
|---------------------|--------------|-------------|--------------|------------|  
| Logistic Regression | 77.14        | 85.04       | 8.56         | 🥇        |  
| ANN (10 Epochs)     | 77.54        | 85.36       | 10.16        | 🧠        |  
| SVM / Kernel SVM    | 76.34        | 84.21       | 5.35         | ⚙️        |  
| Decision Trees      | 75.27        | 83.37       | 1.07         | 🌳        |  
| Random Forest       | 75.27        | 83.37       | 1.07         | 🌲        |  
| XGBoost             | 72.59        | 80.77       | -9.63        | ⚡         |  
| K-NN                | 70.86        | 78.94       | -16.58       | 🔍        |  
| Naive Bayes         | 73.26        | 82.63       | -6.95        | 📜        |  

---

## Key Insights 🌟  

- **Best Models** 🌟:  
  - Logistic Regression and ANN (10 Epochs) are top performers.  

- **Worst Models** ❌:  
  - K-NN has the lowest accuracy and poor R² score.  

- **Imbalanced Performance**:  
  - Naive Bayes shows significant imbalance in precision and recall.  
  - Icon: ⚖️  

---

## Recommendations ✅  

- **Preferred Models** ✅:  
  - Use Logistic Regression or ANN (10 Epochs) for balanced results. 

- **Avoid** ❌:  
  - Avoid K-NN due to poor performance.  

- **Hyperparameter Tuning** ⚙️:  
  - Optimize ensemble methods like Random Forest and XGBoost. 

---

## Conclusion 🏆  

The **best performing model** based on accuracy, ROC-AUC, and R² score is **Logistic Regression**.  
For neural networks, **ANN (10 Epochs)** also performs exceptionally well.  

$$\boxed{\text{Logistic Regression and ANN (10 Epochs) are the top choices.}}$$  

---

## Q&A ❓  

Placeholder for audience questions.  

---

**Prepared by**:  
- Limeng KRUY  
- Oudom CHHOUN  
**Date**: October 25, 2023  