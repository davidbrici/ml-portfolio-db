# Predicting Patient Readmissions Using Machine Learning

## Project Overview
Hospital readmissions pose a significant burden on healthcare systems, leading to increased costs and resource strain. This project aims to predict **whether a patient will be readmitted within 30 days** using machine learning models trained on demographic, clinical, and hospital visit data. The goal is to develop a model that helps hospitals identify high-risk patients and optimize care strategies.

## Objective
- Develop a **classification model** to predict patient readmissions.
- Engineer meaningful features from hospital visit records.
- Compare baseline models with advanced machine learning techniques.
- Optimize performance through hyperparameter tuning.
- Deploy an end-to-end **Jupyter Notebook-based ML pipeline**.

## Dataset
This project utilizes patient data with the following key features:
- **Demographics:** Age, gender, etc.
- **Clinical Data:** Number of lab procedures, medications, diagnoses.
- **Hospital Visit Data:** Outpatient, inpatient, emergency visits, and total visits.
- **Target Variable:** Binary classification (Readmitted: Yes/No).

## Workflow
### **1. Data Preprocessing**
- Handling **missing values** and replacing placeholders.
- **Feature Engineering:** Encoding categorical features, transforming diagnosis information, and deriving insights from visit counts.
- **Feature Selection:** Removing redundant features to avoid multicollinearity.

### **2. Model Development**
- Splitting the dataset into **training and testing sets**.
- Implementing different machine learning models:
  - **Baseline Models:** Logistic Regression, Decision Tree.
  - **Advanced Models:** Random Forest, XGBoost, and CatBoost.
- Hyperparameter tuning using **Grid Search & Random Search**.
- Evaluating models with **accuracy, precision, recall, and AUC-ROC**.

### **3. Model Evaluation & Optimization**
- Comparing multiple models and selecting the best performer.
- **Feature importance analysis** to understand key drivers of readmission.
- Examining class imbalance and model bias.

## **Key Findings**
### **Baseline Model: Logistic Regression**
- Provided an **accuracy of ~61%**.
- Struggled with **recall for readmitted patients**, indicating bias towards predicting non-readmissions.
- **Top predictive features:** Diabetes medication, inpatient visits, and total visits.

### **Advanced Model: CatBoost Classifier**
- Improved recall for readmitted patients while maintaining accuracy at **~61%**.
- Handled categorical variables and class imbalance effectively **without requiring a GPU**.
- Hyperparameter tuning identified the best model settings (**learning rate: 0.05, depth: 6, iterations: 508**).
- **AUC-ROC Score:** 0.652, showing moderate predictive power.

## **Conclusion**
This project successfully built a machine learning model to predict **hospital readmissions** using structured patient data. **The CatBoost model outperformed logistic regression**, particularly in identifying patients at risk of readmission. Feature importance analysis highlighted **lab procedures, medications, and hospital visit history** as key predictors. Despite these improvements, **the overall accuracy suggests further refinements are needed** for real-world deployment.

## **Future Steps**
🔹 **Model Optimization:**
- Implement **early stopping** to reduce unnecessary training.
- Experiment with **threshold tuning and ensemble learning**.

🔹 **Feature Engineering:**
- Explore **interaction terms** and **nonlinear transformations**.
- Investigate additional **medical history-based features**.

🔹 **Alternative Models:**
- Compare performance with **XGBoost, LightGBM, and ensemble approaches**.
- Consider deep learning models for capturing complex relationships.

🔹 **Deployment:**
- Package the model using **Flask/FastAPI** for real-time predictions.
- Explore integration with **hospital management systems**.

## **Technologies Used**
- **Programming Language:** Python
- **Libraries:** pandas, numpy, sklearn, XGBoost, TensorFlow/Keras, matplotlib, seaborn
- **Environment:** Jupyter Notebook
