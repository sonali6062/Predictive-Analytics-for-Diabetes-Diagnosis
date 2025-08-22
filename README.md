# Predictive-Analytics-for-Diabetes-Diagnosis
This repository contains a project focused on building and evaluating machine learning models to predict the onset of diabetes based on various diagnostic health measurements. The analysis includes data preprocessing, exploratory data analysis, outlier handling, and a comparative study of several classification algorithms.

## Project Goal
The primary objective is to develop a reliable predictive model for diabetes diagnosis. The project emphasizes the importance of model evaluation metrics, such as recall, which are critical in a healthcare context to minimize false negatives (i.e., failing to identify individuals who have diabetes).

## Dataset
The analysis is performed on the `diabetes_data.csv` dataset. It includes the following features:
- `pregnancies`: Number of times pregnant
- `glucose`: Plasma glucose concentration
- `diastolic`: Diastolic blood pressure (mm Hg)
- `triceps`: Triceps skinfold thickness (mm)
- `insulin`: 2-Hour serum insulin (mu U/ml)
- `bmi`: Body mass index
- `dpf`: Diabetes pedigree function
- `age`: Age (years)
- `diabetes`: Target variable (0 for non-diabetic, 1 for diabetic)

## Project Workflow
The project follows a systematic approach to build and evaluate the predictive models:

1.  **Exploratory Data Analysis (EDA):**
    *   The dataset was inspected for data types, missing values, and initial statistics.
    *   Illogical zero values in features like `glucose`, `diastolic`, `triceps`, `insulin`, and `bmi` were identified for imputation.
    *   A correlation heatmap was generated to understand the relationships between features.

2.  **Data Preprocessing:**
    *   **Imputation:** Illogical zero values were replaced with the mean or median of the respective feature.
    *   **Outlier Handling:** Box plots were used to visualize outliers, which were then managed using the Interquartile Range (IQR) method.
    *   **Standardization:** Features were scaled using `StandardScaler` to have a mean of 0 and a standard deviation of 1.

3.  **Handling Imbalanced Data:**
    *   The target variable showed a class imbalance. The **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to the training data to create a balanced class distribution.

4.  **Model Training & Evaluation:**
    *   The dataset was split into training and testing sets.
    *   Four classification models were trained and evaluated:
        *   Logistic Regression
        *   Gaussian Naive Bayes
        *   K-Nearest Neighbors (KNN)
        *   Random Forest Classifier
    *   Performance was measured using `accuracy`, `precision`, `recall`, `f1-score`, and the `confusion matrix`.

## Results
The performance of the trained models on the test set was compared, with a focus on recall for the "Diabetic" class.

| Model | Accuracy | Recall (Diabetic) | Precision (Diabetic) | F1-Score (Diabetic) |
| :--- | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.748 | 0.72 | 0.60 | 0.66 |
| Gaussian Naive Bayes | 0.723 | 0.67 | 0.57 | 0.62 |
| **KNN Classifier** | **0.710** | **0.77** | **0.54** | **0.64** |
| Random Forest | 0.744 | 0.71 | 0.60 | 0.65 |

### Conclusion
While Logistic Regression achieved the highest overall accuracy, the **K-Nearest Neighbors (KNN) Classifier** demonstrated the highest recall score (0.77) for the diabetic class. In a medical diagnostic scenario, maximizing recall is crucial to ensure that as many true positive cases as possible are identified. Therefore, the KNN model is recommended as the most suitable for this specific use case.

## Files in this Repository
*   `Predictiveanalysis_in_diabetes.ipynb`: The Jupyter Notebook containing the full end-to-end analysis, from data loading to model comparison.
*   `diabetes_data.csv`: The raw dataset used for the analysis.
*   `classification_model (1).pkl`: The saved, trained Logistic Regression model serialized using pickle.
*   `LICENSE`: The MIT License for this project.

## How to Run
1.  Clone the repository:
    ```bash
    git clone https://github.com/sonali6062/Predictive-Analytics-for-Diabetes-Diagnosis.git
    cd Predictive-Analytics-for-Diabetes-Diagnosis
    ```
2.  Install the required Python libraries:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn imblearn
    ```
3.  Open and run the `Predictiveanalysis_in_diabetes.ipynb` notebook in a Jupyter environment to see the complete workflow.

4.  To use the pre-trained model for predictions:
    ```python
    import pickle
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    # Load the trained model
    with open('classification_model (1).pkl', 'rb') as file:
        model = pickle.load(file)

    # Load the scaler (Note: The scaler should be saved from the notebook for correct scaling)
    # As an example, we create a new scaler here. For accurate results, use the scaler fitted on the original training data.
    scaler = StandardScaler()
    
    # Example new data (must be scaled)
    # This data should be in the same format as the training features before scaling
    new_data = pd.DataFrame([[6,148,72,35,30.5,33.6,0.627,50]], columns=['pregnancies', 'glucose', 'diastolic', 'triceps', 'insulin', 'bmi', 'dpf', 'age'])
    
    # Fit scaler and transform data (in a real scenario, you'd just transform)
    scaled_data = scaler.fit_transform(new_data)

    # Make a prediction
    prediction = model.predict(scaled_data)
    print(f"Prediction (0: Non-Diabetic, 1: Diabetic): {prediction[0]}")
    ```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
