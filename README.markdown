# Patient Survival Prediction

## Overview
This project aims to predict patient survival in a hospital setting using machine learning techniques. The model is trained on a dataset containing various patient features, such as age, BMI, medical conditions, and clinical measurements, to predict whether a patient will survive or not (binary classification). The project uses a Random Forest Classifier as the primary model and includes data preprocessing, model evaluation, and deployment preparation for a Streamlit application.

## Dataset
The dataset used in this project contains 91,713 patient records with 85 features, including:
- **Demographic Information**: `age`, `bmi`, `height`, `gender`, `ethnicity`
- **Clinical Information**: `icu_admit_source`, `apache_3j_bodysystem`, `apache_2_bodysystem`
- **Medical Conditions**: `diabetes_mellitus`, `hepatic_failure`, `immunosuppression`, `leukemia`, `lymphoma`, `solid_tumor_with_metastasis`
- **Target Variable**: `hospital_death` (0 for survival, 1 for death)

The dataset is loaded into a Pandas DataFrame for analysis and preprocessing.

## Dependencies
The project requires the following Python libraries:
- `pandas`
- `matplotlib`
- `seaborn`
- `numpy`
- `scipy`
- `scikit-learn`
- `imblearn`
- `xgboost`
- `pickle`

Install the dependencies using:
```bash
pip install pandas matplotlib seaborn numpy scipy scikit-learn imblearn xgboost
```

## Methodology
1. **Data Preprocessing**:
   - Missing values are handled (if any).
   - Numerical features are scaled using `MinMaxScaler`.
   - Categorical variables are likely encoded (not shown in the provided code snippet).
   - Class imbalance is addressed using `SMOTE` (Synthetic Minority Oversampling Technique).

2. **Model Training**:
   - Multiple classifiers are imported, but the Random Forest Classifier (`RandomForestClassifier`) is used for predictions.
   - Hyperparameter tuning is performed using `GridSearchCV` (assumed based on imports).

3. **Model Evaluation**:
   - The model’s performance is evaluated using a confusion matrix, classification report, and accuracy score.
   - A ROC curve is plotted, showing an AUC of 0.98 for the Random Forest Classifier, indicating excellent performance.

4. **Prediction**:
   - The model can predict survival for custom input data after scaling.
   - Example prediction: For a given set of features, the model outputs whether the patient will survive (`0`) or not (`1`).

5. **Model Export**:
   - The trained Random Forest model (`rf`) and the `MinMaxScaler` (`scaler`) are saved using `pickle` for use in a Streamlit application.

## Files
- `Project1.ipynb`: The main Jupyter Notebook containing the data analysis, preprocessing, model training, and evaluation.
- `project1.sav`: The saved Random Forest model.
- `minmaxscaler1.sav`: The saved MinMaxScaler object for feature scaling.

## Usage
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook**:
   - Open `Project1.ipynb` in Jupyter Notebook or JupyterLab.
   - Ensure the dataset is available in the same directory or update the file path in the notebook.
   - Run all cells to preprocess the data, train the model, and evaluate performance.

4. **Make Predictions**:
   - Use the provided code snippet to input custom patient data and predict survival:
     ```python
     y_cust = rf.predict(scaler.transform([[3.0, 6.0, 4.0, 0.0, 39.3, 0.0, 37.0, 37.0, 119.0, 46.0, 46.0, 74.0, 73.0, 73.0, 37.2, 0.1, 0.05]]))
     result = y_cust.item()
     print("Patient will survive" if result == 0 else "Patient will not survive")
     ```

5. **Deploy with Streamlit**:
   - The saved model (`project1.sav`) and scaler (`minmaxscaler1.sav`) can be used in a Streamlit app for interactive predictions.
   - Create a Streamlit script (e.g., `app.py`) to load the model and scaler, then deploy using:
     ```bash
     streamlit run app.py
     ```

## Results
- The Random Forest Classifier achieves an AUC of 0.98, indicating strong predictive performance.
- The model can accurately predict patient survival based on the provided features.
- A sample prediction demonstrates the model’s ability to classify a patient as likely to survive.

## Future Improvements
- Include additional feature engineering to enhance model performance.
- Explore other classifiers (e.g., XGBoost, Gradient Boosting) for comparison.
- Add cross-validation to ensure model robustness.
- Enhance the Streamlit app with a user-friendly interface for inputting patient data.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
For questions or contributions, please open an issue or contact the repository owner.