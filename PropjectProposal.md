HU14 CISC 593 P1: Project Proposal KA2
Project Title: Predicting Customer Churn Using UCI Machine Learning Repository Data
1. Overview
This project proposes the development of a software system designed to predict customer churn by
leveraging publicly available datasets from the UCI Machine Learning Repository. The system will utilize
machine learning techniques to analyze historical customer data and generate predictive models that
identify customers likely to discontinue a service. This predictive capability aims to empower businesses
to implement proactive retention strategies, reducing customer attrition and improving profitability.
2. Major Features
• Data Ingestion and Preprocessing:
The system will support automated loading of multiple UCI churn-related datasets. Preprocessing
steps will include data cleaning, handling missing values, normalization/scaling of features, and
encoding categorical variables to prepare data suitable for model training.
• Exploratory Data Analysis (EDA) Module:
This feature will provide visualizations and statistical summaries of the datasets, helping users
understand feature distributions, correlations, and identify salient patterns related to churn.
• Machine Learning Model Training:
The software will implement multiple classification algorithms such as Logistic Regression,
Random Forest, and Gradient Boosting. Users can select models, tune hyperparameters, and train
models on the processed datasets.
• Model Evaluation and Selection:
The system will evaluate trained models using metrics like accuracy, precision, recall, F1-score,
and ROC-AUC. Cross-validation capabilities will ensure robustness. The best-performing model
will be selectable for deployment.
• Prediction Interface:
A user interface allowing input of new customer data to generate churn risk predictions using the
selected model. This interface will support batch as well as individual predictions.
• Reporting and Export:
Generate detailed reports summarizing analysis, model performance, and predictions, exportable in
common formats such as PDF or CSV for business use.
3. Unit Testing Plan
Unit tests will focus on the reliability and accuracy of key modules:
• Data Preprocessing Module: Tests will verify correct handling of missing data, encoding, and
normalization for representative dataset samples.
• Feature Engineering Functions: Validate transformations such as encoding categorical variables
and generating derived features.
• Machine Learning Pipeline Components: Confirm that model training functions correctly
instantiate models, fit data, and output expected metrics on controlled inputs.
• Prediction Function: Ensure predictions are logically consistent according to trained model
outputs and handle edge cases like incomplete input data.
Automated testing frameworks will be used to run these unit tests regularly during development to
maintain code quality.
4. Final System Test Plan
The final system test will validate the integrated functionality of the software by executing an end-to-end
workflow:
1. 2. 3. 4. 5. 6. Import a selected UCI churn dataset and verify correct loading and preprocessing.
Perform exploratory data analysis and verify report generation.
Train multiple machine learning models and validate evaluation metric outputs.
Select the best model and run prediction tests on a holdout or new data sample.
Generate and export a summary report including insights and predictions.
Verify system responsiveness and error handling during user interactions.
This comprehensive acceptance testing ensures the software meets functional requirements and performs
robustly in realistic scenarios.
5. Chosen Programming Language and Platform
The software will be implemented primarily in Python due to its rich ecosystem for data science and
machine learning, including libraries such as pandas, scikit-learn, matplotlib, and seaborn. For the user
interface and reporting features, frameworks such as Flask or Dash will be employed to enable lightweight
web-based access. The platform will support deployment on standard desktop environments and cloud-
based servers, ensuring accessibility and scalability.
