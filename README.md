# McDonald's Customer Sentiment Analysis
### This project aims to build a Deep Learning model to predict customer sentiment toward McDonald's. Using Natural Language Processing (NLP), this project analyzes customer reviews and various factors such as product categories and restaurant locations. The model helps McDonald's improve its services through objective data-driven insights.

## Dataset Source
```
Dataset is taken from :

McDonald's Store Reviews
By Nidula Elgiriyewithana

```
<a href="https://www.kaggle.com/datasets/nelgiriyewithana/mcdonalds-store-reviews/data">DATASET LINK</a>

## Role in the Project
As a Data Scientist, my responsibilities included:

- Conducting Exploratory Data Analysis (EDA) to understand customer sentiment distribution.
- Preprocessing and transforming text data using Natural Language Processing (NLP) techniques.
- Building and optimizing a Deep Learning model using TensorFlow for sentiment prediction.
- Evaluating model performance and fine-tuning hyperparameters for better accuracy.
- Providing actionable insights for McDonald's management to enhance customer experience. 

## Background
In the fast-food industry, customer satisfaction plays a crucial role in business growth. Understanding customer feedback allows McDonald's to:

- Identify trends in customer satisfaction and service quality.
- Assess restaurant performance across different locations.
- Improve product offerings based on real-time customer insights.
- Make data-driven decisions instead of relying on assumptions.

## Project Objectives
✅ Develop a sentiment analysis model to classify customer reviews. <br>
✅ Identify key factors influencing positive and negative feedback. <br>
✅ Provide McDonald's with data-driven insights to improve customer experience. <br>
✅ Automate the sentiment analysis process for real-time monitoring.

## Justification
- https://altametrics.com/topics/the-advantages-of-customer-data-analytics-in-quick-service-restaurants/
- https://www.wowapps.com/how-restaurant-reviews-impact-peoples-decisions/
- https://www.questionpro.com/blog/mcdonalds-customer-experience/

## Target Users
- McDonald's Managers → To monitor and improve restaurant performance.
- Customer Support Team → To respond effectively to customer concerns.
- Marketing & Branding Team → To refine branding strategies based on sentiment insights.

## Tools & Technologies Used
```
🔹 TensorFlow → Building and training the Deep Learning model.
🔹 Natural Language Processing (NLP) → Text preprocessing and feature engineering.
🔹 Pandas & NumPy → Data manipulation and analysis.
🔹 Matplotlib & Seaborn → Visualizing customer sentiment trends.
🔹 Streamlit → Deploying the model for real-time predictions.
```

## Model Performance Summary
- Test Accuracy: 87%
- Test Loss: 0.32
- Classification Report:
   + Negative Sentiment: Precision (85%), Recall (88%), F1-Score (86%)
   + Neutral Sentiment: Precision (83%), Recall (84%), F1-Score (83%)
   + Positive Sentiment: Precision (90%), Recall (88%), F1-Score (89%)
- Confusion Matrix Analysis:
   + The model struggles slightly in distinguishing Negative & Neutral Sentiments.
   + Positive Sentiment predictions are the most accurate.
- Training & Validation Trends:
   + Accuracy improves steadily over epochs, indicating good learning.
   + Loss decreases consistently, showing minimal overfitting.

## Conclusion
✅ The model performs well with an accuracy of 87%, making it suitable for customer sentiment analysis at McDonald's. <br>
✅ Predictions are stable, but improvements can be made in differentiating Neutral & Negative Sentiments. <br>
✅ This model can be used by McDonald's management to analyze customer satisfaction trends and optimize their services.

## Model Workflow
```
This project follows these steps:

1️⃣ Data Collection & Cleaning → Gather customer reviews and preprocess text data.
2️⃣ Exploratory Data Analysis (EDA) → Analyze sentiment distribution and key patterns.
3️⃣ Feature Engineering → Convert text into numerical features for Deep Learning.
4️⃣ Model Development → Train and optimize the ANN model for sentiment classification.
5️⃣ Model Evaluation → Assess model performance and fine-tune parameters.
6️⃣ Deployment & Inference → Deploy the trained model for real-time sentiment analysis.
```

## Deployment
Model deployed on Hugging face. Deployment file can be found in deployment folder.
Click this link, and click Restart this space
<a href="https://huggingface.co/spaces/rizkystiawanp/Sentiment_Analysis">Huggingface Link</a>