<b>Prediction of House Prices Using Machine Learning Techniques</b>

In this project, three machine learning models, namely linear regression (LR), random forest (RF), and extreme gradient boosting (XGBoost), were used to predict house prices. Additionally, principle component analysis (PCA) was also used to reduce the dimensionality of the dataset. Besides, cross-validation was also perform to prevent overfitting of the models.

The dataset used in this work was obtained from Kaggle, which is a dataset presenting the house prices of King County in the U.S. State of Washington. The dataset can be found from the link: https://www.kaggle.com/datasets/shivachandel/kc-house-data

<b>Table 1.</b> Experimental results of the machine learning models on the original dataset and PCA-reduced dataset.
![image](https://user-images.githubusercontent.com/129178911/228336334-8ee2da3b-b41e-4b31-99c8-eb50f3ee3d51.png)

<b>Table 2.</b> Experimental results of the machine learning models on the original dataset and PCA-reduced dataset after cross validation.
![image](https://user-images.githubusercontent.com/129178911/228336455-7876f412-5c29-4487-95c5-4b2b91a4668d.png)

In overall, the results show that XGBoost demonstrated the best performance in predicting house price after cross-validation against the original dataset. The results obtained is consistent with most of the literatures reviewed. This means that XGBoost is effective in predicting house prices and this algorithm can be recommended to property developers, real-estate agents, and house seller.
