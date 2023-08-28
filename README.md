# **PredictingInsurancePremiums_in_R and serving prediction through an API endpoint**

![bandicam 2023-08-27 12-35-04-831 (2)](https://github.com/franciskyalo/PredictingInsurancePremiums_in_R/assets/94622826/1a73354d-74d1-4c3c-b669-562277aad831)

# Overview 

As outlined in a working paper titled "Insurance Premiums: Determinants and Policy Implications" by the International Monetary Fund (IMF), insurance companies encounter multiple challenges when they determine insurance premiums. The process involves evaluating the likelihood of an insurance event for each policyholder, which incorporates a wide array of factors such as age, gender, medical history, and the specific type and expense of coverage. This assessment procedure can be intricate and time-intensive.

A significant risk faced by the insurance sector is adverse selection. This situation arises when an applicant secures insurance at a cost that is lower than their actual risk level, primarily due to essential variables not being factored in during the premium calculation process.

# Business understanding 

This risk arises when applicants secure insurance at a cost that inadequately reflects their actual level of risk. This discrepancy is primarily attributable to crucial variables that are not factored into the premium calculation process. Addressing these challenges is crucial to maintaining a balanced and sustainable insurance business model.

## Main objective

The objective of this project is to develop a predictive model capable of reliably estimating insurance premiums with an accuracy exceeding 80%. Subsequently, a PlumberApi will be established to facilitate predictions within a production setting.


# Data understanding

The dataset consists of 1338 rows and 7 columns. These columns encompass 4 numerical and 3 categorical attributes that will play a pivotal role in the selection of target and predictor variables for this analysis.

Here's a breakdown of the attributes:

- Age: This corresponds to the individual's age in years.
- BMI: This represents the individual's body mass index.
- Children: This indicates the count of dependents covered by the health insurance.
- Smoker: This denotes whether an individual is a smoker or non-smoker (yes for smoker, no for non-smoker).
- Region: This describes the geographical location of the beneficiary within the USA.
- Expenses: This pertains to the medical costs billed to the individual by the health insurance.

# Modelling and evaluation

The model was fit using a multiple linear regression model. An initial model was fitted using all the variables in their untransformed form. The initial model yielded an R-squared of 75% but showed signs of heteroskedasticity. A second model was fitted with the target variable being log-transformed and also an interaction term included between BMI and smoking. The second model had an improved R-squared of 82% and was selected for deployment. The second model also had a 2% Mean Absolute percentage error on new predictions after being evaluated on the test set.


# Deployment 

A Plumber Api endpoint was used to serve the prediction of the model in a production environment such as AWS, Azure or GCP

# Recommendations 

The company might want to implement elevated premiums for smokers, given that the data suggests their expected expenses are higher.

1.Taking the number of children into account while determining premiums could be advantageous for the company, given the positive correlation this variable shows with expenses.

2. Offering discounted premiums to policyholders from the SouthWest and Northwest regions might be a strategic option. These regions display lower expected expenses compared to the SouthEast region.

3. Enhancing the accuracy of the pricing model could be achieved by collecting additional data on factors like medical conditions, previous claims, lifestyle, and occupation.

4. The option of providing discounts to individuals maintaining a healthy BMI could be explored, as this attribute exhibited a negative correlation with expenses.

5. Incorporating the age of a policyholder into premium calculations could be prudent, given the positive correlation observed between age and expenses.








