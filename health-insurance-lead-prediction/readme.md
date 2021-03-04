# JOB-A-THON
Health Insurance Lead Prediction

| Project Name | Type  | Public Kernel |Metric | Date |
| ------ | ------ | ------ | ------ | ------ | 
| [JOB-A-THON](https://datahack.analyticsvidhya.com/contest/job-a-thon/) | Classification | NA | AUC | Feb 2021 |


Our client (FinMan Company) wishes to cross-sell health insurance to the existing customers who may or may not hold insurance policies with the company. A policy is recommended to a person when they land on their website, and if the person chooses to fill up a form to apply it is considered as a Positive outcome (Classified as lead). All other conditions are considered Zero outcomes.


From the column keys in problem statement we know the following information about each of the features:-

| Variable             | Definition         | 
| ------------------   | ------------------ | 
|ID Unique             | Identifier for a row
|City_Code             | Code for the City of the customers
|Region_Code           | Code for the Region of the customers
|Accomodation_Type     | Customer Owns or Rents the house
|Reco_Insurance_Type |	Joint or Individual type for the recommended insurance
|Upper_Age             |	Maximum age of the customer
|Lower_Age             |	Minimum age of the customer
|Is_Spouse             |	If the customers are married to each other (in |case of joint insurance)
|Health_Indicator      | 	Encoded values for health of the customer
|Holding_Policy_Duration |	Duration (in years) of holding |policy (a policy that customer has already subscribed to with |the company)
|Holding_Policy_Type    |	Type of holding policy
|Reco_Policy_Cat        |	Encoded value for recommended health |insurance
|Reco_Policy_Premium Annual |	Premium (INR) for the recommended |health insurance
|Response (Target) |	0 : Customer did not show interest in the recommended policy, 1 : Customer showed interest in the recommended policy

We can get a naive idea about the type of variables form the definition itself and looking at the data makes it clearer.

