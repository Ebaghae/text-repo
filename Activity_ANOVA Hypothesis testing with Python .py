#!/usr/bin/env python
# coding: utf-8

# # Activity: Hypothesis testing with Python

# ## **Introduction**
# 

# As you've been learning, analysis of variance (commonly called ANOVA) is a group of statistical techniques that test the difference of means among three or more groups. It's a powerful tool for determining whether population means are different across groups and for answering a wide range of business questions.
# 
# In this activity, you are a data professional working with historical marketing promotion data. You will use the data to run a one-way ANOVA and a post hoc ANOVA test. Then, you will communicate your results to stakeholders. These experiences will help you make more confident recommendations in a professional setting. 
# 
# In your dataset, each row corresponds to an independent marketing promotion, where your business uses TV, social media, radio, and influencer promotions to increase sales. You have previously provided insights about how different promotion types affect sales; now stakeholders want to know if sales are significantly different among various TV and influencer promotion types.
# 
# To address this request, a one-way ANOVA test will enable you to determine if there is a statistically significant difference in sales among groups. This includes:
# * Using plots and descriptive statistics to select a categorical independent variable
# * Creating and fitting a linear regression model with the selected categorical independent variable
# * Checking model assumptions
# * Performing and interpreting a one-way ANOVA test
# * Comparing pairs of groups using an ANOVA post hoc test
# * Interpreting model outputs and communicating the results to nontechnical stakeholders

# ## **Step 1: Imports** 
# 

# Import pandas, pyplot from matplotlib, seaborn, api from statsmodels, ols from statsmodels.formula.api, and pairwise_tukeyhsd from statsmodels.stats.multicomp.

# In[1]:


# Import libraries and packages.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# `Pandas` was used to load the dataset `marketing_sales_data.csv` as `data`, now display the first five rows. The variables in the dataset have been adjusted to suit the objectives of this lab. As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[3]:


# RUN THIS CELL TO IMPORT YOUR DATA.

### YOUR CODE HERE ### 
data = pd.read_csv('marketing_sales_data.csv')

# Display the first five rows.

### YOUR CODE HERE ### 

data.head()


# The features in the data are:
# * TV promotion budget (in Low, Medium, and High categories)
# * Social media promotion budget (in millions of dollars)
# * Radio promotion budget (in millions of dollars)
# * Sales (in millions of dollars)
# * Influencer size (in Mega, Macro, Nano, and Micro categories)

# **Question:** Why is it useful to perform exploratory data analysis before constructing a linear regression model?

# [Write your response here. Double-click (or enter) to edit.]

# ## **Step 2: Data exploration** 
# 

# First, use a boxplot to determine how `Sales` vary based on the `TV` promotion budget category.

# In[5]:


# Create a boxplot with TV and Sales.

sns.boxplot(data['TV'], data['Sales'])


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# There is a function in the `seaborn` library that creates a boxplot showing the distribution of a variable across multiple groups.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the `boxplot()` function from `seaborn`.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Use `TV` as the `x` argument, `Sales` as the `y` argument, and `data` as the `data` argument.
# 
# </details>

# **Question:** Is there variation in `Sales` based off the `TV` promotion budget?

# - The boxplot above shows that sales varies based on the TV promotion budget. The mean of sales increases as the TV promotion budget increases.

# Now, use a boxplot to determine how `Sales` vary based on the `Influencer` size category.

# In[6]:


# Create a boxplot with Influencer and Sales.

sns.boxplot(data['Influencer'], data['Sales'])


# **Question:** Is there variation in `Sales` based off the `Influencer` size?

# - Based on influencer size, there is a variation in avearge sales based off the mega influencer and a slight variation in avearge sales based off nano influencers. However it may not be significant.

# ### Remove missing data
# 
# You may recall from prior labs that this dataset contains rows with missing values. To correct this, drop these rows. Then, confirm the data contains no missing values.

# In[7]:


# Drop rows that contain missing data and update the DataFrame.

data = data.dropna(axis=0)

# Confirm the data contains no missing values.

data.isna().sum()


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# There is a `pandas` function that removes missing values.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# The `dropna()` function removes missing values from an object (e.g., DataFrame).
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Verify the data is updated properly after the rows containing missing data are dropped.
# 
# </details>

# ## **Step 3: Model building** 
# 

# Fit a linear regression model that predicts `Sales` using one of the independent categorical variables in `data`. Refer to your previous code for defining and fitting a linear regression model.

# In[8]:


# Define the OLS formula.

ols_formula = 'Sales ~ C(TV)'


# Create an OLS model.

OLS = ols(formula = ols_formula, data = data)


# Fit the model.

model = OLS.fit()


# Save the results summary.

model_results = model.summary()


# Display the model results.

model_results 


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to code you've written to fit linear regression models.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the `ols()` function from `statsmodels.formula.api`, which creates a model from a formula and DataFrame, to create an OLS model.
# 
# </details>
# 

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Use `C()` around the variable name in the ols formula to indicate a variable is categorical.
#     
# Be sure the variable string names exactly match the column names in `data`.
# 
# </details>

# **Question:** Which categorical variable did you choose for the model? Why?

# - TV was cused for the model because the variation in TV budget seemed to have an effect on Sales.
# 
# - Influencer was not used because it did not seem to have a strong effect on sales.

# ### Check model assumptions

# Now, check the four linear regression assumptions are upheld for your model.

# **Question:** Is the linearity assumption met?

# - Because the model does not have any continuous independent variables, the linearity assumption is not required.

# The independent observation assumption states that each observation in the dataset is independent. As each marketing promotion (row) is independent from one another, the independence assumption is not violated.

# Next, verify that the normality assumption is upheld for the model.

# In[10]:


# Calculate the residuals.

residuals = model.resid


# Create a histogram with the residuals.

plt.hist(residuals)
plt.title('Histogram of Residuals')


# In[11]:


# create a q-q plot with the residuals

sm.qqplot(residuals, line='s')


# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use `model.resid` to get the residuals from a fit model called `model`.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# For the histogram, pass the residuals as the first argument in the `seaborn` `histplot()` function.
#     
# For the QQ-plot, pass the residuals as the first argument in the `statsmodels` `qqplot()` function.
# 
# </details>

# **Question:** Is the normality assumption met?

# -There is reasonable concern that the normality assumption is not met when TV is used as the independent variable predicting Sales. The normal q-q forms an 'S' that deviates off the red diagonal line, which is not desired behavior.

# Now, verify the constant variance (homoscedasticity) assumption is met for this model.

# In[12]:


# Create a scatter plot with the fitted values from the model and the residuals.

fig = sns.scatterplot(x = model.fittedvalues, y = model.resid)

# Set the x axis label
fig.set_xlabel("Fitted Values")

# Set the y axis label
fig.set_ylabel("Residuals")

# Set the title
fig.set_title("Fitted Values v. Residuals")

# Add a line at y = 0 to visualize the variance of residuals above and below 0.

### YOUR CODE HERE ### 

fig.axhline(0)

# Show the plot
plt.show()


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Access the fitted values from the model object fit earlier.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use `model.fittedvalues` to get the fitted values from the fit model called `model`.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# 
# Call the `scatterplot()` function from the `seaborn` library and pass in the fitted values and residuals.
#     
# Add a line to a figure using the `axline()` function.
# 
# </details>

# **Question:** Is the constant variance (homoscedasticity) assumption met?

# -The variance where there are fitted values is similarly distributed, validating that the constant variance assumption is met.

# ## **Step 4: Results and evaluation** 

# First, display the OLS regression results.

# In[13]:


# Display the model results summary.

model_results


# **Question:** What is your interpretation of the model's R-squared?

# - The R-squared value shows that TV variables explains 87.4% of the variations in Sales.

# **Question:** What is your intepretation of the coefficient estimates? Are the coefficients statistically significant?

# The default TV category for the model is High, because there are coefficients for the other two TV categories, Medium and Low. According to the model, Sales with a Medium or Low TV category are lower on average than Sales with a High TV category. For example, the model predicts that a Low TV promotion would be 208.813 (in millions of dollars) lower in Sales on average than a High TV promotion.
# 
# The p-value for all coefficients is  0.000
#  , meaning all coefficients are statistically significant at  𝑝=0.05
#  . The 95% confidence intervals for each coefficient should be reported when presenting results to stakeholders. For instance, there is a  95%
#   chance the interval  [−215.353,−202.274]
#   contains the true parameter of the slope of  𝛽𝑇𝑉𝐿𝑜𝑤
#  , which is the estimated difference in promotion sales when a Low TV promotion is chosen instead of a High TV promotion.

# **Question:** Do you think your model could be improved? Why or why not? How?

# - Given how accurate TV was as a predictor, the model could be improved with a more granular view of the TV promotions, such as additional categories or the actual TV promotion budgets. Further, additional variables, such as the location of the marketing campaign or the time of year, may increase model accuracy.

# ### Perform a one-way ANOVA test
# 
# With the model fit, run a one-way ANOVA test to determine whether there is a statistically significant difference in `Sales` among groups.i,e does sales value differ across the different TV budgets?

# In[14]:


# Create an one-way ANOVA table for the fit model.

sm.stats.anova_lm(model, typ=2)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Review what you've learned about how to perform a one-way ANOVA test.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# There is a function in `statsmodels.api` (i.e. `sm`) that peforms an ANOVA test for a fit linear model.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Use the `anova_lm()` function from `sm.stats`. Specify the type of ANOVA test (for example, one-way or two-way), using the `typ` parameter.
#    
# 
# </details>

# **Question:** What are the null and alternative hypotheses for the ANOVA test?

# - Null Hypothesis: mean of sales is not statistically different among the TV budget promotion groups.
# - Alternative Hypothesis: mean of sales is statistically different among the TV budget promotion groups.

# **Question:** What is your conclusion from the one-way ANOVA test?

# - The F-test statistic is 1971.46 and the p-value is  8.81∗10−256
#   (i.e., very small). Because the p-value is less than 0.05, we reject the null hypothesis that there is no difference in Sales based on the TV promotion budget.

# **Question:** What did the ANOVA test tell you?

# - There is a statistically significant difference in sales based off the Tv promotion budget.

# ### Perform an ANOVA post hoc test
# 
# If you have significant results from the one-way ANOVA test, you can apply ANOVA post hoc tests such as the Tukey’s HSD post hoc test. 
# 
# Run the Tukey’s HSD post hoc test to compare if there is a significant difference between each pair of categories for TV.

# In[15]:


# Perform the Tukey's HSD post hoc test.

tukey_oneway = pairwise_tukeyhsd(endog = data["Sales"], groups = data["TV"])

# Display the results
tukey_oneway.summary()


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Review what you've learned about how to perform a Tukey's HSD post hoc test.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the `pairwise_tukeyhsd()` function from `statsmodels.stats.multicomp`.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# The `endog` argument in `pairwise_tukeyhsd` indicates which variable is being compared across groups (i.e., `Sales`). The `groups` argument in `pairwise_tukeyhsd` tells the function which variable holds the group you’re interested in reviewing.
# 
# </details>

# **Question:** What is your interpretation of the Tukey HSD test?

# - The first row, which compares the High and Low TV groups, indicates that we reject the null hypothesis that there is no significant difference between the Sales of these two groups.
# 
# - We reject the null hypotheses for the two other pairwise comparisons that compare High to Medium and Low to Medium.

# **Question:** What did the post hoc tell you?**

# A post hoc test was conducted to determine which TV groups are different and how many are different from each other. This provides more detail than the one-way ANOVA results, which can at most determine that at least one group is different. Further, using the Tukey HSD controls for the increasing probability of incorrectly rejecting a null hypothesis from peforming multiple tests.
# 
# The results were that Sales is not the same between any pair of TV groups.

# ## **Considerations**
# 
# **What are some key takeaways that you learned during this lab?**
# 
# [Write your response here. Double-click (or enter) to edit.]
# 
# 
# **What summary would you provide to stakeholders? Consider the statistical significance of key relationships and differences in distribution.**
# 
# High TV promotion budgets result in significantly more sales than both medium and low TV promotion budgets. Medium TV promotion budgets result in significantly more sales than low TV promotion budgets.
# 
# Specifically, following are estimates for the difference between the mean sales resulting from different pairs of TV promotions, as determined by the Tukey's HSD test:
# 
# Estimated difference between the mean sales resulting from High and Low TV promotions: $208.81 million (with 95% confidence that the exact value for this difference is between 200.99 and 216.64 million dollars).
# Estimated difference between the mean sales resulting from High and Medium TV promotions: $101.51 million (with 95% confidence that the exact value for this difference is between 93.69 and 109.32 million dollars).
# difference between the mean sales resulting from Medium and Low TV promotions: $107.31 million (with 95% confidence that the exact value for this difference is between 99.71 and 114.91 million dollars).
# The linear regression model estimating Sales from TV had an R-squared of $0.871, making it a fairly accurate estimator. The model showed a statistically significant relationship between the TV promotion budget and Sales.
# 
# The results of the one-way ANOVA test indicate that the null hypothesis that there is no difference in Sales based on the TV promotion budget can be rejected. Through the ANOVA post hoc test, a significant difference between all pairs of TV promotions was found.
# 
# The difference in the distribution of sales across TV promotions was determined significant by both a one-way ANOVA test and a Tukey’s HSD test.
# 

# #### **Reference**
# [Saragih, H.S. *Dummy Marketing and Sales Data*](https://www.kaggle.com/datasets/harrimansaragih/dummy-advertising-and-sales-data)

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged.
