# In this lab, you will practice using Python operators to perform operations between two variables and write conditional statements.
# 
# As a data analyst, you'll use conditional statements in your approach to many different tasks using Python. Using Boolean values, conditional statements will determine how a code block is executed based on how a condition is met.
# 
# While continuing work for the movie theater, you're now examining marketing campaigns. Specifically, you'll be analyzing user behavior to determine when a customer's activity has prompted a marketing email.
# 


# ## Task 1: Define a comparator function
# 
# You are a data professional for a movie theater, and your task is to use customers' purchasing data to determine whether or not to send them a marketing email.
# 
# *   Define a function called `send_email` that accepts the following arguments:
#     *  `num_visits` - the number of times a customer has visited the theater
#     *  `visits_email` - the minimum number of visits the customer must have made for them to receive a marketing email
# *   The function must print either: `Send email.` or `Not enough visits.`
# 
# *Example:*
# 
# ```
#  [IN] send_email(num_visits=3, visits_email=5)
# [OUT] 'Not enough visits.'
# 
#  [IN] send_email(num_visits=5, visits_email=5)
# [OUT] 'Send email.'
# ```
# 
# **Note that there is more than one way to solve this problem.**
# 

# In[1]:


def send_email(num_visits, visits_email):
    if num_visits >= visits_email:
        print('Send email.')
    else:
        print('Not enough visits.')


# ### Test your function
# Test your function against the following cases by running the cell below.

# In[2]:


send_email(num_visits=3, visits_email=5)    # Should print 'Not enough visits.'
send_email(num_visits=5, visits_email=5)    # Should print 'Send email.'
send_email(num_visits=15, visits_email=10)  # Should print 'Send email.'


# ## Task 2: Add logical branching to your function
# 
# The theater is offering a promotion where customers who have visited the theater more than a designated number of times will also receive a coupon with their email. Update the function that you created above to include additional logical branching.
# 
# *   Include an additional argument `visits_coupon` that represents the minimum number of visits the customer must have made for them to receive a coupon with their email.
# 
# *   The function must print one of three possible messages:
#     1. `Send email with coupon.`
#     2. `Send email only.`
#     3. `Not enough visits.`
# 
# *Example:*
# 
# ```
#  [IN] send_email(num_visits=3, visits_email=5, visits_coupon=8)
# [OUT] 'Not enough visits.'
# 
#  [IN] send_email(num_visits=5, visits_email=5, visits_coupon=8)
# [OUT] `Send email only.`
# 
#  [IN] send_email(num_visits=8, visits_email=5, visits_coupon=8)
# [OUT] `Send email with coupon.`
# ```
# 
# **Note that there is more than one way to solve this problem.**

# In[3]:


def send_email(num_visits, visits_email, visits_coupon):
    if num_visits >= visits_coupon:
        print('Send email with coupon.')
    elif num_visits < visits_coupon and num_visits >= visits_email:
        print('Send email only.')
    else:
        print('Not enough visits.')


# ### Test your function
# Test your function against the following cases by running the cell below.

# In[4]:


send_email(num_visits=3, visits_email=5, visits_coupon=8)   # Should print 'Not enough visits.'
send_email(num_visits=5, visits_email=5, visits_coupon=8)   # Should print 'Send email only.'
send_email(num_visits=6, visits_email=5, visits_coupon=8)   # Should print 'Send email only.'
send_email(num_visits=8, visits_email=5, visits_coupon=8)   # Should print 'Send email with coupon.'
send_email(num_visits=10, visits_email=5, visits_coupon=8)  # Should print 'Send email with coupon.'
