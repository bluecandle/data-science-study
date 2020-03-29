# Poisson
The poisson loss function is used for regression when modeling count data. Use for data follows the poisson distribution. Ex: churn of customers next week.

Minimizing the Poisson loss is equivalent of maximizing the likelihood of the data under the assumption that the target comes from a Poisson distribution, conditioned on the input.

When to use Poisson loss function
Use the Poisson loss when you believe that the target value comes from a Poisson distribution and want to model the rate parameter conditioned on some input. Examples of this are the number of customers that will enter a store on a given day, the number of emails that will arrive within the next hour, or how many customers that will churn next week.
