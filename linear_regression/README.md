## Linear regression practice

### Overview

One & multi variable linear regression. Datasets are taken from the [coursera machine learning course](coursera.org/learn/machine-learning).

#### Datasets

- `housing_prices.in` - `area, number of bedrooms, selling price` 
- `restaurant_profit.in` - `city population, profit made` - Negative means negative profit

#### Single variable linear regression

![img](gradient_descent.jpg) 

```
def gradient_descent(X, y, nr_iterations):

    m = X.shape[0]
    theta = np.zeros(2)
    for _ in range(nr_iterations):
        theta_zero = theta[0] - alpha/m * sum(theta[0] + theta[1] * X - y)
        theta_one = theta[1] - alpha/m * sum((theta[0] + theta[1] * X  - y) * X)

        theta[0] = theta_zero
        theta[1] = theta_one

    return theta
```


##### Result

![img](single_variable_plot.png)
