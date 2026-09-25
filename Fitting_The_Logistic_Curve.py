# using a plain logistic regression in scipy...
# importing pandas as pd...
import pandas as pd

# importing LogisticRegression from sklearn.linear_model...
from sklearn.linear_model import LogisticRegression

# https://bit.ly/33ebs2R...
df = pd.read_csv('https://bit.ly/33ebs2R', delimiter=',')

# extract input variables
# all rows, all columns except the last column
X = df.values[:, :-1]

# extract output column
# all rows, last column
Y = df.values[:, -1]

# perform logistic regression...
# turn off regularization/penalty...
model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)

# fit the model
model.fit(X, Y)

# print beta1 (coefficient)
print("Beta1 =", model.coef_.flatten()[0])

# print beta0 (intercept)
print("Beta0 =", model.intercept_.flatten()[0])

# calculating the joint likelihood of observing all the points
# for a given logistic regression...
# importing math...
import math

# importing pandas...
import pandas as pd

# load the data...
patient_data = pd.read_csv(
    'https://bit.ly/33ebs2R',
    delimiter=","
).itertuples()

# value of coefficient b0...
b0 = -3.17576395

# value of coefficient b1...
b1 = 0.69267212

# creating function for logistic probability...
def logistic_function(x):
    p = 1.0 / (1.0 + math.exp(-(b0 + b1 * x)))
    return p

# calculate the joint likelihood...
joint_likelihood = 1.0

# apply for-loop...
for p in patient_data:

# probability of y = 1...
    probability = logistic_function(p.x)

    if p.y == 1:
        joint_likelihood *= probability

    elif p.y == 0:
        joint_likelihood *= (1.0 - probability)

# print joint likelihood...
print("Joint likelihood =", joint_likelihood) 

# using gradient descent on logistic regression...
# importing * from sympy...
from sympy import * 

# importing pandas as pd...
import pandas as pd

# load the data...
# creat the dataset...
# load the data...
# https://tinyurl.com/y2cocoo7...
data = {
    'x': [1.0, 1.5, 2.1, 2.4, 2.5, 3.1, 4.2, 4.4, 4.6,
          4.9, 5.2, 5.6, 6.1, 6.4, 6.6, 7.0, 7.6, 7.8,
          8.4, 8.8, 9.2],

    'y': [0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1,
          0, 1, 1, 1, 1, 1]
}

# convert the data into points...
points = [
    (data['x'][i], data['y'][i])
    for i in range(len(data['x']))
]

# create symbols...
b1, b0, i, n = symbols('b1 b0 i n')

# last define for symbols...
x, y = symbols('x y', cls=Function)

# Joint likelihood...
joint_likelihood = Sum(
    log(
        (1.0 / (1.0 + exp(-(b0 + b1 * x(i))))) ** y(i)
        *
        (1.0 - (1.0 / (1.0 + exp(-(b0 + b1 * x(i))))))
        ** (1 - y(i))
    ),
    (i, 0, n)
)

# partial derivative for b1...
d_b1 = diff(joint_likelihood, b1) \
    .subs(n, len(points) - 1).doit() \
    .replace(x, lambda i: points[i][0]) \
    .replace(y, lambda i: points[i][1])

# partial derivative for b0...
d_b0 = diff(joint_likelihood, b0) \
    .subs(n, len(points) - 1).doit() \
    .replace(x, lambda i: points[i][0]) \
    .replace(y, lambda i: points[i][1])

# compile using lambdify for faster computation...
d_b1 = lambdify([b1, b0], d_b1)
d_b0 = lambdify([b1, b0], d_b0)

# perform gradient descent...
b1 = 0.01
b0 = 0.01
L = 0.01

# applies the for-loop...
for j in range(10_000):
    b1 += d_b1(b1, b0) * L
    b0 += d_b0(b1, b0) * L

# prints b1 and b2...
print(b1, b0)