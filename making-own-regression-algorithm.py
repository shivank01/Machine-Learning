from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")

xs = np.array([1,2,3,4,5], dtype=np.float64)
ys = np.array([5,4,6,5,6], dtype=np.float64)

def m_and_c(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
        ((mean(xs)*mean(xs)) - mean(xs*xs)))
    c = mean(ys) - m*mean(xs)
    return ( m , c )

def squared_error(ys_orig,ys_line):
    return sum((ys_line - ys_orig) * (ys_line - ys_orig))

def coefficient_of_determination(ys_orig,ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr/squared_error_y_mean)

m, c = m_and_c(xs,ys)
regression_line = [ m*x + c for x in xs ]

r_squared = coefficient_of_determination(ys,regression_line)
print(r_squared)

predict_x = 8
predict_y = m*predict_x + c

plt.scatter(xs, ys ,label='data')
plt.scatter(predict_x, predict_y , color='g', label='predicted data')
plt.plot(xs, regression_line, label='regression line')
plt.legend(loc=4)
plt.show()