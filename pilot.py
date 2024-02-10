import sympy as sp

# Define symbolic variables
w, a, b, c, A, p, t, k1, k2 = sp.symbols('w a b c A p t k1 k2')

# Define the function y(w)
y_w = -(a * A * t**2 * sp.sin(p + t * w)) / (a**2 * t**4 - 2 * a * c * t**2 + b**2 * t**2+ c**2) + \
      (A * c * sp.sin(p + t * w)) / (a**2 * t**4 - 2 * a * c * t**2 + b**2 * t**2 + c**2) - \
      (A * b * t * sp.cos(p + t * w)) / (a**2 * t**4 - 2 * a * c * t**2 + b**2 * t**2 + c**2) + \
      k1 * sp.exp(0.5 * w * (-sp.sqrt(b**2 - 4 * a * c)/a - b/a)) + \
      k2 * sp.exp(0.5 * w * (sp.sqrt(b**2 - 4 * a * c)/a - b/a))

# Evaluate y(w) with given values of variables
values = {a: 1, b: 1, c: 1, A: 1, w: 2*p}
result = y_w.subs(values)
print("y(w) =", result)
import numpy as np
import matplotlib.pyplot as plt

# Define the function
def y(t, k1, k2, p):
    numerator_1 = k1 * np.exp(p * (-1 - np.sqrt(3) * 1j))
    numerator_2 = k2 * np.exp(p * (-1 + np.sqrt(3) * 1j))
    denominator = t**4 - t**2 + 1
    term_1 = numerator_1
    term_2 = numerator_2
    term_3 = t**2 * np.sin(2 * p * t + p) / denominator
    term_4 = t * np.cos(2 * p * t + p) / denominator
    term_5 = np.sin(2 * p * t + p) / denominator
    return term_1 + term_2 - term_3 - term_4 + term_5

# Generate values for t
t_values = np.linspace(-10, 10, 1000)

# Set values for k1, k2, and p
k1 = 1
k2 = 2
p = np.pi  # using pi in place of p

# Calculate y values
y_values = y(t_values, k1, k2, p)

# Plot the function
plt.figure(figsize=(10, 6))
plt.plot(t_values, y_values, label='y(t)', color='blue')
plt.title('Plot of y(t)')
plt.xlabel('t')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.show()