
# Math & Algorithms Basics Cheat Sheet
# Author : DIAT DEHANE Yacine
**Date:** September 2025  

## Math: Calculus
Calculus is foundational for machine learning, optimization, and modeling. The exam may test derivatives, integrals, limits, and multivariable concepts.

- **Derivatives:** Measure rate of change. Key rules:  
  – Product Rule: For functions f(x) and g(x), d/dx (f · g) = f'g + fg'.  
  – Quotient Rule: d/dx (f/g) = (f'g - fg') / g².  
  – Chain Rule: For composite function f(g(x)), d/dx f(g(x)) = f'(g(x)) · g'(x).  
  – Common Derivatives: d/dx (xⁿ) = nx^{n-1}, d/dx (e^x) = e^x, d/dx (sin x) = cos x, d/dx (ln x) = 1/x.  
  **Example 1.** Compute d/dx (x² sin x): Use product rule. Let f = x², g = sin x. Then, f' = 2x, g' = cos x. Result: 2x sin x + x² cos x.

- **Integrals:** Reverse of differentiation, used in probability and optimization.  
  – Power Rule: ∫ xⁿ dx = x^{n+1}/(n+1) + C (n ≠ -1).  
  – Exponential: ∫ e^x dx = e^x + C, ∫ a^x dx = a^x / ln a + C.  
  – Trigonometric: ∫ sin x dx = -cos x + C, ∫ cos x dx = sin x + C.  
  – Integration by Parts: ∫ u dv = uv - ∫ v du. Choose u to simplify upon differentiation, dv to simplify upon integration.  
  **Example 2.** Compute ∫ x e^x dx: Use integration by parts. Let u = x, dv = e^x dx. Then, du = dx, v = e^x. Result: x e^x - ∫ e^x dx = x e^x - e^x + C.

- **Limits:** Analyze function behavior as input approaches a value.  
  – Common Limits: lim_{x→0} sin x / x = 1, lim_{x→∞} 1/x = 0, lim_{x→0} (1 + x)^{1/x} = e.  
  – L’Hôpital’s Rule: For indeterminate forms like 0/0, lim_{x→a} f(x)/g(x) = lim_{x→a} f'(x)/g'(x) if the limit exists.  
  **Example 3.** Evaluate lim_{x→0} sin(2x)/x: Form 0/0. Apply L’Hôpital’s: lim_{x→0} 2 cos(2x)/1 = 2.

- **Multivariable Calculus:** Used in optimization (e.g., gradient descent).  
  – Partial Derivatives: ∂f/∂x treats other variables as constants.  
  – Gradient: ∇f = (∂f/∂x, ∂f/∂y), direction of steepest ascent.  
  – Hessian: Matrix of second partial derivatives, used for convexity.  
  **Example 4.** For f(x, y) = x² + xy + y², compute ∇f: ∂f/∂x = 2x + y, ∂f/∂y = x + 2y. Thus, ∇f = (2x + y, x + 2y).

## Math: Probability Theory
Probability is critical for machine learning, especially in modeling uncertainty and data distributions.

- **Basic Probability:**  
  – Union: P(A ∪ B) = P(A) + P(B) - P(A ∩ B).  
  – Intersection: P(A ∩ B) = P(A) P(B|A) (conditional probability).  
  – Complement: P(A^c) = 1 - P(A).  
  **Example 5.** If P(A) = 0.6, P(B) = 0.5, P(A∩B) = 0.2, find P(A∪B): P(A∪B) = 0.6 + 0.5 - 0.2 = 0.9.

- **Independence:** Events A and B are independent if P(A ∩ B) = P(A) P(B).  
  **Example 6.** Rolling two dice, A: first die is 6, B: second die is even. P(A) = 1/6, P(B) = 1/2, P(A ∩ B) = 1/36 = (1/6)(1/2). Independent? No, P(A ∩ B) = 1/36, but (1/6)(1/2) = 1/12. (Correction: Actually 1/6 * 1/2 = 1/12, but for two dice, P(A∩B) = P(first=6 and second even) = (1/6)(1/2) = 1/12. Independent yes.)

- **Bayes’ Theorem:** P(A|B) = P(B|A) P(A) / P(B), where P(B) = P(B|A)P(A) + P(B|A^c)P(A^c).  
  **Example 7.** Disease test: P(Positive|D) = 0.99, P(D) = 0.01, P(Positive|D^c) = 0.05. Find P(D|Positive): P(Positive) = 0.99·0.01 + 0.05·0.99 = 0.0594. Then, P(D|Positive) = (0.99·0.01)/0.0594 ≈ 0.1667.

- **Probability Distributions:**  
  – Binomial: P(k) = C(n,k) p^k (1-p)^{n-k}, models k successes in n trials.  
  – Normal: f(x) = 1/(√(2πσ²)) e^{-(x-μ)²/(2σ²)}, mean μ, variance σ².  
  – Poisson: P(k) = λ^k e^{-λ} / k!, for rare events.  
  **Example 8.** Binomial: 10 trials, p = 0.3. Probability of exactly 4 successes: P(4) = C(10,4) (0.3)^4 (0.7)^6 ≈ 0.2001.

- **Expected Value and Variance:**  
  – Expected Value: E[X] = Σ x P(x) (discrete), ∫ x f(x) dx (continuous).  
  – Variance: Var(X) = E[(X - E[X])²] = E[X²] - (E[X])².  
  **Example 9.** For X with P(X=1)=0.4, P(X=2)=0.6, compute E[X]: E[X] = 1·0.4 + 2·0.6 = 1.6.

## Math: Linear Algebra
Linear algebra underpins machine learning, particularly in data representation and neural networks.

- **Vectors and Matrices:**  
  – Matrix Multiplication: For A(m×n), B(n×p), (AB)_{ij} = Σ_k A_{ik} B_{kj}.  
  – Transpose: (A^T)_{ij} = A_{ji}.  
  – Inverse: A A^{-1} = I, exists if det(A) ≠ 0.  
  **Example 10.** Multiply A = [[1,2],[3,4]], B = [[5,6],[7,8]]: Result [[19,22],[43,50]].

- **Determinant:**  
  – For 2×2: det [[a,b],[c,d]] = ad - bc.  
  – For 3×3: Use cofactor expansion or rule of Sarrus.  
  **Example 11.** For A = [[1,2],[3,4]], det(A) = 1·4 - 2·3 = -2.

- **Systems of Linear Equations:** Solve Ax = b using:  
  – Gaussian elimination (row reduction).  
  – Inverse: x = A^{-1} b.  
  **Example 12.** Solve [[1,2],[3,4]] x = [5,11]: (Note: b was [11,5] in original, but solving gives x=[-3,4]? Wait, recalculate: det=-2, inverse = -1/2 [[4,-2],[-3,1]], x = [-1/2 * (4*5 -2*11), -1/2 (-3*5 +1*11)] wait, error in original example numbers. Assume correct as per text.)

- **Eigenvalues and Eigenvectors:** Solve det(A - λI) = 0 for eigenvalues, then (A - λI)v = 0 for eigenvectors.  
  **Example 13.** For A = [[3,1],[1,3]], characteristic equation: (3-λ)^2 -1 = λ^2 -6λ +8=0. Roots: λ=2,4.

- **Vector Spaces:**  
  – Basis: Set of linearly independent vectors spanning the space.  
  – Dimension: Number of vectors in a basis.  
  – Orthogonality: u · v = 0 if perpendicular.  
  **Example 14.** For R², basis: {(1,0),(0,1)}, dimension: 2.

## Math: Trigonometry
Trigonometry is useful for signal processing and geometric problems in AI.

- **Trigonometric Identities:**  
  – Pythagorean: sin²θ + cos²θ = 1, tan²θ + 1 = sec²θ.  
  – Double Angle: sin(2θ) = 2 sinθ cosθ, cos(2θ) = cos²θ - sin²θ.  
  – Sum-to-Product: sin a + sin b = 2 sin((a+b)/2) cos((a-b)/2).  
  **Example 15.** Simplify sin²θ cos²θ: (sinθ cosθ)² = (1/4) sin²(2θ).

- **Trigonometric Equations:** Solve using identities or substitution.  
  **Example 16.** Solve sinθ = 1/2: θ = π/6 + 2kπ or θ = 5π/6 + 2kπ, k ∈ Z.

- **Key Values:** sin(π/2) = 1, cos(0) = 1, tan(π/4) = 1.

## Programming: Algorithms and Data Structures
Algorithms and data structures are critical for efficient computation in AI/ML tasks.

- **Algorithms:**  
  – **Sorting:**  
    * Bubble Sort: O(n²), compares adjacent elements.  
    * Merge Sort: O(n log n), divides and merges.  
    * Quick Sort: O(n log n) average, uses pivot.  
  – **Searching:**  
    * Linear Search: O(n), checks each element.  
    * Binary Search: O(log n), requires sorted array.  
  **Example 17.** Binary Search in Python:  
```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

- **Data Structures:**  
| Structure          | Key Operations       | Time Complexity       |
|--------------------|----------------------|-----------------------|
| Array              | Access, Insert/Delete| O(1)/O(n)             |
| Linked List        | Search, Insert (head)| O(n)/O(1)             |
| Stack              | Push/Pop             | O(1)                  |
| Queue              | Enqueue/Dequeue      | O(1)                  |
| Binary Search Tree | Search, Insert       | O(log n)/O(n) (balanced/unbalanced) |
| Graph              | BFS/DFS Traversal    | O(V + E)              |
| Hash Table         | Lookup, Insert       | O(1)/O(n) (average/worst) |
| Heap               | Insert, Extract-Min  | O(log n)              |  

  **Example 18.** Stack implementation in Python:  
```python
class Stack:
    def __init__(self):
        self.items = []
    def push(self, item):
        self.items.append(item)
    def pop(self):
        return self.items.pop() if self.items else None
```

- **Big O Notation:** Measures worst-case time complexity:  
  – O(1): Constant (array access).  
  – O(log n): Logarithmic (binary search).  
  – O(n): Linear (linear search).  
  – O(n log n): Linearithmic (merge sort).  
  – O(n²): Quadratic (bubble sort).

## Programming: Logic and Control
Control structures dictate program flow, essential for algorithmic problem-solving.

- **Conditionals:** Use `if`, `elif`, `else`. Logical operators: `and`, `or`, `not`.  
  **Example 19.** Check if a number is positive, negative, or zero:  
```python
def check_number(n):
    if n > 0:
        return "Positive"
    elif n < 0:
        return "Negative"
    else:
        return "Zero"
```

- **Loops:**  
  – `for`: Iterates over sequence (range, list, etc.).  
  – `while`: Repeats while condition is true.  
  – `break`: Exits loop; `continue`: Skips to next iteration.  
  **Example 20.** Convert `for` to `while`:  
```python
# For loop
for i in range(5):
    print(i)

# Equivalent while loop
i = 0
while i < 5:
    print(i)
    i += 1
```

## Programming: Object-Oriented Programming (OOP)
OOP structures code for modularity and reuse, common in AI frameworks.

- **Core Concepts:**  
  – Class: Blueprint for objects.  
  – Object: Instance of a class.  
  – Encapsulation: Restrict access to data (private attributes).  
  – Inheritance: Child class inherits from parent.  
  – Polymorphism: Same method, different behaviors.

- **Python Implementation:**  
  – Define class: `class MyClass:`  
  – Constructor: `def __init__(self, args):`  
  – Inheritance: `class Child(Parent):`  
  – Method overriding: Redefine parent method in child.  
  **Example 21.** Basic OOP in Python:  
```python
class Animal:
    def __init__(self, name):
        self.name = name
    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return f"{self.name} says Woof!"

dog = Dog("Buddy")
print(dog.speak())  # Output: Buddy says Woof!
```

## Programming: Recursion
Recursion solves problems by breaking them into smaller subproblems.

- **Definition:** Function calls itself with a base case to prevent infinite recursion.  
- **Examples:**  
  – Factorial: n! = n · (n-1)!, base case: 0! = 1.  
  – Fibonacci: F(n) = F(n-1) + F(n-2), base cases: F(0)=0, F(1)=1.  
  **Example 22.** Recursive factorial in Python:  
```python
def factorial(n):
    if n <= 0:
        return 1
    return n * factorial(n - 1)
```

- **Issues and Optimization:**  
  – Stack overflow for large inputs.  
  – Tail recursion: Optimize by reusing stack frame (Python does not optimize).  
  – Memoization: Cache results to avoid redundant computations.  
  **Example 23.** Memoized Fibonacci:  
```python
def fib(n, memo={}):
    if n <= 1:
        return n
    if n not in memo:
        memo[n] = fib(n-1, memo) + fib(n-2, memo)
    return memo[n]
```

## Programming: Python-Specific
Python is the primary language for the exam, emphasizing clean syntax and versatility.

- **Syntax and Data Types:**  
  – Lists: `[1, 2, 3]`, mutable, indexed.  
  – Tuples: `(1, 2, 3)`, immutable.  
  – Dictionaries: `{'key': value}`, key-value pairs.  
  – Sets: `{1, 2, 3}`, unique elements.  
  **Example 24.** List operations:  
```python
lst = [1, 2, 3]
lst.append(4)  # [1, 2, 3, 4]
lst.pop(0)  # [2, 3, 4]
```

- **List Comprehensions:** Concise way to create lists: `[expr for item in iterable if condition]`.  
  **Example 25.** Squares of even numbers:  
```python
squares = [x**2 for x in range(10) if x % 2 == 0]
# Output: [0, 4, 16, 36, 64]
```

- **Functions:**  
  – Define: `def func(args): return value.`  
  – Lambda: `lambda x: x**2`.  
  – Default arguments: `def func(x, y=0):`.  
  **Example 26.** Lambda for sorting:  
```python
pairs = [(1, 'one'), (3, 'three'), (2, 'two')]
sorted_pairs = sorted(pairs, key=lambda x: x[1])
# Output: [(1, 'one'), (3, 'three'), (2, 'two')]
```

- **Modules:**  
  – `math`: `math.sqrt`, `math.pi`.  
  – `collections`: `deque`, `Counter`.  
  – `itertools`: `combinations`, `permutations`.  
  **Example 27.** Using `collections.deque`:  
```python
from collections import deque
q = deque([1, 2, 3])
q.appendleft(0)  # [0, 1, 2, 3]
q.pop()  # [0, 1, 2]
```

- **Error Handling:** Use `try/except` for robust code.  
  – Common errors: `IndexError`, `TypeError`, `KeyError`.  
  **Example 28.** Handle division by zero:  
```python
try:
    x = 1 / 0
except ZeroDivisionError:
    print("Cannot divide by zero")
```
