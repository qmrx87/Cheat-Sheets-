# Python Essentials for Data Scientists
 **Date:** September 2025  

## 1 Installing Python
### 1.1 Using Anaconda
The recommended way to install Python for data science is to use the Anaconda distribution. It provides Python itself and a package manager called conda to manage environments and libraries.

### 1.2 Creating a Virtual Environment
To avoid conflicts between different projects, we create a dedicated virtual environment. For example, to create an environment named dsfs with Python 3.13:  
`conda create -n dsfs python=3.13`

### 1.3 Activating the Environment
After the environment is created, activate it with:  
`conda activate dsfs`  
You can confirm the Python version with:  
`python --version`

### 1.4 Installing IPython
Inside the environment, install IPython, an enhanced interactive Python shell:  
`python -m pip install ipython`  
Run it with:  
`ipython`

### 1.5 Installing Data Science Libraries
Finally, install the essential libraries for this course:  
`python -m pip install matplotlib pandas jupyter`  
Or, equivalently:  
`conda install matplotlib pandas jupyter`

## 2 Modules
Python provides many features through modules, which must be imported before use.  
- Import a whole module: `import re` then use `re.compile(...)`.  
- Use an alias to shorten long names: `import matplotlib.pyplot as plt`.  
- Import specific items: `from collections import Counter`.  
- Bad practice: avoid `from module import *` as it may overwrite existing variables.

## 3 Functions
A function is a block of code that takes zero or more inputs and returns an output. In Python, functions are usually defined with `def`:  
```python
def double(x):
    """Returns twice the input value"""
    return x * 2
```  
Python functions are first-class citizens, meaning they can be assigned to variables or passed as arguments to other functions:  
```python
def apply_to_one(f):
    return f(1)

my_double = double
x = apply_to_one(my_double)  # equals 2
```  
For short, one-off functions, Python provides lambdas (anonymous functions):  
```python
y = apply_to_one(lambda x: x + 4)  # equals 5
```  
While you can assign lambdas to variables, it is usually better practice to use `def` for readability:  
```python
another_double = lambda x: 2 * x  # not recommended

def another_double(x):  # preferred
    return 2 * x
```  
Functions can also have default arguments, which are used if no value is provided:  
```python
def my_print(message="default message"):
    print(message)

my_print("hello")  # prints "hello"
my_print()  # prints "default message"
```  
Finally, arguments can be specified by name for clarity:  
```python
def full_name(first="Someone", last="Unknown"):
    return first + " " + last

full_name("Malak", "Guergour")  # "Malak Guergour"
full_name("Malak")  # "Malak Unknown"
full_name(last="Guergour")  # "Someone Guergour"
```

## 4 Strings
Python offers flexible ways to create and manipulate strings:  
- Delimited by single or double quotes: `'data'` or `"data"`.  
- Special characters use backslashes: `"\t"` is a tab (`len("\t") == 1`).  
- Raw strings preserve backslashes: `r"\t"` means two characters, `'\'` and `'t'`.  
- Multiline strings use triple quotes: `"""line1\nline2\nline3"""`.  
- f-strings (Python 3.6+) allow easy interpolation: `f"{first} {last}"` is clearer than concatenation or `.format()`.

## 5 Exceptions
When something goes wrong, Python raises an exception. If not handled, the program will crash.  
- Use `try/except` to catch errors safely:  
```python
try:
    print(0 / 0)
except ZeroDivisionError:
    print("cannot divide by zero")
```  
- Exceptions are not considered bad practice in Python. They are often used to write cleaner and safer code.

## 6 Lists
A list is the most fundamental data structure in Python. It is an ordered collection that can contain different types of elements.

### 6.1 Creating Lists
```python
integer_list = [1, 2, 3]
heterogeneous_list = ["string", 0.1, True]
list_of_lists = [integer_list, heterogeneous_list, []]
```

### 6.2 Basic Operations
- Length: `len([1, 2, 3]) == 3`  
- Sum: `sum([1, 2, 3]) == 6`  
- Indexing: `x[0]` (first), `x[-1]` (last)

### 6.3 Slicing
- `x[:3]` → first 3 elements  
- `x[3:]` → from 4th to end  
- `x[1:5]` → elements 2 to 5  
- `x[-3:]` → last 3 elements  
- `x[::3]` → every 3rd element  
- `x[5:2:-1]` → reverse slice

### 6.4 Membership Test
`1 in [1, 2, 3]`  # True  
`0 in [1, 2, 3]`  # False

### 6.5 Modifying Lists
- Extend: `x.extend([4, 5, 6])`  
- Concatenate: `y = x + [4, 5, 6]`  
- Append: `x.append(0)`

### 6.6 Unpacking
If the number of elements is known, you can unpack a list:  
`x, y = [1, 2]`  # x = 1, y = 2  
Use underscore `_` to ignore values:  
`_, y = [1, 2]`  # y = 2

### 6.7 Summary
Lists are flexible:  
- Ordered, mutable collections  
- Support indexing, slicing, and iteration  
- Can grow dynamically with `append` and `extend`  
- Useful for storing and manipulating sequences of data

## 7 Dictionaries
A dictionary is a data structure that maps keys to values. It allows fast lookups and flexible data representation.

### 7.1 Creating Dictionaries
`empty_dict = {}`  
`grades = {"Joel": 80, "Tim": 95}`

### 7.2 Accessing Values
`grades["Joel"]`  # 80  
`grades.get("Kate", 0)`  # 0 (default if missing)

### 7.3 Modifying Entries
`grades["Tim"] = 99`  # update  
`grades["Kate"] = 100`  # add new entry  
`len(grades)`  # 3

### 7.4 Checking Keys
`"Joel" in grades`  # True  
`"Kate" in grades`  # False

### 7.5 Iterating
```python
tweet = {"user": "joel", "text": "Data Science!"}
tweet.keys()  # dict keys
tweet.values()  # dict values
tweet.items()  # key-value pairs
```

### 7.6 defaultdict
The `defaultdict` from `collections` automatically creates default values for missing keys:  
```python
from collections import defaultdict
word_counts = defaultdict(int)
for word in document:
    word_counts[word] += 1
```

### 7.7 Summary
- Dictionaries map keys to values for fast access.  
- Keys must be immutable (e.g., strings, numbers, tuples).  
- `get()` avoids errors for missing keys.  
- `defaultdict` simplifies counting and grouping tasks.

## 8 Counters
A `Counter` (from the `collections` module) transforms a sequence into a dictionary mapping key → count:  
```python
from collections import Counter
c = Counter([0, 1, 2, 0])  # {0:2, 1:1, 2:1}
```  
Very useful for word counting:  
```python
word_counts = Counter(document)
for word, count in word_counts.most_common(10):
    print(word, count)
```  
**Summary**  
- Simplifies element counting.  
- Provides `most_common()` to get frequent items.

## 9 Sets
A set is a collection of distinct elements (no duplicates). Definition:  
`primes = {2, 3, 5, 7}`  
`s = set()`  
`s.add(1)`  # {1}  
`s.add(2)`  # {1, 2}  
`2 in s`  # True  
`3 in s`  # False

### Why Use Sets?
We'll use sets for two main reasons:  
1. Fast membership tests: Checking if an element is in a set is much faster than in a list.  
```python
stopwords_list = ["a", "an", "at"] + hundreds_of_other_words + ["yet", "you"]
"zip" in stopwords_list  # False, but checks every element

stopwords_set = set(stopwords_list)
"zip" in stopwords_set  # Very fast check
```  
2. Finding distinct items: Sets allow us to remove duplicates easily.  
```python
item_list = [1, 2, 3, 1, 2, 3]
num_items = len(item_list)  # 6
item_set = set(item_list)  # {1, 2, 3}
num_distinct_items = len(item_set)  # 3
distinct_item_list = list(item_set)  # [1, 2, 3]
```  
We'll use sets less frequently than dictionaries and lists.  

**Summary**  
- Unordered collection without duplicates.  
- Very efficient for membership tests.  
- Useful to deduplicate a collection.

## 10 Control Flow
Control flow in Python is based on conditionals and loops:  
- `if / elif / else`: execute code depending on conditions.  
- Ternary operator (`x if condition else y`): one-line conditional.  
- Loops:  
  – `while`: repeats while a condition is true.  
  – `for ... in ...`: iterates over an iterable (e.g. `range`).  
- Keywords:  
  – `continue`: skip to the next iteration.  
  – `break`: exit the loop entirely.

## 11 Truthiness
In Python, certain values are considered false (falsy), while all others are considered true (truthy):  
- Falsy: `False`, `None`, `0`, `0.0`, `""`, `[]`, `{}`, `set()`.  
- Truthy: everything else.  
Useful functions:  
- `all(iterable)`: true if all elements are truthy.  
- `any(iterable)`: true if at least one element is truthy.

## 12 Sorting
Python provides two ways to sort sequences:  
- `.sort()`: sorts a list in place.  
- `sorted()`: returns a new sorted list.  
Options:  
- `reverse=True`: sort in descending order.  
- `key=function`: sort using a function applied to elements (e.g. `abs`, `lambda`).

## 13 List Comprehensions
- Quickly create new lists by filtering and/or transforming elements from an existing list.  
- Syntax: `[expression for item in iterable if condition]`  
- Examples:  
```python
even_numbers = [x for x in range(5) if x % 2 == 0]  # [0, 2, 4]
squares = [x * x for x in range(5)]  # [0, 1, 4, 9, 16]
even_squares = [x * x for x in even_numbers]  # [0, 4, 16]
```  
- Can also generate dictionaries and sets.  
- Multiple `for` loops and `if` conditions are supported.

## 14 Automated Testing and assert
- `assert` checks if a condition is true; raises `AssertionError` if false.  
- Can include an optional message:  
`assert 1 + 1 == 2, "1 + 1 should equal 2"`  
- Useful for testing functions:  
```python
def smallest_item(xs):
    return min(xs)

assert smallest_item([10,20,5,40]) == 5
```  
- Can also validate function inputs.

## 15 Object-Oriented Programming (OOP)
- Classes encapsulate data (attributes) and functions (methods).  
- Create objects (instances) from a class.  
- Example class:  
```python
class CountingClicker:
    def __init__(self, count=0):
        self.count = count

    def click(self, num_times=1):
        self.count += num_times

    def read(self):
        return self.count

    def reset(self):
        self.count = 0
```  
- Use `assert` to test class behavior.  
- Inheritance: create subclasses to extend or override parent class methods.

## 16 Iterables and Generators
A list is convenient because you can access an element by its index. But you don’t always need all the elements in memory: A list of 1 billion numbers takes up a lot of memory. If you only want to go through the elements one by one, there is no need to keep all the elements in memory.

### 16.1 yield
```python
def generate_range(n):
    i = 0
    while i < n:
        yield i  # produces a value
        i += 1

for i in generate_range(5):
    print(i)
# Output: 0, 1, 2, 3, 4
```  
Explanation: `yield` produces a value and suspends the function. The next iteration resumes where it left off.

### 16.2 Infinite sequence
```python
def natural_numbers():
    n = 1
    while True:
        yield n
        n += 1
```  
There must be a mechanism to stop, otherwise the loop is infinite.

### 16.3 Limitations of generators
A generator can only be iterated over once.  
To iterate multiple times:  
- recreate the generator each time, or  
- use a list (if generating the items is expensive).

### 16.4 Generator expressions
`evens_below_20 = (i for i in range(20) if i % 2 == 0)`  
Values are only computed when iterated over (`for` or `next`). Useful for creating data processing pipelines:  
```python
data = natural_numbers()
evens = (x for x in data if x % 2 == 0)
even_squares = (x**2 for x in evens)
even_squares_ending_in_six = (x for x in even_squares if x % 10 == 6)
```

### 16.5 Getting indices with enumerate
When iterating over a list or generator, you often want both the index and the value:  
```python
names = ["Alice", "Bob", "Charlie"]
for i, name in enumerate(names):
    print(f"Name {i}: {name}")
```  
`enumerate` automatically creates pairs (index, value), which is clearer than managing a counter manually.

## 17 Randomness
- Use `random` module to generate reproducible pseudorandom numbers:  
```python
import random
random.seed(10)  # fix the randomness to get the same results
print(random.random())  # consistent output
```  
- Useful functions:  
  – `random.random()`: float in [0,1)  
  – `random.randrange(start, stop)`: random integer  
  – `random.shuffle(list)`: reorder list randomly  
  – `random.choice(list)`: pick one element  
  – `random.sample(list, k)`: pick k unique elements

## 18 Regular Expressions
- Use the `re` module for pattern matching:  
```python
import re
assert re.search("a", "cat")  # True, "a" found anywhere in the string
assert not re.match("a", "cat")  # False, "a" does not match the start of the string
```  
- Common operations:  
  – `match`: checks for a match only at the beginning of the string.  
  – `search`: searches for a match anywhere in the string.  
  – `split`: splits a string by the occurrences of the pattern.  
  – `sub`: replaces occurrences of the pattern with a specified string.  
- Examples:  
`re.split("[ab]", "carbs")`  # ['c', 'r', 's'], splits at 'a' or 'b'  
`re.sub("[0-9]", "-", "R2D2")`  # "R-D-", replaces digits with "-"

## 19 Zip and Argument Unpacking
### 19.1 Creating pairs with zip
```python
list1 = ['a', 'b', 'c']
list2 = [1, 2, 3]
pairs = list(zip(list1, list2))  # [('a', 1), ('b', 2), ('c', 3)]
```

### 19.2 Unpacking tuples
`letters, numbers = zip(*pairs)`  # unpacks the tuples  
`*` operator unpacks the elements of a list or tuple as separate arguments.  
Explanation: - `zip(list1, list2)` combines elements of two lists into tuples. - `zip(*pairs)` reverses the operation, separating the tuples back into individual sequences.

## 20 Arbitrary Arguments: *args and **kwargs
### 20.1 Basic usage
```python
def magic(*args, **kwargs):
    print("args:", args)
    print("kwargs:", kwargs)

magic(1, 2, key="value")
# args: (1, 2)
# kwargs: {'key': 'value'}
```  
`*args` collects all non-keyword (positional) arguments into a tuple. `**kwargs` collects all keyword arguments into a dictionary.

### 20.2 Example with higher-order function
```python
def doubler_correct(f):
    def g(*args, **kwargs):
        return 2 * f(*args, **kwargs)
    return g
```

## 21 Type Annotations
- Improve readability, documentation, and editor support.  
- Examples:  
```python
from typing import List, Dict, Tuple, Callable, Optional

def add(a: int, b: int) -> int:
    return a + b

numbers: List[int] = [1, 2, 3]
best_so_far: Optional[float] = None

def repeater(f: Callable[[str, int], str], s: str) -> str:
    return f(s, 2)
```  
- Use parameterized types for lists, dictionaries, tuples, iterables, and functions.
