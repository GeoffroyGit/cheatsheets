# best practice:

# only one class per .py file

# Packages and modules have short, all-lowercase names: pandas, lewagon.
# Classes names use UpperCamelCase: DataFrame, Student.
# Variables and functions use lower_snake_case: name, first_name, from_birth_year()

# examples:

# create class "Student" in module "lewagon":
'''
mkdir lewagon-project
cd lewagon-project
mkdir csv
mkdir notebooks
mkdir lewagon
touch lewagon/student.py
touch lewagon/__init__.py
'''

class Student:
    # class attribute
    school = 'lewagon'

    # initializer of instance attributes
    def __init__(self, name, age): # Note the `self` parameter
        self.name = name.capitalize()
        self.age = age

    # instance method
    def says(self, something): # x.f(y) equivalent to Class.f(x,y)
        print(f'{self.name} says {something}')

    # Class method
    @classmethod
    def from_birth_year(cls, name, birth_year): # Note the `cls` parameter
        return cls(name, date.today().year - birth_year)

# in another .py file:

from lewagon.student import Student

class DataStudent(Student):
    cursus = 'datascience'
    def __init__(self, name, age, batch):
        super().__init__(name, age)
        self.batch = batch

# add module to PYTHONPATH
# open your zshrc file
'''
code ~/.zshrc
'''
# add this at the bottom of your zshrc file
'''
export PYTHONPATH='/Users/.../lewagon-project'
'''
# you can check your pythonpath specifically
import os; os.environ['PYTHONPATH']
# double check your root_dir is your path list for package imports
import sys; sys.path
