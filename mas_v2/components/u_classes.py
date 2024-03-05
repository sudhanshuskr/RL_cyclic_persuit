from components.imports import *
from components.rk4 import rk4_step
from components.system import *


class Person:
  def __init__(self, name, age):
    self.name = name
    self.age = age
    self.target = None

  def __str__(self):
    return f"{self.name}({self.age})"
  
  def setTarget(self,ag):
    self.target = ag

  def myfunc(self):
    print("Hello my name is " + self.name)


class Agent:
    def __init__(self, x, y, theta = 0, name = "untitled"):
        self.x = x
        self.y = y
        self.name = str(name)
        self.theta = theta
        self.target = None 
    def setTarget(self,ag):
      self.target = ag
    def __str__(self):
      return f"Name : {self.name} |:| Coordinates : ({self.x},{self.y}) |:| Heading : {self.theta} radians"
    # def nextState(self,dynamics,controller,t):
    #   ds = rk4_step(dynamics,self,t,controller)
    #   self.x = self.x + ds[0]
    #   self.y = self.y + ds[1]
    #   self.theta = (self.theta + ds[2])
      
      

