# -*- coding: utf-8 -*-
import math


class OnlineRegression():

  def __init__(self):
    self.n=0
    self.x=0.0
    self.y=0.0
    self.xx=0.0
    self.yy=0.0
    self.xy=0.0
  def __add__(self, other):
    if not isinstance(other, OnlineRegression): return
    combined = OnlineRegression()
    combined.n = self.n + other.n
    combined.x = self.x + other.x
    combined.y = self.y + other.y
    combined.xx = self.xx + other.xx
    combined.yy = self.yy + other.yy
    combined.xy = self.xy + other.xy
    return combined


  """ 追加 """
  def push(self, x, y):
    self.n += 1
    self.x += x
    self.y += y
    self.xx += x * x
    self.yy += y * y
    self.xy += x * y


  """標準誤差"""
  def stderr(self):
    return math.sqrt(self.residuals()) / (self.n-2.0)


  """切片"""
  def intercept(self):
    return  (self.y - self.slope() * self.x) / self.n if self.n > 0 else self.mean_y()


  """傾き"""
  def slope(self):
    devsqx=self.dev_sq_x()
    return self.dev_prod_xy() / devsqx if devsqx > 0 else 0.0

  """平均Ｘ"""
  def mean_x(self):
    return self.x / self.n if self.n > 0 else 0.0
  
  """平均Ｙ"""
  def mean_y(self):
    return self.y / self.n if self.n > 0 else 0.0

  """偏差平方和 Ｘ"""
  def dev_sq_x(self):
    return (self.xx * self.n - self.x * self.x) / self.n if self.n > 0 else 0.0

  """偏差平方和 Ｙ"""
  def dev_sq_y(self):
    return (self.yy * self.n - self.y * self.y) / self.n if self.n > 0 else 0.0

  """偏差積和 ＸＹ"""
  def dev_prod_xy(self):
    return (self.xy * self.n - self.x * self.y) / self.n if self.n > 0 else 0.0

  """残差平方和"""
  def residuals(self):
    devsqx=self.dev_sq_x()
    return self.dev_sq_y() - (pow(self.dev_prod_xy(),2) / devsqx) if devsqx > 0 else 0.0

