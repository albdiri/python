#!/usr/bin/env python
# coding: utf-8

# Laboratory work 1 ф.и.о Адь-бдири мухаммад хади
# група рим -111060
# 

# 1.Сначала устанавливаем в программу необходимые библиотеки.

# In[ ]:


import numpy
import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
from matplotlib import pyplot

from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf

import h5py
from scipy.stats import skew,kurtosis

get_ipython().run_line_magic('matplotlib', 'inline')


# 2.Создать ВР являющийся выборкой  случайной  величины  с нормальным распределением 

# In[2]:


X = rand.randn(10000)


# 3.Создать  для  него  ряд  временных  отсчетов,  на  которых  он  будет определен: 
# t = np.linspace(3, 5, num = 10000)

# In[3]:


t = np.linspace(3, 5, num = 10000)


#  4.Построить ВР на заданной временной сетке с помощью функцийplt.
#  plt.plot(t, X)

# In[4]:


plt.figure(figsize=(10, 5))  # size of the drawn area
plt.plot(t, X)
plt.show()


# 6.Найти мат. ожидание данного ВР двумя способами:Во-первых, спомощью функции
# M = np.mean(X)

# In[6]:


def variance(X):     # Number of observations
    n = len(X)       # Mean of the data
    mean = np.sum(X) / n  # Square deviations
    deviations = [(x - mean) ** 2 for x in X]   # Variance
    variance = sum(deviations) / n
    return variance


# 7.Найти дисперсию (variance) данногоВР двумя способами:С помощью функции
# D = np.var(X)

# In[7]:


def calc_Expectation(X):      # variable prb is for probability
                              # of each element which is same for
                              # each element
    n = len(X)
    prb = 1 / n               # calculating expectation overall
    sum = 0
    for i in range(0, n):
        sum += (X[i] * prb)   # returning expectation as sum
    return float(sum)


# 8.Найти асимметрию ВР по формуле (2.9).Найти в Pythonфункцию, которая считает ту же самую характеристику, искать по ключевому слову Skew. Сравнить полученные результаты расчетов.

# In[8]:


def my_fullfunction(X):
    t = np.linspace(3, 5, num=len(X))
    print(X)
M = np.mean(X)
print(M, "using np.mean(X)")   # Function to calculate expectation


# 9.Найти  эксцесс  ВР  по  формуле  (2.10).Найти  в Pythonфункцию, которая считает ту же самую характеристику, искать по ключевому словуKurtosis.Для нее использовать параметр fisher= False.Сравнить полученные результаты расчетов.
# 

# In[9]:


n = len(X)                    # Function for calculating expectation
expect = calc_Expectation(X)  # Display expectation of given array
print(expect, "using loops")  # don't use  loops
M2 = np.sum(X) / n
print(M2, "without loops")


# 10.остроить оценку выборочной автокорреляции ВР несколькимиспособами(до 20 лага)и построить ее на графике:С помощью функции plot_acf(X[0:20])

# In[10]:


D = np.var(X)
print(D, "dvariancein using np.var(X)")
print(variance(X))
from scipy.stats import kurtosis, skew
SX = skew(X)
print('skewness of normal distribution (should be 0): {}'.format(SX))
KX = kurtosis(X)
print('excess kurtosis of normal distribution (should be 0): {}'.format(KX))
plot_acf(X[0:10])
plot_acf(X)
np.correlate(X, X, mode='full')
plt.show()


# 11.Написать полную функцию,  которая  имеет один  входной параметр–это исходныйвременнойряддля анализа.Функция  должна  выполнять  всевышеперечисленныеперечисленные действия (кроме 1 пункта, конечно же) для того ВР, что был передан ей в  качестве  параметра.То  есть  вычислять  мат.  ожидание,  дисперсию, асимметрию, эксцесс и строить АКФ.
# 12.Получить  у  преподавателя mat-файлы,  содержащие  массивы некоторых  ВР, по  вариантам.  Номер  варианта  определяется  по последним двум цифрам студенческого билета.
# 13.Загрузить  из  этих mat-файлов  массив  ВР.  Например,  для  12-го варианта:
# Xmat= h5py.File('12.mat', 'r') 
# Xmat = Xmat.get('z12') 
# Xmat = np.array(Xmat)
# Xmat.ravel()
# Используем десятый файл.

# In[1]:


#%matplotlib inline
import numpy as np
import h5py
def variance(X):
    n = len(X)
    mean = np.sum(X) / n                                                                          
    deviations = [(x - mean) ** 2 for x in X]
    variance = sum(deviations) / n
    return variance
def calc_Expectation(X):
    n = len(X)
    prb = 1 / n
    sum = 0
    for i in range(0, n):
        sum += (X[i] * prb)
    return float(sum)
def f(X):
    t = np.linspace(3, 5, num=len(X))
    plt.figure(figsize=(10, 5))
    plt.plot(t, X)
    plt.show()
    print(t)
    print(X)
    M = np.mean(X)
    print(M, "using np.mean(X)")
    n = len(X)
    expect = calc_Expectation(X)
    print(expect, "using loops")
    M2 = np.sum(X) / n
    print(M2, "without loops")
    D = np.var(X)
    print(D, "dvariancein using np.var(X)")
    print(variance(X))
    from scipy.stats import kurtosis, skew
    SX = skew(X)
    print('skewness of normal distribution (should be 0): {}'.format(SX))
    KX = kurtosis(X)
    print('excess kurtosis of normal distribution (should be 0): {}'.format(KX))
    plot_acf(X[0:10])
    plot_acf(X)
    #np.correlate(X, X, mode='full')
    plt.show()

    return 0
X = np.random.randn(10000)
data = h5py.File('10.mat', 'r')
Xmat = Xmat.get('z10')
Xmat = np.array(Xmat)
Xmat.ravel()
f(Xmat)


# In[ ]:




