from ipanema import Sample
from ipanema.samples import DataSet
import numpy as np

x = np.array([1,2,3,4,5,6])
y = 2*np.array([1,2,3,4,5,6])



s1 = Sample.from_numpy({'x':x, 'y':y},name='s1')
s2 = Sample.from_numpy({'x':x, 'y':y},name='s2')




d1 = DataSet()
d1.add_category(s1)
d1.add_category(s2)


d1.category('s1')

d1.split('s1',cut='x>3',childnames=['biased','unbiased'])


d1.categories

s1.df.query('~(x>3)')



d1.categories
d1.find_categories('d.*')


list(d1._categories.keys())


for k in d1:
  print(k)
