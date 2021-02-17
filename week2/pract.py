# list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]
# print(list[:-30])
#
# print(list[-9:])
# print(list[-1])
import numpy as np
# # p = np.random.seed(10)
# # p = np.arange(14)
# # babies = range(10)
# # months = np.arange(13)
# # data = [(month, np.dot(month, 24.7) + 96 + np.random.normal(loc=0, scale=20))
# #         for month in months
# #         for baby in babies]
# # print(data)
#
# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([1,2,3,4,5,6,7,8,9,10])
# print(sum(x))
# print(y**2)
# np.random.seed(1)
# for i in range(10):
#     np.random.seed(i)
#     print(np.random.rand())


arr =[1,2,3,4,5,6,7,8,9,10]

arr1 = np.array(arr).reshape(-1,1)
print(arr)
arr2 =np.delete(arr1,2,0)
print(arr1)
print(arr2)
