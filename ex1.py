import numpy as np
import amge63basetoorig

amgMatrix = np.loadtxt('e63amgpricesbase.txt', delimiter=",")

X = amgMatrix[:, 0:2]
y = amgMatrix[:, 2]
m = len(amgMatrix)

x0 = np.ones((m,1))
X = np.hstack((x0,X))

def normaleq(X, y):
    theta = np.zeros((X.shape[1], 1))
    theta = np.linalg.pinv(np.transpose(X).dot(X)).dot(np.transpose(X).dot(y))
    return theta

theta = normaleq(X, y)
print(theta)
year = int(input("Year: "))
mileage = int(input("Mileage: "))
priceMatrix = np.matrix([1,year,mileage])
price = priceMatrix.dot(theta)
print(price)
addons = input("Add-on features? (y/n): ")
if (addons == "y"):
    finalPrice = amge63basetoorig.basetoorig(price)
else:
    exit()
print(finalPrice)