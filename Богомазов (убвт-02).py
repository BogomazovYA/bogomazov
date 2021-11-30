import math, time
def decoratorFunc(func):
        def wrapper(*args):
                print("Вызвана функция: {}.".format(func.__name__))
                start = time.time()
                result = func(*args)
                print("На выполнение было потрачено: {} секунд".format(time.time()-start))
                return result
        return wrapper

@decoratorFunc
def getSquare(s, n):
	return (n*(math.pow(s, 2)))/(4*math.tan(math.pi/n))

@decoratorFunc
def getSum(n):
    return math.ceil((n*(n+1))/2)
        
        
s = float(input("Длина стороны: "))
n = math.ceil(float(input("Количество сторон: ")))
print("Площадь: ", getSquare(s, n))

n = int(input("Положительное число: "))
print("Сумма первых чисел: ", getSum(n))
