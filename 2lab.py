import random
chisla = []
for i in range(1000):
    chisla.append(random.randint(1, 1000))
suma = 0
for num in chisla:
    if num % 2 == 0:
        suma += num
print("Список:", chisla)
print("Сумма четных чисел:", suma)
