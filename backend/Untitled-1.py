def checkPrime(x):
    if x < 2:
        return False
    for j in range(2, int(x ** 0.5) + 1):
        if x % j == 0:
            return False
    return True

number = int(input("Enter a number: "))
print(f"{number} is {'prime' if checkPrime(number) else 'not prime'}.")


def checkPrime(x):
    if x < 2:
        return False
    for j in range(2, int(x ** 0.5) + 1):
        if x % j == 0:
            return False
    return True

number = int(input("Enter a number: "))
print(f"{number} is {'prime' if checkPrime(number) else 'not prime'}.")
