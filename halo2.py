def gcd(a, b):
    if (abs(b) > abs(a)):
        return gcd(b, a)

    while(abs(b) > 0):
        print(a, b)
        a, b = b, a % b
    return a

print(gcd(23, 37))

def extendedEuclideanAlgorithm(a, b):
    if (mod(b) > mod(a)):
        (x, y, a) = extendedEuclideanAlgorithm(b, a)
        return (y, x, a)
    
    if abs(b) == 0:
      return (1, 0, a)

    x1, x2, y1, y2 = 0, 1, 1, 0
    while abs(b) > 0:
        q, r = divmod(a, b)
        x = x2 - q*x1
        y = y2 - q*y1
        a, b, x2, x1, y2, y1 = b, r, x1, x, y1, y
    
    return (x2, y2, a)


