# Setup phase

# We'll use y^2 = x^3 + 3 for our curve, over F_101
p = 101
F = FiniteField(p)
R.<x_f> = F[]
E = EllipticCurve(F, [0, 3])
print(E)

e_points = E.points()
# Our generator G_1 = E(1, 2)
G_1 = e_points[1]

# Let's find a subgroup with this generator.
print(f"Finding a subgroup with generator {G_1} ...")
# `r+1` is smallest positive value for which [r+1]_1 = 0
r = 1
elem = G_1
while True:
    # G_1 + G_1 + {r_times} G_1 
    elem += G_1
    r += 1
    if elem == e_points[0]:
        break

print(f"Found subgroup of order r={r} using generator {G_1}")

# Our group have order 17 and our proofs
# will be a point in the group. So we would
# do all our operations in F17
F17 = Integers(17)
P.<x> = F17[]
x = P.0

# Now let's find the embedding degree.
# The embedding degree is the smallest k such that r|p^k - 1
# In other words:
# => p^k == 1 + r*x (where x is quotient)
# => p^k mod r == 1
k = 1
print(f"Finding embedding degree for {p}^k mod {r} ...")
while True:
    if p ^ k % r == 1:
        break
    k += 1
    
print(f"Found embedding degree: k={k}")


# Our extension field. The polynomial x^2+2 is irreducible in F_101.
F2.<u> = F.extension(x_f^2+2, 'u')
assert u^2 == -2
print(F2)
E2 = EllipticCurve(F2, [0, 3])
print(E2)
# One of the generators for this curve we can use is (36, 31u)
G_2 = E2(36, 31*u)

# Now we build the trusted setup. The SRS is a list of EC points
# parameterized by a randomly generated secret number s.
# According to the PLONK protocol paper, a circuit with n gates requires
# an SRS with at least n+5 elements.
# Check page 11 of plonk paper

# We choose 2 as our random number for demo purposes.
s = 2
# Our circuit will have 3 gates.
n_gates = 3

SRS = []
for i in range(0, n_gates+3):
	SRS.append(s^i * G_1)
for i in range(0, 2):
	SRS.append(s^i * G_2)
    
# Composing our circuit. We'll test a^2 + b^2 = c:
# x_1 * x_1 = x_2
# x_3 * x_3 = x_4
# x_2 + x_4 = x_5
#
# In order to satisfy these constraints, we need to supply five numbers
# as wire values that make all of the equations correct.
# e.g. x=(2, 4, 3, 9, 13) would work.

# A full PLONK gate looks like this:
# (q_L)*a + (q_R)*b + (q_O)*c + (q_M)*a*b + q_C = 0
#
# Where a, b, c are the left, right, output wires of the gate.
#
# a + b = c  ---> q_L=1 , q_R=1, q_O=-1, q_M=0, q_C = 0
# a * b = c  ---> q_O=-1, q_M=1, and the rest = 0
#
# To bind a variable to a public value:
# q_R = q_O = q_M = 0
# q_L = 1
# q_C = public_value
#
# Considering all inputs as private, we get these three PLONK gates
# representing our circuit:
# 0*a_1 + 0*b_1 + (-1)*c_1 + 1*a_1*b_1 + 0 = 0    (a_1 * b_1 = c_1)
# 0*a_2 + 0*b_2 + (-1)*c_2 + 1*a_2*b_2 + 0 = 0    (a_2 * b_2 = c_2)
# 1*a_3 + 1*b_3 + (-1)*c_3 + 0*a_3*b_3 + 0 = 0    (a_3 + b_3 = c_3)
#
# So let's test with (2, 3, 12)
# a_i (left) values will be (2, 3, 4)
# b_i (right) values will be (2, 3, 9)
# c_i (output) values will be (4, 9, 13)


# Selectors
q_L = vector(F17, [0, 0, 1])
q_R = vector(F17, [0, 0, 1])
q_O = vector(F17, [-1, -1, -1])
q_M = vector(F17, [1, 1, 0])
q_C = vector(F17, [0, 0, 0])
# Assignments
a = vector(F17, [2, 3, 4])
b = vector(F17, [2, 3, 9])
c = vector(F17, [4, 9, 13])

# Roots of Unity.
# The vectors for our circuit and assignment are all length 3, so the domain
# for our polynomial interpolation must have at least three elements.
roots_of_unity = []
F_r = FiniteField(r)
for i in F_r:
    # 4th root of unity
	if i^4 == 1:
		roots_of_unity.append(i)

# our domain size is 3 
omega_0 = roots_of_unity[0]
omega_1 = roots_of_unity[1]
omega_2 = roots_of_unity[3]


# Cosets
# k_1 not in H, k_2 not in H nor k_1H
k_1 = 2
k_2 = 3
H = vector(F17, [omega_0, omega_1, omega_2])
k1H = vector(F17, [H[0]*k_1, H[1]*k_1, H[2]*k_1])
k2H = vector(F17, [H[0]*k_2, H[1]*k_2, H[2]*k_2])
print("Polynomial interpolation using roots of unity")
print(f"H:   {H}")
print(f"k1H: {k1H}")
print(f"k2H: {k2H}")

# Interpolating using the Roots of Unity
# The interpolated polynomial will be degree-2 and have the form:
# f_a(x) = c + b*x + a*x^2
# f_a(1) = 2, f_a(4) = 3, f_a(16) = 4
# Note that the above x is H (the omegas)
#
# This gives a system of equations:
# f_a(1)  = c + b*1 + a*1^2   = 2
# f_a(4)  = c + b*4 + a*4^2   = 3
# f_a(16) = c + b*16 + a*16^2 = 4

# We can find coefficients of 3 degreee polynomial with evals (o1, o2, o3)
# at root of unity (1, 4, 16) by:
#   D * C = O 
#       where D is domain matrix
#             C is coeff vector
#             O is output vector
#   => C = D^-1 * O
# 
# D (in our case) is
#   1, 1, 1^2
#   1, 4, 4^2,
#   1, 16, 16^2


D = Matrix(F17, [
		[1^0, 1^1, 1^2],
		[4^0, 4^1, 4^2],
		[16^0, 16^1, 16^2],
])
Di = D.inverse()

# Now we can find polynomials (fa, fb, fc, fqL, fqR....)
# using matrix multiplication
fa = P(list(Di * a))
fb = P(list(Di * b))
fc = P(list(Di * c))
fqL = P(list(Di * q_L))
fqR = P(list(Di * q_R))
fqO = P(list(Di * q_O))
fqM = P(list(Di * q_M))
fqC = P(list(Di * q_C))

# The copy constraints involving left, right, output values are encoded as
# polynomials S_sigma_1, S_sigma_2, S_sigma_3 using the cosets we found
# earlier. The roots of unity H are used to label entries in vector a,
# the elements of k1H are used to label entries in vector b, and vector c is
# labeled by the elements of k2H.
print("Copy constraints:")
print(f"a: {a}")
print(f"b: {b}")
print(f"c: {c}")
# a1 = b1, a2 = b2, a3 = c1
sigma_1 = vector(F17, [k1H[0], k1H[1], k2H[0]])
print(f"sigma_1: {sigma_1}")
# b1 = a1, b2 = a2, b3 = c2
sigma_2 = vector(F17, [H[0], H[1], k2H[1]])
print(f"sigma_2: {sigma_2}")
# c1 = a3, c2 = b3, c3 = c3
sigma_3 = vector(F17, [H[2], k1H[2], k2H[2]])
print(f"sigma_3: {sigma_3}")

fsa = P(list(Di * sigma_1))
fsb = P(list(Di * sigma_1))
fsc = P(list(Di * sigma_1))


# Proving phase

# Round 1

# Create vanishing polynomial that evaluates to 0 
# in our subgroup (i.e. 4th root of unity).
# Since H = {x ε F_17 | x^4 == 1}
#   Z = x^4 - 1
Z = x^4 - 1
assert Z(1) == 0
assert Z(4) == 0
assert Z(16) == 0

# 9 random blinding values. We will use:
# 7, 4, 11, 12, 16, 2
# 14, 11, 7 (used in round 2)

# Blind our witness polynomials
# The blinding factors will disappear at the evaluation points.
a = (7*x + 4) * Z + fa
b = (11*x + 12) * Z + fb
c = (16*x + 2) * Z + fc

# So now we evaluate a, b, c at secret `s` with these powers of G
a_s = ZZ(a(s)) * G_1
b_s = ZZ(b(s)) * G_1
c_s = ZZ(c(s)) * G_1

# Round 2
