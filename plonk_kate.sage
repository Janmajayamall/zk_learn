# Based upon 
# https://research.metastate.dev/plonk-by-hand-part-1/
# https://github.com/darkrenaissance/darkfi/tree/master/script/research

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
n_gates = 4

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


# Roots of unity and Cosets
# k_1 not in H, k_2 not in H nor k_1H
k_1 = 2
k_2 = 3
H = vector(F17, [1, 4, 16, 13])
k1H = vector(F17, [H[0]*k_1, H[1]*k_1, H[2]*k_1, H[3]*k_1])
k2H = vector(F17, [H[0]*k_2, H[1]*k_2, H[2]*k_2, H[3]*k_2])
print("Polynomial interpolation using roots of unity")
print(f"H:   {H}")
print(f"k1H: {k1H}")
print(f"k2H: {k2H}")


# Selectors
# Since we only need 3 gates, but have 
# 4 4th roots of unity, our last gate only 
# consists of 0
q_L = vector(F17, [0, 0, 1, 0])
q_R = vector(F17, [0, 0, 1, 0])
q_O = vector(F17, [-1, -1, -1, 0])
q_M = vector(F17, [1, 1, 0, 0])
q_C = vector(F17, [0, 0, 0, 0])
# Assignments
a = vector(F17, [2, 3, 4, 0])
b = vector(F17, [2, 3, 9, 0])
c = vector(F17, [4, 9, 13, 0])


# Interpolating polynomials on roots_of_unity
# The interpolated polynomial will be degree-3 and have the form (fa):
# fa(x) = d + cx + b*x^2 + a*x^3
# fa(1) = 2, fa(4) = 3, fa(16) = 4, f(13) = 0
#
# This gives a system of equations:
# fa(x={1,3,16,13})  = d + cx + b*x^2 + a*x^3  = {2,3,4,0}

# We can find coefficients of 4 degreee polynomial with evals
# at (1, 4, 16, 3) by:
#   D * C = O 
#       where D is domain matrix
#             C is coeff vector
#             O is output vector
#   => C = D^-1 * O
# 
# D (in our case) is
#   1, 1, 1^2, 1^3
#   1, 4, 4^2, 4^3
#   1, 16, 16^2, 16^3
#   1, 13, 13^2, 13^2


D = Matrix(F17, [
		[1^0, 1^1, 1^2, 1^3],
		[4^0, 4^1, 4^2, 4^3],
		[16^0, 16^1, 16^2, 16^3],
        [13^0, 13^1, 13^2, 13^3],
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
# a1 = b1, a2 = b2, a3 = c1, a4=a4
sigma_1 = vector(F17, [k1H[0], k1H[1], k2H[0], H[3]])
print(f"sigma_1: {sigma_1}")
# b1 = a1, b2 = a2, b3 = c2, b4=b4
sigma_2 = vector(F17, [H[0], H[1], k2H[1], k1H[3]])
print(f"sigma_2: {sigma_2}")
# c1 = a3, c2 = b3, c3 = c3, c4=c4
sigma_3 = vector(F17, [H[2], k1H[2], k2H[2], k2H[3]])
print(f"sigma_3: {sigma_3}")

fsa = P(list(Di * sigma_1))
fsb = P(list(Di * sigma_2))
fsc = P(list(Di * sigma_3))


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
assert Z(13) == 0

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

# Random transcript challenges
beta = 12
gamma = 13
# Build accumulation
acc = 1
accs = []
for i in range(4):
    # H_{n + j} corresponds to b(H[i])
    # and H_{2n + j} is c(H[i])
    accs.append(acc)
    acc = acc * (
        (a(H[i]) + beta * H[i] + gamma)
        * (b(H[i]) + beta * k_1 * H[i] + gamma)
        * (c(H[i]) + beta * k_2 * H[i] + gamma) /
        (
            (a(H[i]) + beta * fsa(H[i]) + gamma)
            * (b(H[i]) + beta * fsb(H[i]) + gamma)
            * (c(H[i]) + beta * fsc(H[i]) + gamma)
        )
    )

acc = P(list(Di * vector(F17, accs)))

Zx = (14*x^2 + 11*x + 7) * Z + acc
# Evaluate z(x) at our secret point
Z_s = ZZ(Zx(s)) * G_1


# Round 3

alpha = 15

t1Z = a * b * fqM + a * fqL + b * fqR + c * fqO + fqC

t2Z = ((a + beta * x + gamma)
    * (b + beta * k_1 * x + gamma)
    * (c + beta * k_2 * x + gamma)) * Zx * alpha

# w[1] is our first root of unity
Zw = Zx(H[1] * x)
t3Z = -((a + beta * fsa + gamma)
    * (b + beta * fsb + gamma)
    * (c + beta * fsc + gamma)) * Zw * alpha

# Lagrangian polynomial which evaluates to 1 at 1
# L_1(w_1) = 1 and 0 on the other evaluation points
L = P(list(Di * vector(F17, [1, 0, 0, 0])))
print(L)
assert L(1) == 1
assert L(4) == 0
assert L(16) == 0
assert L(13) == 0

t4Z = (Zx - 1) * L * alpha^2

tZ = t1Z + t2Z + t3Z + t4Z
# and cancel out the factor Z now
t = P(tZ / Z)

# Split t into 3 parts
# t(X) = t_lo(X) + X^n t_mid(X) + X^{2n} t_hi(X)
t_list = t.list()
t_lo = t_list[0:6]
t_mid = t_list[6:12]
t_hi = t_list[12:18]
# and create the evaluations
t_lo_s = ZZ(P(t_lo)(s)) * G_1
t_mid_s = ZZ(P(t_mid)(s)) * G_1
t_hi_s = ZZ(P(t_hi)(s)) * G_1

# Round 4

zeta = 5

a_ = a(zeta)
b_ = b(zeta)
c_ = c(zeta)
sa_ = fsa(zeta)
sb_ = fsb(zeta)
t_ = t(zeta)
zw_ = Zx(zeta * H[1])
l_ = L(zeta)

r1 = a_ * b_ * fqM + a_ * fqL + b_ * fqR + c_ * fqO + fqC

r2 = ((a_ + beta * zeta + gamma)
    * (b_ + beta * k_1 * zeta + gamma)
    * (c_ + beta * k_2 * zeta + gamma)) * Zx * alpha

r3 = -((a_ + beta * sa_ + gamma)
    * (b_ + beta * sb_ + gamma)
    * beta * zw_ * fsc * alpha)

r4 = Zx * l_ * alpha^2

r = r1 + r2 + r3 + r4

r_ = r(zeta)

# Round 5

vega = 12

v1 = P(t_lo)
# Polynomial was in parts consisting of 6 powers
v2 = zeta^6 * P(t_mid)
v3 = zeta^12 * P(t_hi)
v4 = -t_
v5 = (
    vega * (r - r_)
    + vega^2 * (a - a_) + vega^3 * (b - b_) + vega^4 * (c - c_)
    + vega^5 * (fsa - sa_) + vega^6 * (fsb - sb_)
)

W = v1 + v2 + v3 + v4 + v5
Wz = W / (x - zeta)
# Calculate the opening proof
Wzw = (Zx - zw_) / (x - zeta * H[1])

# Compute evaluations of Wz and Wzw
Wz_s = ZZ(Wz(s)) * G_1
Wzw_s = ZZ(Wzw(s)) * G_1


# Verification Phase
