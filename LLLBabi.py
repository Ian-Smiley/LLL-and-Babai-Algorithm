#LLL Reduction Code used from Nao Yonashiro, user orisano on GitHub
#Combines the LLL algorithm with Babai's algorithm to show the closest vector combination to a target vector
#Takes 2 input vectors, size reduces and orthogonalizes them using gram schmidt orthogonalization to find the nearest point to a target vector
#Made by Ian Smiley

from fractions import Fraction
from typing import List, Sequence
import numpy as np
import matplotlib.pyplot as plt

class Vector(list):
    def __init__(self, x):
        super().__init__(map(Fraction, x))

    def sdot(self) -> Fraction:
        return self.dot(self)

    def dot(self, rhs: "Vector") -> Fraction:
        rhs = Vector(rhs)
        assert len(self) == len(rhs)
        return sum(map(lambda x: x[0] * x[1], zip(self, rhs)))

    def proj_coff(self, rhs: "Vector") -> Fraction:
        rhs = Vector(rhs)
        assert len(self) == len(rhs)
        return self.dot(rhs) / self.sdot()

    def proj(self, rhs: "Vector") -> "Vector":
        rhs = Vector(rhs)
        assert len(self) == len(rhs)
        return self.proj_coff(rhs) * self

    def __sub__(self, rhs: "Vector") -> "Vector":
        rhs = Vector(rhs)
        assert len(self) == len(rhs)
        return Vector(x - y for x, y in zip(self, rhs))

    def __mul__(self, rhs: Fraction) -> "Vector":
        return Vector(x * rhs for x in self)

    def __rmul__(self, lhs: Fraction) -> "Vector":
        return Vector(x * lhs for x in self)

    def __repr__(self) -> str:
        return "[{}]".format(", ".join(str(x) for x in self))

def gramschmidt(v: Sequence[Vector]) -> Sequence[Vector]:
    u: List[Vector] = []
    for vi in v:
        ui = Vector(vi)
        for uj in u:
            ui = ui - uj.proj(vi)
        if any(ui):
            u.append(ui)
    return u

def reduction(basis: Sequence[Sequence[int]], delta: float) -> Sequence[Sequence[int]]:
    n = len(basis)
    basis = list(map(Vector, basis))
    ortho = gramschmidt(basis)
    def mu(i: int, j: int) -> Fraction:
        return ortho[j].proj_coff(basis[i])
    k = 1
    while k < n:
        for j in range(k - 1, -1, -1):
            mu_kj = mu(k, j)
            if abs(mu_kj) > 0.5:
                basis[k] = basis[k] - basis[j] * round(mu_kj)
                ortho = gramschmidt(basis)
        if ortho[k].sdot() >= (delta - mu(k, k - 1)**2) * ortho[k - 1].sdot():
            k += 1
        else:
            basis[k], basis[k - 1] = basis[k - 1], basis[k]
            ortho = gramschmidt(basis)
            k = max(k - 1, 1)
    return [list(map(int, b)) for b in basis]



def babai_nearest_plane(basis: List[Vector], target: Vector) -> Vector:
    ortho = gramschmidt(basis)
    s = Vector(target)
    coffs = [0] * len(basis)
    for i in reversed(range(len(basis))): #The algorithm works backwards. Starts with last item, and ends with first
        babai = basis[i] #babai means regular basis
        babai_asterik = ortho[i] # babai asterick means Gram-Schmidt orthogonalized basis
        lambada = round(Vector(s).dot(babai_asterik) / babai_asterik.sdot())
        coffs[i] = lambada
        s = s - lambada * babai
    return Vector(target) - s, coffs


if __name__ == "__main__":
    # Test case
    test_vectors = [[6, 3], [12, 13]]
    reduced_basis = list(map(Vector, reduction(test_vectors, 0.75)))
  
    #Print Original Basis stacked
    print("Original basis:")
    for vector in test_vectors:
        print(vector)
    #Print LLL Basis stacked
    print("\nLLL Basis:")
    for vector in reduced_basis:
        print(vector)

    #Print Target Vector and Coefficients used to get to Nearest
    target_vector = Vector([6, 20])
    nearest, coffs = babai_nearest_plane(list(map(Vector, reduced_basis)), target_vector)
    print("\nTarget Vector:", target_vector)
    print("Nearest Lattice Vector:", nearest)
    print("Coefficients:", coffs)


v1 = np.array([float(x) for x in reduced_basis[0]])
v2 = np.array([float(x) for x in reduced_basis[1]])

x1 = np.array(nearest[0]) #First output of nearest point
x2 = np.array(nearest[1]) #Second output of nearest point

z1 = np.array(target_vector[0]) #First output of target point
z2 = np.array(target_vector[1]) #Second output of target point



# Range of multipliers for each basis vector
plottedpoints = range(-8, 8)  #Range that is plotted for graph

# Generate lattice points
lattice_points = []
for a in plottedpoints:
    for b in plottedpoints:
        point = a * v1 + b * v2
        lattice_points.append(point)

# Convert to arrays for plotting
lattice_points = np.array(lattice_points)
x_coords = lattice_points[:, 0]
y_coords = lattice_points[:, 1]


# Plot settings
plt.figure(figsize=(8, 8)) #Size of the graph
plt.axhline(0, color="black", linestyle="--") #Plots an axis with -- type of lines
plt.axvline(0, color="black", linestyle="--") #Plots the other axis

# Plot lattice points
plt.scatter(x_coords, y_coords, color='blue', label='Lattice Points', zorder = 3) #Plot hopefully the lattice
plt.scatter([0], [0], color='red', label='Origin', zorder=3) #Plot 0,0. Origin where want to start
plt.scatter(x1, x2, color = 'brown', zorder = 3, label = 'Nearest Point') #Plot nearest point algorithm found
plt.scatter(z1, z2, color = 'black', label = 'Target Point', zorder = 5) #Plot target point

# Plot the basis vectors as arrows
plt.quiver(0, 0, v1[0] * coffs[0], v1[1] * coffs[0], angles='xy', scale_units='xy', scale=1, color='orange', label = 'Vector 1')   #Vector number 1
tx = v1[0] *coffs[0]
ty = v1[1] *coffs[0]
plt.quiver(tx, ty, v2[0] * coffs[1], v2[1] *coffs[1], angles='xy', scale_units='xy', scale=1, color='green', label = 'Vector 2') #Vector number 2



# Axes settings
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Lattice from Basis Vectors")
plt.grid(True)
plt.legend(loc = 'upper left')
plt.tight_layout()
plt.show()
