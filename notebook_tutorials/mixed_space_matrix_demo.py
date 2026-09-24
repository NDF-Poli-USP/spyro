"""Caso mais simples possível: só a equação da onda acústica (escalar).

Sem sólido, sem interface, sem espaço misto — só uma malha, um espaço de
função, e a matriz de massa se formando na sua frente.
"""

import numpy as np
import firedrake as fire
from firedrake import FunctionSpace, TrialFunction, TestFunction, dx, assemble

np.set_printoptions(precision=3, suppress=True, linewidth=140)

# =============================================================================
# PASSO 1 — a malha
# =============================================================================
# Um intervalo 1D dividido em 3 elementos -> 4 nós físicos (0, 1, 2, 3).
# Uso 1D só pra ficar mais fácil de visualizar; o mecanismo é o mesmo em 2D/3D.
mesh = fire.UnitIntervalMesh(3)

print("=" * 70)
print("PASSO 1 — a malha")
print("=" * 70)
print(f"Número de elementos: {mesh.num_cells()}")
print(f"Coordenadas dos nós: {mesh.coordinates.dat.data}")


# =============================================================================
# PASSO 2 — o espaço de função (onde 'mora' a pressão p)
# =============================================================================
V = FunctionSpace(mesh, "CG", 1)  # grau 1: funções de forma lineares (retas)
Np = V.dim()

print("\n" + "=" * 70)
print("PASSO 2 — o espaço de função V")
print("=" * 70)
print(f"V tem {Np} graus de liberdade — um por nó físico, já que é grau 1")


# =============================================================================
# PASSO 3 — TrialFunction e TestFunction
# =============================================================================
# p_trial representa "a incógnita que estou resolvendo" (p no passo n+1).
# q_test é usada pra projetar a equação em cada direção da base -- é o
# mecanismo do método de Galerkin.
p_trial = TrialFunction(V)
q_test = TestFunction(V)

print("\n" + "=" * 70)
print("PASSO 3 — TrialFunction (p_trial) e TestFunction (q_test)")
print("=" * 70)
print("Esses dois objetos ainda não têm valor nenhum -- são só símbolos")
print("que o UFL usa pra você escrever a forma bilinear matematicamente.")


# =============================================================================
# PASSO 4 — a forma bilinear da matriz de massa
# =============================================================================
# Essa É a expressão matemática: integral de p_trial * q_test sobre o domínio.
# É exatamente o termo 'm1' (sem o /dt**2 e sem o resto) do seu
# build_acoustic_form.
massa_form = p_trial * q_test * dx

print("\n" + "=" * 70)
print("PASSO 4 — a forma bilinear: p_trial * q_test * dx")
print("=" * 70)
print("Isso ainda não é uma matriz -- é a RECEITA de como calcular cada")
print("entrada. assemble() é quem executa essa receita pra cada par (i,j).")


# =============================================================================
# PASSO 5 — assemble() executa a receita, preenchendo a matriz
# =============================================================================
M = assemble(massa_form, mat_type="aij").petscmat[:, :]

print("\n" + "=" * 70)
print("PASSO 5 — assemble(massa_form) -> a matriz de massa M")
print("=" * 70)
print(f"M tem shape {M.shape}  (Np x Np = {Np} x {Np}, porque só tem UM campo)")
print(M)


# =============================================================================
# PASSO 6 — confirmando manualmente UMA entrada, pra tirar a caixa-preta
# =============================================================================
# M[1,1] deveria ser a integral de phi_1 * phi_1 sobre a malha inteira.
# phi_1 é a função "chapéu" que vale 1 no nó 1 e 0 nos nós 0 e 2 (os únicos
# vizinhos onde ela não é zero por fora do seu próprio elemento).
h = 1.0 / 3  # tamanho de cada elemento nessa malha

# A integral de uma "função chapéu" ao quadrado, sobre os dois elementos
# que a tocam (um de cada lado), tem fórmula fechada conhecida: 2h/3.
valor_esperado = 2 * h / 3

print("\n" + "=" * 70)
print("PASSO 6 — conferindo M[1,1] na mão, sem depender do assemble()")
print("=" * 70)
print(f"M[1,1] calculado pelo Firedrake: {M[1, 1]:.6f}")
print(f"Valor esperado pela fórmula analítica 2h/3 = {valor_esperado:.6f}")
print(f"Batem: {np.isclose(M[1, 1], valor_esperado)}")

print(
    "\nResumindo o fluxo inteiro:\n"
    "  malha -> espaço de função (Np dofs) -> TrialFunction/TestFunction\n"
    "  -> forma bilinear (a receita) -> assemble() (executa a receita)\n"
    "  -> matriz Np x Np, uma entrada por par (nó_teste, nó_trial).\n"
    "Isso é TUDO que acontece pro caso acústico puro -- sem espaço misto,\n"
    "sem blocos, sem acoplamento. É a base sobre a qual tudo que vimos\n"
    "antes (espaço misto, C_pu, etc.) se constrói."
)