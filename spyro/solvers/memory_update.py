"""Atualização das variáveis de memória GSLS, rápida e visível ao pyadjoint.

As variáveis de memória são guardadas no espaço do deslocamento (chi_l em V),
e a deformação de memória é obtida por linearidade: zeta_l = eps(chi_l).
Isso é exato porque a recorrência tem coeficientes escalares constantes
(y_l, omega_l) e condição inicial nula. A variação espacial da atenuação
(Q_vp, Q_vs) entra só em Gamma, dentro da forma, e não na recorrência.

Recorrência (Euler explícito, como no código original):
    chi_l^{n+1} = a_l chi_l^n + b_l u^n,   a_l = 1 - dt*omega_l,  b_l = dt*omega_l
    chi_mem     = sum_l y_l chi_l^{n+1}

As contas são feitas direto nos Vec do PETSc. Para o adjunto automático,
um único bloco do pyadjoint por passo grava a operação inteira, com
recompute (checkpointing), adjunto, TLM e Hessiana. Como o bloco é linear
com coeficientes constantes, o adjunto é a recorrência transposta:
    lambda_l     = adj(chi_l^{n+1}) + y_l adj(chi_mem)
    adj(u^n)     = sum_l b_l lambda_l
    adj(chi_l^n) = a_l lambda_l
"""
from firedrake import Cofunction, Function
from pyadjoint import Block, annotate_tape, get_working_tape, stop_annotating


def _axpy(vec_out, alpha, obj):
    """vec_out += alpha * obj (ignora obj None)."""
    if obj is None or alpha == 0.0:
        return
    with obj.dat.vec_ro as v:
        vec_out.axpy(alpha, v)


class MemoryUpdateBlock(Block):
    """Bloco pyadjoint para a atualização das L variáveis de memória.

    Dependências: [u_old, chi_0, ..., chi_{L-1}] (versões antigas)
    Saídas:       [chi_0, ..., chi_{L-1}, chi_mem] (versões novas)
    """

    def __init__(self, u_old, chis, a, b, y):
        super().__init__()
        self.a = tuple(float(v) for v in a)
        self.b = tuple(float(v) for v in b)
        self.y = tuple(float(v) for v in y)
        self.L = len(chis)
        self.add_dependency(u_old)
        for chi in chis:
            self.add_dependency(chi)

    def __str__(self):
        return f"MemoryUpdateBlock({self.L} ramos)"

    # ------------------------------------------------------------------
    # Aplicação direta (forward / recompute / TLM): saída idx a partir de
    # valores (ou perturbações) de u e dos chi antigos.
    # ------------------------------------------------------------------
    def _forward_component(self, u, chis, idx, space):
        out = Function(space)
        with out.dat.vec_wo as vo:
            vo.zeroEntries()
            if idx < self.L:
                _axpy(vo, self.a[idx], chis[idx])
                _axpy(vo, self.b[idx], u)
            else:
                for c, a, b, y in zip(chis, self.a, self.b, self.y):
                    _axpy(vo, y * a, c)
                    _axpy(vo, y * b, u)
        return out

    # ------------------------------------------------------------------
    # Aplicação transposta (adjunto / Hessiana).
    # ------------------------------------------------------------------
    def _lambdas(self, out_values, dual_space):
        mem = out_values[self.L]
        lams = []
        for l in range(self.L):
            lam = Cofunction(dual_space)
            with lam.dat.vec_wo as vl:
                vl.zeroEntries()
                _axpy(vl, 1.0, out_values[l])
                _axpy(vl, self.y[l], mem)
            lams.append(lam)
        return lams

    def _transpose_component(self, lams, idx, dual_space):
        out = Cofunction(dual_space)
        with out.dat.vec_wo as vo:
            vo.zeroEntries()
            if idx == 0:                      # d/du_old
                for lam, b in zip(lams, self.b):
                    _axpy(vo, b, lam)
            else:                             # d/dchi_l antigo
                l = idx - 1
                _axpy(vo, self.a[l], lams[l])
        return out

    # ---- recompute (replay / checkpointing) ----
    def recompute_component(self, inputs, block_variable, idx, prepared):
        return self._forward_component(
            inputs[0], inputs[1:], idx, inputs[0].function_space())

    # ---- adjunto ----
    def prepare_evaluate_adj(self, inputs, adj_inputs, relevant_dependencies):
        return self._lambdas(adj_inputs, inputs[0].function_space().dual())

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx,
                               prepared=None):
        return self._transpose_component(
            prepared, idx, inputs[0].function_space().dual())

    # ---- TLM ----
    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx,
                               prepared=None):
        return self._forward_component(
            tlm_inputs[0], tlm_inputs[1:], idx, inputs[0].function_space())

    # ---- Hessiana (bloco linear: só a parte transposta) ----
    def prepare_evaluate_hessian(self, inputs, hessian_inputs, adj_inputs,
                                 relevant_dependencies):
        return self._lambdas(hessian_inputs,
                             inputs[0].function_space().dual())

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs,
                                   block_variable, idx, relevant_dependencies,
                                   prepared=None):
        return self._transpose_component(
            prepared, idx, inputs[0].function_space().dual())


def update_memory_variables(u_old, chis, chi_mem, a, b, y):
    """Atualiza chi_l e chi_mem in place, gravando um bloco se houver fita.

    Parameters
    ----------
    u_old : firedrake.Function
        Deslocamento u^n (após a rotação de estado, ``wave.u_nm1``).
    chis : list of firedrake.Function
        Variáveis de memória chi_l, no mesmo espaço de ``u_old``.
    chi_mem : firedrake.Function
        Combinação sum_l y_l chi_l, usada na forma variacional.
    a, b, y : sequence of float
        Coeficientes da recorrência e pesos GSLS.
    """
    annotate = annotate_tape()
    if annotate:
        # Dependências antes da atualização: captura as versões antigas.
        block = MemoryUpdateBlock(u_old, chis, a, b, y)
        get_working_tape().add_block(block)

    with stop_annotating():
        with u_old.dat.vec_ro as vu, chi_mem.dat.vec_wo as vmem:
            vmem.zeroEntries()
            for chi, al, bl, yl in zip(chis, a, b, y):
                with chi.dat.vec as vc:
                    vc.axpby(bl, al, vu)      # vc = bl*vu + al*vc
                    vmem.axpy(yl, vc)         # vmem += yl*vc

    if annotate:
        # Saídas depois da atualização: cria as versões novas.
        for chi in chis:
            block.add_output(chi.create_block_variable())
        block.add_output(chi_mem.create_block_variable())
