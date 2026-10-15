# Aceleração separada da atualização temporal: relatório

Objetivo: substituir o solve para $u^{n+1}$ por

$$M(c)\,a^n = f^n - K u^n - g(u^n,u^{n-1}), \qquad u^{n+1} = 2u^n - u^{n-1} + \Delta t^2 a^n,$$

preservando forward e gradiente, e medir se isso reduz o armazenamento do adjunto automático (pyadjoint + `checkpoint_schedules`).

**Conclusão até aqui:** a reformulação preserva os resultados até o arredondamento, mas **não reduz memória no tape genérico: aumenta**, em todas as agendas. O original já guarda só um campo de pressão próprio por passo. O desperdício real está em outro lugar: cópias redundantes nos checkpoints do pyadjoint e cofunções da fonte pontual. Recomendação: **não adotar a reformulação como padrão** sem antes resolver isso (ver "O que falta").

Branch: `claude/spyro-acceleration-adjoint-870c10` (base `main` @ `d5458bef`), **sem commits**.

---

## 1. O que foi feito

### 1.1 Código (sem PML)

| Arquivo | Mudança |
|---|---|
| `spyro/solvers/acoustic_solver_construction_no_pml.py` | O solve passa a ser para `wave.acceleration` com `M(c)` (mesmo espaço, quadratura e `solver_parameters`, `mat_type=matfree`). A fonte, a NRBC e o amortecimento HABC continuam como termos explícitos no lado direito, sem alteração. `CentralDifferenceAccelerationSolver.solve()` executa o solve da aceleração e depois `u_np1.assign(2u_n - u_nm1 + dt²a)`. `build_acceleration_solver(wave, bcs=None)` é reutilizável. |
| `spyro/solvers/mms_acoustic.py` | Reaproveita `build_acceleration_solver` com Dirichlet homogêneo na aceleração (antes duplicava a montagem do solver). |
| `spyro/solvers/acoustic_wave.py` | Declara `self.acceleration` em `matrix_building`. |
| `spyro/solvers/time_integration_central_difference.py` | Só o comentário: `end_timestep()` continua logo após o avanço completo. |

Como `wave.solver.solve()` continua avançando `u_n → u_np1`, o integrador forward, o adjunto implementado (`backward_time_integration.py`), os receptores e o funcional não mudaram. A PML não foi tocada.

**Posição de `end_timestep()`:** ela continua depois do par solve + atualização. O estado de reinício esperado é `{u^n, u^{n+1}}`. No entanto, o `LinearVariationalSolver` registra a própria solução como dependência (chute inicial), e por isso a aceleração anterior entra no `checkpointable_state`. Resultado: o reinício custa 3 campos em vez de 2. A variante *in-place*, que resolve `a` dentro do buffer `u_np1`, elimina isso (medida, não proposta).

### 1.2 Testes

`tests/on_one_core/test_acceleration_central_difference.py` (6 testes, todos passando):

- `test_acceleration_step_matches_pressure_solve[None|all_sides]`: um passo comparado com o solve de $u^{n+1}$ escrito de forma independente (massa agrupada, rigidez e NRBC). Erro relativo < 1e-12; aceleração < 1e-10.
- `test_gradient_with_few_checkpoints[None|all_sides]`: mixed com 3 snapshots (verifica que houve recomputação) contra o tape sem checkpointing. J e gradiente < 1e-12.
- `test_reevaluation_after_velocity_change`: replay do tape checkpointado em um novo $c$ contra uma gravação nova em $c$, com NRBC. J < 1e-12, gradiente < 1e-10.
- `test_taylor_test_with_nrbc_and_recomputation`: taxa > 1.9 com NRBC e mixed de 5 snapshots.

Regressão: 85 testes passam (`test_gradient_checkpointing`, `test_gradient_auto_adjoint_nrbc`, `test_gradient_2d`, `test_MMS`, `test_variational_solver`, `test_analytical_solution`, `test_fwi_*`, `test_serial_shots_adjoint`, `test_time_convergence`, `test_forward_examples`, `test_forward_acoustic`, entre outros).

9 falhas em `test_reference_model_abc.py` (8) e `test_habc_tools.py::test_habc_tools[T-3]` **já existiam**: falham igual, com os mesmos valores, no código original.

`make lint` passa.

### 1.3 Problema de referência

Acústico 2D, sem PML, malha `firedrake_mesh` com `edge_length=0.05`, P2 com massa agrupada (2481 dofs, 1 campo = 19 848 B). $\Delta t = 10^{-3}$, $T=0.8$ (801 passos). Uma fonte Ricker de 5 Hz e 5 receptores.

- Modelo verdadeiro: duas camadas (1.5/3.5).
- Chute: heterogêneo, $1.8 - 0.6z + 0.2\,e^{-(x-0.5)^2/0.02}$.

(Com `dt=0.004` nessa malha o forward explode: J ≈ 1e283. O assert de instabilidade só é verificado a cada `output_frequency` passos.)

### 1.4 Equivalência (33 configurações: 3 formulações × single/mixed/Revolve × 4…64 snapshots)

Pior erro relativo contra o original com tape completo:

| J | receptores | campo final | gradiente |
|---|---|---|---|
| 1.2e-15 | 1.2e-14 | 9.0e-15 | 2.4e-14 |

### 1.5 Conteúdo do tape (o que realmente é guardado)

Um passo do tape original tem os blocos: rotação (2 assigns), interpolação nos receptores, misfit e funcional, montagem da fonte, `assign` da fonte e solve.

- **Rotação:** `u_nm1.assign(u_n)` e `u_n.assign(u_np1)` usam `DelegatedFunctionCheckpoint`, ou seja, o checkpoint é compartilhado e não copiado.
- **Tape completo / `SingleMemoryStorageSchedule`:** 3 campos próprios por passo no original. Apenas 1 é pressão (`u^{n+1}`); os outros 2 são cofunções da fonte pontual (`assemble(wavelet*source_cof)` e o `assign` para `source_function`). Na reformulação são 4 por passo (`a^n` e `u^{n+1}` passam a ser saídas próprias).
- **Unidades do mixed:** censo da 1ª passagem reversa. Os campos `c` e `source_cof` são compartilhados e não entram na contagem.

| unidade | original | reformulada | necessário em princípio |
|---|---|---|---|
| reinício (`write_ics`) | 2 campos | 3 (+ `a` como chute inicial) | 2 |
| dependências do adjunto (`write_adj_deps`) | 7 cópias | 9 cópias | 1 ($a^n$) |

`TimeStep.checkpoint` faz `saved_output._ad_create_checkpoint()` para cada block variable, o que gera uma cópia profunda até dos delegados. Assim, `u^n` (que é o `u^{n+1}` do passo anterior) é copiado de novo.

- **Registros observados:** o `real_shot` criado por passo no VOM nasce com `last_use=-1` e entra no `checkpointable_state` de todos os passos anteriores. Cada checkpoint de reinício regravado no reverso leva cerca de 190 desses (`nt·nrec` no total). Aqui são pequenos (5 receptores), mas crescem com a aquisição.

### 1.6 Memória: pico de checkpoints no reverso (em campos) e passos recomputados

| agenda | s | orig mem / recomp. | nova | in-place |
|---|---|---|---|---|
| tape completo (single) | – | 2408 / 0 | 3209 / 0 | 3209 / 0 |
| mixed | 8 | 79 / 2717 | 95 | 88 |
| mixed | 16 | 151 / 1521 | 183 | 168 |
| mixed | 32 | 295 / 1048 | 359 | 328 |
| Revolve | 8 | 25 / 3290 | 33 | 26 |
| Revolve | 16 | 41 / 2232 | 57 | 42 |
| Revolve | 32 | 73 / 1808 | 105 | 74 |
| Revolve | 64 | 137 / 1536 | 201 | 138 |

As recomputações são idênticas entre formulações para o mesmo `s` (dependem só da agenda).

**Mesmo orçamento de memória (original):**

| orçamento | mixed | Revolve |
|---|---|---|
| 50 campos | s=4: 5740 recomp. | s=16: 2232 |
| 100 campos | s=8: 2717 | s=32: 1808 |
| 200 campos | s=16: 1521 | s=64: 1536 |

Cada snapshot do mixed custa cerca de 9 campos e cada um do Revolve cerca de 2. A docstring de `AutomatedAdjoint` afirma que o mixed recomputa menos "para o mesmo número de unidades", mas isso não se sustenta quando se compara por bytes, ao menos até cerca de 200 campos. A otimalidade do mixed depende de unidades de custo igual, e aqui elas não são iguais.

Tabela completa: `results/analyse_probe.txt`. Censos: `results/census/`.

### 1.7 Tempo (preliminar)

Rodadas isoladas no início, sem o probe:

| | tape completo | single |
|---|---|---|
| original | 12.8 s (fwd 2.8 + grad 10.0) | 13.4 s |
| reformulado | 19.6 s (3.7 + 15.9) | 19.2 s |

A varredura de tempo completa **não foi feita**: a máquina estava ocupada por outros processos (`case.py` de `latent_diagnostics`, load average > 100).

---

## 2. O que falta fazer

1. **Varredura de tempo e RSS** com a máquina ociosa, um processo por vez:
   ```
   ./setup_trees.sh
   ./sweep.sh timing 1
   python analyse.py
   ```
2. **Decidir sobre a reformulação.** Os dados indicam não adotar como padrão. Se for mantida (por clareza ou para preparar um reverso especializado), adotar a variante *in-place* ou evitar a dependência do chute inicial, para não pagar +1 campo por checkpoint de reinício.
3. **Passo 6 (reverso especializado):** o armazenamento genérico é de fato um gargalo (7 cópias por unidade de dependências do adjunto, contra 1 necessária). Mas um bloco spyro sozinho não resolve, porque o `CheckpointManager` do pyadjoint revisa `adjoint_dependencies` adicionando **todas** as dependências de todos os blocos do passo, sem consultar o bloco. Caminhos possíveis:
   - (a) no pyadjoint/Firedrake: não copiar de novo checkpoints delegados/compartilhados em `TimeStep.checkpoint`. Isso é genérico e beneficia o original.
   - (b) API de bloco para declarar quais dependências o adjunto realmente lê. Só então um bloco de passo que use $\delta J_n = -{z^n}^T \delta M(c)\,a^n$ guardaria apenas $a^n$.
   - (c) alternativa já existente: o adjunto implementado do spyro, que guarda $u$ por passo.
4. **Fonte pontual:** 2 dos 3 campos por passo no tape completo são cofunções da fonte (montagem + `assign`), independentemente da formulação. Representar a fonte como amplitude escalar vezes uma cofunção fixa reduziria isso a um escalar por passo. É um ganho imediato e independente desta mudança.
5. **Registros observados no estado de reinício:** criar os `real_shot` sem que entrem retroativamente no `checkpointable_state` (por exemplo, pré-criá-los fora do laço ou como dados não anotados).
6. **Revisar a docstring de `AutomatedAdjoint`** sobre mixed vs Revolve, ou expor o Revolve no spyro com a ressalva do custo por unidade.
7. **Avaliar com NRBC/HABC e em malhas maiores** (a varredura usou só o caso sem absorção). `bench.py --abc nrbc` já suporta NRBC.
8. **Falhas pré-existentes** em `test_reference_model_abc.py` / `test_habc_tools.py[T-3]`: tratar em PR separado.
9. **Antes de um PR:** um humano revisa e assume cada mudança (AGENTS.md); declarar o uso de IA (Claude Code, Opus 5.5) para implementação, testes e medições; decidir se esta pasta de benchmark (3.3 MB, inclui `.npz`) entra no repositório.

## 3. Como reproduzir

```
source ~/dev_code/venv-firedrake/bin/activate
cd benchmarks/adjoint_acceleration
./setup_trees.sh                  # trees/baseline (original) e trees/inplace
./sweep.sh probe 4                # bytes de checkpoint (paralelo ok)
./sweep.sh timing 1               # tempo e RSS (máquina ociosa)
python analyse.py                 # tabelas
```

`bench.py` também aceita `--taylor`, `--reeval` (nova avaliação em outro modelo de velocidade) e `--abc nrbc`. Ele mede bytes de checkpoint percorrendo `TimeStep._checkpoint` e os `_checkpoint` das block variables, contando cada buffer uma vez, e conta recomputações pelas ações `Forward` executadas após a gravação.
