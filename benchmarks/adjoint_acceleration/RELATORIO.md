# Aceleração separada da atualização temporal: relatório

Objetivo: substituir o solve para $u^{n+1}$ por

$$M(c)\,a^n = f^n - K u^n - g(u^n,u^{n-1}), \qquad u^{n+1} = 2u^n - u^{n-1} + \Delta t^2 a^n,$$

preservando forward e gradiente, e medir se isso reduz o armazenamento do adjunto automático (pyadjoint + `checkpoint_schedules`).

**Conclusão:** a reformulação preserva os resultados até o arredondamento, mas **não reduz memória no tape genérico: aumenta**, em todas as agendas, e o tempo preliminar também é pior. **Decisão: manter o solver do `main`** (solve para $u^{n+1}$). A reformulação foi revertida. Da investigação ficou a fonte pontual como amplitude escalar (seção 1.8), aplicada sobre o solver do `main`: o tape completo cai 3× (2408 → 808 campos).

**Atenção (seção 1.9):** no `main`, refazer o tape checkpointado em outro modelo de velocidade dá gradiente errado (erro de 2e-3). A FWI com adjunto automático via TAO faz exatamente isso a cada iteração. O bug já existia antes desta branch e a causa ainda não foi encontrada. A formulação por aceleração não tinha esse problema.

Branch: `claude/spyro-acceleration-adjoint-870c10` (base `main` @ `d5458bef`). Os commits `wip` e `cc246b6` têm a reformulação. O commit seguinte a reverte para o solver do `main` e mantém só a fonte por amplitude. As seções 1.1 a 1.7 descrevem a reformulação; o código dela fica no histórico (`cc246b6`).

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

### 1.8 Fonte pontual como amplitude escalar (item 4)

Implementado no caminho sem PML com fonte em VOM (o que o adjunto automático usa). A fonte entra na forma como

$$s(t)\int q_h\,v\,dx, \qquad M q_h = q,$$

onde `wave.source_amplitude` ($s$) é uma `Function` no espaço `R` e `wave.source_density` ($q_h$) é calculada uma vez por propagação por `set_point_source` (um solve com a massa da mesma quadratura; com massa agrupada é exato). Por passo, o tape registra só `source_amplitude.assign(wavelet[step])`: 8 bytes, em vez de `assemble(wavelet*source_cof)` mais o `assign` em `source_function` (2 campos).

Por que não `s * source_cof` diretamente no lado direito: o Firedrake 2026.10 só avalia pesos de `FormSum` que sejam `Constant`/`Function` puros, e o residual `J u - L` transforma o peso em `-1*s`, o que dá `ValueError: Expecting a scalar weight expression`.

O adjunto implementado reutiliza o mesmo solver e injeta a fonte adjunta por `source_function`. Por isso `backward_wave_propagator` zera `source_amplitude` (o forward a deixa no último valor da wavelet). O PML e o elástico não mudaram.

Medição sobre o solver do `main` (mesmo problema da 1.3, Firedrake 2026.10 em Docker, `bench.py --probe-bytes`):

| agenda | `main` (campos) | `main` + fonte por amplitude | recomp. |
|---|---|---|---|
| tape completo (single) | 2408 | **808** | 0 / 0 |
| mixed s=16 | 156.0 | 140.0 | 1521 / 1521 |

Equivalência contra o `main`: J 8.9e-15, receptores 4.7e-14, campo final 3.6e-14, gradiente 5.8e-14. Os tempos dessas rodadas não valem como medição, porque o `--probe-bytes` percorre o tape a cada ação.

No tape completo, cada passo passa de 3 campos (pressão + 2 cofunções da fonte) para 1, só a pressão. No mixed, o ganho é de 1 campo por unidade, e não 2. O motivo é que `TimeStep.checkpoint` copia em cada unidade todas as dependências do adjunto, inclusive as entradas que nunca mudam. Assim, cada unidade passa a copiar `source_density` e também `source_function`, que fica zerada no forward mas continua na forma para o adjunto implementado. No Revolve não se espera ganho (na reformulação ele foi nulo), porque a fonte não está nos checkpoints de reinício. O restante depende do item 3(a) (não recopiar entradas constantes), ou de tirar `source_function` do solver do forward.

Na reformulação (antes da reversão) o efeito foi o mesmo: tape completo 3209 → 1609, mixed s=16 188 → 173, Revolve s=16 inalterado.

Testes: `tests/on_one_core/test_point_source_amplitude.py`.
- O termo assemblado reproduz $s\,q$ (< 1e-12).
- Depois do 1º passo, nenhum passo do tape tem cofunção de saída. Antes eram 2 por passo, conferido no HEAD.
- O adjunto implementado ignora a amplitude deixada pelo forward. O teste falha se o reset em `backward_wave_propagator` for removido.

`tests/on_one_core/test_automated_adjoint_recomputation.py` (antes `test_acceleration_central_difference.py`, sem o teste específico da aceleração) verifica o gradiente com poucos checkpoints contra o tape completo, a reavaliação em outro modelo e o teste de Taylor com NRBC e recomputação.

### 1.9 Bug pré-existente: reavaliação do tape checkpointado no `main`

`test_reevaluation_after_velocity_change` grava o tape em $c$, chama `rf(c_novo)` e `rf.derivative()`, e compara com uma gravação nova em `c_novo` (NRBC em todos os lados, malha 0.2, `dt=0.004`, `T=0.8`):

| | J | gradiente |
|---|---|---|
| sem checkpointing | 0.0 | 0.0 |
| mixed, 5 snapshots | 0.0 | **2.1e-3** |

- O resultado é idêntico no `main` puro (`d5458bef`) e com a fonte por amplitude, então não foi introduzido aqui.
- Com a formulação por aceleração o teste passava (< 1e-10).
- `Inversion._run_fwi_tao` otimiza sobre `reduced_functional`, então a FWI com adjunto automático e `snapshots` usa esse gradiente a partir da 2ª iteração.
- O teste ficou como `xfail(strict=True)` até a correção.

Suspeitas, não verificadas:
- O solve para $u^{n+1}$ depende de $c$ tanto na matriz (`constant_jacobian=True`) quanto no lado direito ($M(c)(2u^n-u^{n-1})/\Delta t^2$).
- Algum checkpoint que depende de $c$ pode estar sendo restaurado do modelo antigo no reverso.

---

## 2. O que falta fazer

1. **Varredura de tempo e RSS** com a máquina ociosa, um processo por vez:
   ```
   ./setup_trees.sh
   ./sweep.sh timing 1
   python analyse.py
   ```
2. ~~**Decidir sobre a reformulação**~~: decidido manter o solver do `main`.
2b. **Corrigir o bug da seção 1.9** (gradiente errado ao refazer o tape checkpointado em outro $c$). Isso afeta a FWI com adjunto automático e checkpointing. Prioridade alta, em PR separado.
3. **Passo 6 (reverso especializado):** o armazenamento genérico é de fato um gargalo (7 cópias por unidade de dependências do adjunto, contra 1 necessária). Mas um bloco spyro sozinho não resolve, porque o `CheckpointManager` do pyadjoint revisa `adjoint_dependencies` adicionando **todas** as dependências de todos os blocos do passo, sem consultar o bloco. Caminhos possíveis:
   - (a) no pyadjoint/Firedrake: não copiar de novo checkpoints delegados/compartilhados em `TimeStep.checkpoint`. Isso é genérico e beneficia o original.
   - (b) API de bloco para declarar quais dependências o adjunto realmente lê. Só então um bloco de passo que use $\delta J_n = -{z^n}^T \delta M(c)\,a^n$ guardaria apenas $a^n$.
   - (c) alternativa já existente: o adjunto implementado do spyro, que guarda $u$ por passo.
4. ~~**Fonte pontual**~~: feito (seção 1.8). Pendente: tirar `source_function` (zerada) das dependências do solver do forward, ou resolver o 3(a), para ganhar o segundo campo por unidade do mixed. A fonte via `apply_source` (sem VOM) e a da PML continuam como cofunção por passo.
5. **Registros observados no estado de reinício:** criar os `real_shot` sem que entrem retroativamente no `checkpointable_state` (por exemplo, pré-criá-los fora do laço ou como dados não anotados).
6. **Revisar a docstring de `AutomatedAdjoint`** sobre mixed vs Revolve, ou expor o Revolve no spyro com a ressalva do custo por unidade.
7. **Avaliar com NRBC/HABC e em malhas maiores** (a varredura usou só o caso sem absorção). `bench.py --abc nrbc` já suporta NRBC.
8. **Falhas pré-existentes** em `test_reference_model_abc.py` / `test_habc_tools.py[T-3]`: tratar em PR separado.
9. **Antes de um PR:** um humano revisa e assume cada mudança (AGENTS.md); declarar o uso de IA (Claude Code, Opus 5.5) para implementação, testes e medições; decidir se esta pasta de benchmark (3.3 MB, inclui `.npz`) entra no repositório.

## 3. Como reproduzir

Os comandos abaixo reproduzem as seções 1.4–1.7 e supõem que a árvore de trabalho tem a reformulação. Depois da reversão, faça antes `git checkout cc246b6`. Na árvore atual, `bench.py` com `PYTHONPATH` apontando para o repositório mede o solver do `main` com a fonte por amplitude (seção 1.8).

```
source ~/dev_code/venv-firedrake/bin/activate
cd benchmarks/adjoint_acceleration
./setup_trees.sh                  # trees/baseline (original) e trees/inplace
./sweep.sh probe 4                # bytes de checkpoint (paralelo ok)
./sweep.sh timing 1               # tempo e RSS (máquina ociosa)
python analyse.py                 # tabelas
```

`bench.py` também aceita `--taylor`, `--reeval` (nova avaliação em outro modelo de velocidade) e `--abc nrbc`. Ele mede bytes de checkpoint percorrendo `TimeStep._checkpoint` e os `_checkpoint` das block variables, contando cada buffer uma vez, e conta recomputações pelas ações `Forward` executadas após a gravação.
