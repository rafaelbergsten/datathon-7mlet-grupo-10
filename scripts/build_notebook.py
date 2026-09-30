"""Build the educational notebook from transparent, runnable cells."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def markdown(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


markdown("""
# Mensagens adaptativas | Datathon MLET

**Grupo 10 | 7MLET** · Rafael Elias Ferreira Bergsten (rafaeleliasf@gmail.com)

Este notebook cobre a EDA, a preparação da base, a comparação entre baseline e bandit, o Golden Set
e o registro no MLflow. Instruções de execução, arquitetura AWS e governança estão no README.

A base Kaggle calibra taxas reais de conversão, mas não diz como os clientes reagiriam a cada
uma das três mensagens. Essas respostas são simuladas, como explicado na seção 3.

Ambiente: Python 3.12 com `requirements-lock.txt`. As 30 execuções levam alguns minutos.
""")
markdown("""
## 1. Problema, ações e recompensa

O banco precisa escolher como apresentar uma oferta de depósito a prazo já aprovada.
As ações são três mensagens: **curta**, **explicação guiada** e **resumo de benefícios**.
O contexto é o tipo de contato, conhecido antes da escolha: `cellular` ou `telephone`.

A recompensa é 1 quando o cliente converte e 0 quando não converte.
Decisões sem resultado informado ficam pendentes na API.
""")
code("""
import inspect
import json
import os
import sys
import tempfile
from pathlib import Path

root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'config.json').is_file() and (p / 'banditlab').is_dir())
os.chdir(root)
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from IPython.display import Code, Image, Markdown, display

from banditlab.bandit import ACTIONS, CONTEXTS, ThompsonSampling
from banditlab.data import load_data, prepare_data
from banditlab.experiment import build_environment, run_experiments
from demo import golden_set

pd.set_option('display.max_columns', 12)
config = json.loads((root / 'config.json').read_text())
print('Python:', sys.version.split()[0])
print('Evaluation: semi-synthetic | Seeds:', config['repetitions'])
""")
markdown("""
## 2. Fonte, versão e preparação

[Bank Marketing no Kaggle](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing),
versão 1, arquivo `bank-additional-full.csv`. Fonte original:
[Moro, Rita e Cortez, UCI, DOI 10.24432/C5K306](https://doi.org/10.24432/C5K306), licença CC BY 4.0.

O pipeline confere o SHA-256 do arquivo, mantém a ordem cronológica, converte `y` em 0/1 e
fica só com seis colunas. `duration` é descartada porque só existe depois da ligação.
""")
code("""
if not (root / 'data/processed/bank_safe.csv').exists():
    prepare_data()
df = load_data()
manifest = json.loads((root / 'data/manifest.json').read_text())
display(pd.DataFrame([
    {'item': key, 'value': manifest[key]}
    for key in ['source', 'kaggle_version', 'rows', 'source_file', 'upstream_license', 'source_csv_sha256']
]))
display(pd.DataFrame({
    'dtype': df.dtypes.astype(str),
    'missing': df.isna().sum(),
    'unique_values': df.nunique(),
}))
assert 'duration' not in df.columns
assert df.converted.isin([0, 1]).all()
assert not df.isna().any().any()
""")
markdown("""
### Qualidade, categorias e redução de colunas

Em `poutcome`, `nonexistent` quer dizer que o cliente não participou de campanha anterior.
Valores ausentes ou alvo inválido interrompem o pipeline.
Linhas repetidas depois da redução de colunas foram mantidas, porque são contatos diferentes.
""")
code("""
quality = pd.DataFrame([
    {'metric': 'observations', 'value': len(df)},
    {'metric': 'conversion_rate', 'value': df.converted.mean()},
    {'metric': 'identical_minimized_rows_not_removed', 'value': df.duplicated().sum()},
])
display(quality)
display(df.groupby('poutcome', observed=True).agg(observations=('converted', 'size'), conversion=('converted', 'mean')))

counts = df.converted.value_counts().reindex([0, 1], fill_value=0)
fig, ax = plt.subplots(figsize=(7.5, 3.6), layout='constrained')
bars = ax.bar(['Não converteu', 'Converteu'], counts, color=['#8093A4', '#087E8B'])
ax.bar_label(bars, padding=4)
ax.set_title('Distribuição factual do alvo | Bank Marketing')
ax.set_ylabel('Registros históricos')
ax.set_ylim(0, counts.max() * 1.15)
ax.spines[['top', 'right']].set_visible(False)
(root / 'results').mkdir(exist_ok=True)
fig.savefig(root / 'results/eda_target.png', dpi=150)
display(Image(filename=str(root / 'results/eda_target.png')))
plt.close(fig)
""")
markdown("""
### Conversão histórica por contato e ao longo do tempo

O celular converte bem mais que o telefone fixo, e a taxa muda bastante entre os períodos.
São associações: parte da diferença pode vir do perfil de público e da época, e não só do canal.
O alvo é desbalanceado (11,3% de conversão), o que reforça usar conversão, e não acurácia, como métrica.
""")
code("""
display(df.groupby('contact', observed=True).agg(observations=('converted', 'size'), conversion=('converted', 'mean')))
display(df.groupby('month', observed=True).agg(observations=('converted', 'size'), conversion=('converted', 'mean')))

cut1 = int(len(df) * 0.6)
cut2 = int(len(df) * 0.8)
blocks = {'Primeiros 60%': df.iloc[:cut1], 'Próximos 20%': df.iloc[cut1:cut2], 'Últimos 20%': df.iloc[cut2:]}
temporal = pd.DataFrame([
    {'block': name, 'observations': len(block), 'conversion': block.converted.mean(),
     'cellular_share': block.contact.eq('cellular').mean()}
    for name, block in blocks.items()
])
display(temporal)
display(Markdown('**Leitura:** mudanças entre blocos reforçam a limitação de transportar taxas antigas para uma operação atual.'))
""")
markdown("""
## 3. Simulador

Os primeiros 60% da base calibram uma taxa de conversão por contexto, suavizada com Beta(1,1).
Cada mensagem multiplica essa taxa por um fator fixado no `config.json` antes dos experimentos.
O cenário principal tem metade do tráfego em cada contexto. O `observed_mix` usa a proporção de
contatos dos períodos mais recentes, e o `equal_arms` dá a mesma taxa às três mensagens.

Um piloto de 3.000 escolhas aleatórias serve de ponto de partida para todas as políticas.
Depois, cada execução avalia 20.000 decisões novas; as conversões do piloto não entram na conta.
""")
code("""
environment = build_environment(df, config)
display(pd.DataFrame(config['assumed_message_multipliers'], index=CONTEXTS, columns=ACTIONS).rename_axis('Hypothetical multipliers'))
display(pd.DataFrame(environment['probabilities'], index=CONTEXTS, columns=ACTIONS).rename_axis('Simulated conversion probability'))
display(pd.DataFrame([environment['split_rows']]))
print('Held-out factual labels used to generate rewards:', environment['test_labels_used_for_rewards'])
""")
markdown("""
## 4. Thompson Sampling

Cada contexto tem uma distribuição Beta por mensagem, começando em Beta(1,1) e atualizada com o piloto.
A cada decisão:

1. Sorteamos uma taxa de conversão de cada distribuição do contexto.
2. Escolhemos a mensagem com a maior taxa sorteada.
3. Observamos o resultado e atualizamos `alpha += reward` e `beta += 1 - reward`.

A classe abaixo é a mesma usada no experimento e na API.
Referência: [A Tutorial on Thompson Sampling, Russo et al.](https://arxiv.org/abs/1707.02038).
""")
code("""
display(Code(inspect.getsource(ThompsonSampling), language='python'))

example_policy = ThompsonSampling(seed=42)
action = example_policy.select(context=0)
example_policy.update(context=0, action=action, reward=1)
print('Example selected action:', ACTIONS[action])
display(pd.DataFrame(example_policy.means, index=CONTEXTS, columns=ACTIONS))
""")
markdown("""
## 5. Experimento e tracking

Comparamos a melhor mensagem única do piloto (baseline), a melhor mensagem por contexto,
Epsilon-Greedy com epsilon=0,10 e Thompson Sampling. O oráculo conhece as probabilidades
do simulador e mostra o teto possível.

As sementes vão de 42 a 71. Em cada uma, todas as políticas recebem os mesmos clientes e resultados
possíveis, mas cada uma só vê o resultado da mensagem que escolheu. A API usa a política da semente 42.

Esta célula refaz tudo e cria os runs no MLflow local.
""")
code("""
summary, comparisons = run_experiments()
display(summary.loc[summary.scenario.eq('contextual'), ['policy', 'conversion', 'ci_low', 'ci_high', 'expected_regret']])
display(comparisons)
""")
markdown("""
### Métricas e intervalos

Conversão é `conversões / decisões`, e o ganho é a diferença em pontos percentuais.
O arrependimento soma, a cada decisão, quanto se perdeu em relação à melhor mensagem possível.
Os intervalos de 95% vêm de bootstrap (5.000 reamostragens) sobre as médias das 30 execuções,
usando diferenças pareadas para o ganho.
""")
code("""
display(Image(filename=str(root / 'results/comparison.png')))
display(Image(filename=str(root / 'results/learning.png')))
display(Image(filename=str(root / 'results/allocation.png')))

main = comparisons.query("scenario == 'contextual' and baseline == 'fixed'").iloc[0]
segmented = comparisons.query("scenario == 'contextual' and baseline == 'segmented'").iloc[0]
display(Markdown(
    f"**Resultado simulado:** ganho médio de {main.gain_pp:.3f} p.p. contra mensagem única "
    f"(IC 95% [{main.ci_low_pp:.3f}; {main.ci_high_pp:.3f}]). "
    f"Contra a segmentação fixa: {segmented.gain_pp:.3f} p.p. "
    "A política adaptativa não precisa vencer uma segmentação que já conhece bons braços em um ambiente estável."
))
""")
markdown("""
### Controle negativo e cenário com tráfego real

No `equal_arms` as três mensagens convertem igual, então nenhum método deveria ganhar,
e o resultado confirma isso. No `observed_mix` o ganho diminui, porque quase todo o tráfego é de celular.
""")
code("""
display(comparisons.loc[comparisons.scenario.ne('contextual')])
metrics = pd.read_csv(root / 'results/metrics.csv')
assert metrics.loc[metrics.scenario.eq('equal_arms'), 'expected_regret'].eq(0).all()
assert len(metrics) == 3 * 5 * config['repetitions']
print('All scenarios and seeds are included:', len(metrics), 'policy evaluations')
""")
markdown("""
## 6. Golden Set: cinco clientes fictícios

Dois clientes recebem mensagem e três não recebem, por inelegibilidade, falta de permissão ou contexto desconhecido.
Usamos o modo `greedy` para o resultado ser reproduzível; no modo `sample` a API às vezes explora outra mensagem.
""")
code("""
golden = golden_set()
(root / 'results/golden_set.json').write_text(json.dumps(golden, indent=2, ensure_ascii=False) + '\\n')
golden_table = pd.DataFrame(golden)[['case', 'contact', 'eligible', 'consent', 'actual_action', 'actual_reason', 'passed', 'rationale']]
display(golden_table)
assert golden_table.passed.all()
""")
markdown("""
## 7. API na prática

Aqui chamamos os endpoints da API com um banco temporário, sem afetar o estado da API local.
O feedback atualiza só a mensagem daquela decisão, e reenviar o mesmo resultado não conta duas vezes.
Para testar pela interface, rode `uvicorn banditlab.api:app --host 127.0.0.1 --port 8000` e abra `/docs`.
""")
code("""
from fastapi.testclient import TestClient

from banditlab.api import create_app

with tempfile.TemporaryDirectory() as directory:
    with TestClient(create_app(Path(directory) / 'notebook.sqlite')) as client:
        recommendation = client.post('/recommend', json={
            'contact': 'cellular', 'eligible': True, 'consent': True, 'mode': 'greedy'
        }).json()
        display(recommendation)
        event = {'decision_id': recommendation['decision_id'], 'reward': 1}
        display(client.post('/feedback', json=event).json())
        display(client.post('/feedback', json=event).json())
        display(client.get('/metrics').json())
        blocked = client.post('/recommend', json={
            'contact': 'telephone', 'eligible': True, 'consent': False
        }).json()
        assert blocked['action'] is None
        display(blocked)
""")
markdown("""
## 8. MLflow

O run pai guarda parâmetros, hashes, ganhos, código-fonte e a política; os 450 runs filhos guardam
uma combinação de cenário, política e semente cada. O banco é o `mlflow.db`, com artefatos em `mlartifacts/`.
""")
code("""
tracking = json.loads((root / 'results/tracking.json').read_text())
mlflow.set_tracking_uri(f"sqlite:///{root / 'mlflow.db'}")
client = mlflow.tracking.MlflowClient()
parent = client.get_run(tracking['parent_run_id'])
display(pd.DataFrame([{'parent_run_status': parent.info.status, 'child_runs': len(tracking['child_runs']),
                       'evaluation_kind': parent.data.tags['evaluation'],
                       'simulation_version': parent.data.params['simulation_version']}]))
assert parent.info.status == 'FINISHED'
assert len(tracking['child_runs']) == 450
display(pd.DataFrame([{'parameter': k, 'value': v} for k, v in parent.data.params.items()]))
""")
markdown("""
## 9. Conclusão

O pipeline baixa e prepara a base, compara políticas de forma reproduzível, registra tudo no MLflow
e serve recomendações com feedback. O Golden Set cobre os casos de elegibilidade, permissão e contexto desconhecido.

No simulador, escolher a mensagem por contexto converte bem mais que uma mensagem única, e o Thompson Sampling
chega perto do máximo possível. A regra fixa por contexto empata com ele num cenário estável.
As limitações estão na seção 8 do README.
""")

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata = {
    "kernelspec": {"display_name": "Python 3 (Datathon)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}
target = ROOT / "notebooks/01_datathon.ipynb"
target.parent.mkdir(exist_ok=True)
nbf.write(notebook, target)
print(target)
