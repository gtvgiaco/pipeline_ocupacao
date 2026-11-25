### Esse é um projeto avaliativo do Modulo 3 (Integração de IA com dados) do curso de Análise de Dados do SCTEC.

#### Aluno: Gustavo Giacomini


Atuaremos como um Cientista de Dados para estruturar um Pipeline Preditivo completo aplicado à Automação Predial e Eficiência Energética na Indústria 4.0.

#### 1. O Problema: 

Um complexo industrial e tecnológico equipado com sensores ambientais IoT necessita prever em tempo real se determinadas salas técnicas e escritórios operacionais estão ocupados ou vazios, visando automatizar o desligamento inteligente de sistemas de ar-condicionado (HVAC) e iluminação para reduzir custos energéticos. A variável alvo do projeto é binária: Occupancy = 1 (ambiente ocupado / ativo) e Occupancy = 0 (ambiente desocupado / inativo).

#### Base de Dados: Occupancy Detection Dataset (UCI Machine Learning Repository / Kaggle), composta pelas seguintes variáveis ambientais:
- Temperature: Temperatura do ambiente em graus Celsius (°C).
- Humidity: Umidade relativa do ar (%).
- Light: Nível de luminosidade medido pelo sensor (Lux).
- CO2: Concentração de dióxido de carbono no ar (ppm).
- HumidityRatio: Taxa derivada de umidade específica e temperatura.
- Occupancy (Variável Alvo): Indicador binário de presença (0 = Desocupado, 1 = Ocupado).

#### Link para download dos dados: 
"https://archive.ics.uci.edu/ml/machine-learning-databases/00357/occupancy_data.zip"

### 2. Requisitos da Aplicação

#### Fase 1: Análise Exploratória (EDA)

- Apresentaremos as dimensões do dataset (número de linhas e colunas), os tipos de dados das variáveis (.info() / .dtypes) e o resumo estatístico descritivo das colunas numéricas via método .describe(). (Nota: Caso o arquivo possua coluna de data/hora como date, avalie seu formato ou descarte-a antes da modelagem numérica).
- Serão plotados, no mínimo, 3 gráficos analíticos bem fundamentados utilizando bibliotecas como Matplotlib ou Seaborn (ex.: histogramas de distribuição das variáveis preditoras, gráfico de barras verificando a proporção das classes da variável alvo Occupancy e um mapa de calor com a correlação de Pearson entre as variáveis).
- Será inserida uma célula de texto analisando os valores numéricos e os padrões identificados nos gráficos (como a relação entre luminosidade, CO2 e ocupação), explicitando como eles direcionam a estratégia de modelagem.


#### Fase 2: Limpeza e Tratamento de Dados (Data Prep)
- Exclusão de linhas duplicadas.
- Verificação da existência de dados ausentes (NaN). Caso existam (ou caso sejam simulados para validação do pipeline), aplique a imputação por Média ou Mediana, justificando textualmente a escolha com base na simetria e distribuição dos dados.
- Gere gráficos do tipo boxplot para identificar a presença de valores discrepantes (outliers) nas variáveis explicativas (Temperature, Humidity, Light, CO2, HumidityRatio).


#### Fase 3: Feature Engineering

- Criaremos uma nova coluna numérica por meio de operação matemática entre colunas existentes (garantindo que não haja valores nulos antes do cálculo).
- Sugestão (Conforto Térmico / Índice Ambiental): indice_calor_umidade = Temperature * Humidity ou razao_co2_luz = CO2 / (Light + 1). 
(É permitida a criação de outra combinação matemática relevante ao contexto ambiental, desde que explicada no vídeo e documentada no notebook).


#### Fase 4: Divisão e Balanceamento dos Dados

- Separe as variáveis preditoras (X) da variável alvo (y = Occupancy).  
- Divida os dados em conjunto de treino (80%) e teste (20%) utilizando train_test_split com o parâmetro stratify=y e uma semente fixa (random_state=42) para garantir reprodutibilidade e manter a mesma proporção das classes em ambos os subconjuntos.
- Justifique textualmente no notebook a importância de realizar essa divisão antes de qualquer transformação estatística de escala para evitar o vazamento de dados (Data Leakage).


#### Fase 5: Escalonamento de Variáveis (StandardScaler)
- Aplique o StandardScaler nas variáveis numéricas destinadas ao modelo KNN, utilizando obrigatoriamente fit_transform() apenas nos dados de treino e transform() nos dados de teste.
- Mantenha os dados destinados à Árvore de Decisão sem escalonamento, justificando textualmente no código o motivo pelo qual algoritmos baseados em árvores são imunes à diferença de escala entre os sensores (ex.: Lux variando de 0 a >1000 vs. Temperatura variando de 19 a 24). 

#### Fase 6: Ajuste de Parâmetros e Combate ao Overfitting
- No KNN: Treine o modelo variando o hiperparâmetro n_neighbors (K) por, no mínimo, 3 valores ímpares (ex.: K = 3, 5, 7) e registre a acurácia obtida tanto no treino quanto no teste.
- Na Árvore de Decisão: Treine o modelo variando o hiperparâmetro max_depth por, no mínimo, 3 limites (ex.: 3, 5 e None) e registre a acurácia obtida no treino e no teste.
Insira uma célula de texto identificando em quais configurações ocorreu overfitting (alta acurácia no treino acompanhada de queda no teste) e qual configuração garantiu maior estabilidade.


### Fase 7: Avaliação da Acurácia e Veredito Final
- Calcule e exiba a acurácia final do melhor modelo KNN e da melhor Árvore de Decisão utilizando exclusivamente os dados de teste.
- Compare os resultados numéricos e escreva uma conclusão técnica justificando qual modelo apresentou desempenho superior e deve ser adotado pelo sistema de automação predial.

