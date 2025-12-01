1º Passo -> Carregamento e processamento dos dados

2º Passo -> Método do cotovelo

3º Passo -> Aplicar K-Means

4º Passo -> Análise individual dos clusters

***

O programa irá carregar os dados e gerar um gráfico para que o 'k' seja escolhido, isso é, o número de clusters seja definido. Isso é feito por meio da análise do 'cotovelo' formado no gráfico, onde a queda que vai diminuindo conforme k aumenta passa a ser bem menos abrupta. Esse então é o k selecionado e que deve ser inserido quando requisitado pelo programa.

Após isso o programa gera as informações separadas dos clusters para uma análise. Com o gráfico permitindo a interpretação de um k = 3 os resultados são os seguintes:

Cluster 0:

 - Nº De Alunos: 172 (25,2%)

 - Coeficiente: 
    - Média: 0.62
    - Mediana: 0.63
    - Desvio Padrão: 0.15

- Nota ENEM:
    - Média: 636.12
    - Mediana: 635.93
    - Desvio Padrão: 54.53

- Distribuição por Sexo:
    - Masculino: 150 (87.2%)
    - Feminino: 22 (12.8%)

- Distribuição por Escola:
    - Pública: 172 (100%)



Cluster 1:

 - Nº De Alunos: 290 (42.5%)

 - Coeficiente: 
    - Média: 0.46
    - Mediana: 0.53
    - Desvio Padrão: 0.3

- Nota ENEM:
    - Média: 670.28
    - Mediana: 674.11
    - Desvio Padrão: 53.73

- Distribuição por Sexo:
    - Masculino: 263 (90.7%)
    - Feminino: 27 (9.3%)

- Distribuição por Escola:
    - Particular: 290 (100%)


Cluster 2:

 - Nº De Alunos: 221 (32.4%)

 - Coeficiente: 
    - Média: 0.14
    - Mediana: 0.07
    - Desvio Padrão: 0.16

- Nota ENEM:
    - Média: 563.60
    - Mediana: 563.82
    - Desvio Padrão: 61.17

- Distribuição por Sexo:
    - Masculino: 263 (85.1%)
    - Feminino: 27 (14.9%)

- Distribuição por Escola:
    - Pública: 187 (84.6%)
    - Particular: 34 (15.4%)

***

Com isso podemos realizar uma análise crítica:

As diferenças apresentadas mostram a desigualdade educacional no Brasil, fica aparente que conforme a maior porcentagem de alunos provenientes de escolas particulares maior a média de nota do Enem e (de certa forma) consequentemente a maior média de coeficiente de desempenho acadêmico. Também com os dados fica aparente que as diferenças em notas de Enem e coeficiente estão muito mais atreladas a condição social (e portanto acesso a educação particular, usualmente de maior qualidade) do que a gênero, pois mesmo em grupos com diferentes porcentagens de estudantes masculinos ainda existe um desnível nas notas.

Porém, os clusters mostram agrupamentos determinísticos e isso não quer dizer que um aluno fazer parte de um grupo específico garantira o desempenho determinado para esse grupo (seja ele ruim ou bom).
