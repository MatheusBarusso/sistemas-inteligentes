#### No exemplo de Multilayer Perceptron (MLP) para resolver o problema do XOR (ou exclusivo) a função "train" realiza uma busca aleatória, uma espécie de força bruta utilizando valores aleatórios. A busca aleatória não é um dos métodos mais adequados para a tarefa, porém exibe bom desempenho neste problema graças a uma escolha inteligente da função de ativação neural: a função heaviside, que transforma o espaço de estados interno em valores booleanos. Essa escolha é dependente do problema e apresentaria desempenho baixo em outros problemas.

#### Remova a linha 27 (return heaviside...) e descomente a linha seguinte (return relu...). Agora sua rede neural estará utilizando a função de ativação Unidade Linear Retificada (ReLU). Tente treinar a rede e verá que o programa "trava". Isso acontece, pois a busca aleatória dificilmente encontrará uma solução num espaço de busca que agora é contínuo e, portanto, muito maior.

#### Modifique a função "train" para utilizar uma busca global. É apresentado um exemplo em "basinhopping.py". A biblioteca scipy disponibiliza outros algoritmos de otimização global, tente utilizá-los.


#### OBS: Não realize busca local! É necessário encontrar um mínimo global da função de perdas para que o problema seja adequadamente resolvido, isto é, encontrado o mapeamento da função XOR.

***

- No arquivo ``` perceptron_nonlinear_train_differential_evolution.py ``` o método train foi alterado para um otimizador global puro. Inicialmente foi utilizado basinhopping, porém como esse método utiliza um otimizador local ele foi substituído por um inteiramente global.

- Quando ``` differential_evolution(...) ``` é chamada ela utiliza os parâmetros:
    - ``` loss_w ```, que chama a função ``` loss(x, y, w) ``` já fornecida.
    - ``` bounds[...] ```, que define o intervalo de busca para cada um dos 9 parâmetros (do vetor w), evitando exploração ilimitada.
    - ``` maxiter = 1000 ```, que define o número máximo de iterações da evolução diferencial.
    - ``` popsize = 15 ```, controla o tamanho da população NP, com 135 iniciais.
    - ``` tol = 1e-6 ```, que é o critério de tolerância de parada com base na melhor solução entre iterações.
    - ``` polish = False ```, desabilita o polimento final utilizando otimizador local, para manter a busca puramente global.
    - ``` seed = 37 ```, número escolhido arbitrariamente que serve de semente ao gerador de números aleatórios que é usado por ``` differential_evolution ```.

