### Utilize busca em profundidade para implementar um oráculo que irá ajudar o jogador ao determinar qual a melhor jogada a partir de cada estado de um jogo da velha, através do exemplo "tictactoe.py". Entende-se por "estado" o conjunto de condições do problema em um dado momento. No caso do jogo da velha, consiste no tabuleiro e as posições jogadas até aquele momento. Temos ainda:
### - um estado inicial, o tabuleiro vazio
### - um ou mais estados objetivos, qualquer estado que resulte numa condição de vitória do jogador desejado
### - uma função teste de objetivo, determina se um estado é um estado objetivo ou não
### - uma ação, a transformada de um estado ao outro consistindo numa jogada
### - a função sucessora, mapeando um estado ao conjunto de próximos estados de acordo com as ações possíveis
### - o espaço de estados, consistindo no espaço de busca do problema, isto é, o conjunto de todos os estados alcançáveis a partir do estado inicial
### - solução, uma sequência ótima de ações que levam o estado inicial, ou atual, para um estado objetivo

### Não é obrigatório porém a estruturação de sua solução utilizando estes conceitos irá facilitar a implementação. Utilizando os conceitos da Atividade 1 - Jogo da Velha implemente não mais um oráculo mas sim um jogador virtual livre de entradas do usuário.
### Jogos de soma zero são aqueles em que os ganhos de um jogador igualam as perdas dos demais. O Jogo da Velha é um jogo adversarial de soma zero já que a vitória de um jogador implica na derrota do outro. O algoritmo minimax é aplicável nestas situações pois visa maximizar o menor ganho possível de uma jogada ou em outros termos, maximizar as jogadas de um jogador considerando que o outro tenta minimizá-las. Opcionalmente, pesquise e implemente o algoritmo minimax na sua busca em árvore.

***
1. A implementação do Oráculo (sugestão de jogadas) foi feito por meio de:
    - A função ``` TicTacToeOracle.get_empty_cells(grid) ``` que retorna a lista de tuplas das células sem valor (no caso, valor 0) para que ``` dfs ``` e ``` find_best_move ``` enumerem movimentos válidos.
    - A função ``` TicTacToeOracle.dfs(...) ```, responsável por:
        - Calcular o score por meio de evaluate.
            - Se score == 10, retorna 10 - depth (para priorizar vitórias mais rápidas)
            - Se score == -10, retorna -10 + depth (tentar adiar a derrota)
            - Caso ``` is_game_over ``` e condição de vitória não é encontrada, retorna 0 que sinaliza empate.
        - Se ``` is_maximizing ```, então inicializa best = -inf para as células vazias e simula jogada do player, utiliza recursão com ``` is_maximizing=False ```, restaura a célula e atualiza o best com o maior valor que foi encontrado.
        - Com isso, é implementado minimax por DFS com corte por profundidade.
    - A função ``` TicTacToeOracle.find_best_move(grid, player) ```, que itera sobre as células vazias simulando o movimento do jogador, após isso chama ``` dfs(..., is_maximizing = False) ``` e desfaz o movimento. Então guarda em ``` self ``` o movimento com maior ``` move_value ``` e retorna essa tupla (com linha e coluna).


2. Para a implementação do Bot (Jogador controlado pela máquina) o seguinte foi feito:
    - A função ``` TicTacToeOracle.init ``` inicializa o oráculo (similar ao que foi implementado em ``` tictactoe_oraculo.py ```) e cria a tupla ``` self.best_move ``` para armazenara melhor jogada encontrada.
    - A função ``` TicTacToeOracle.get_empty_cells(grid) ```, que identifica as jogadas possíveis no grid 3x3 do jogo e as enumera para a busca minimax.
    - A função ``` TicTacToeOracle.evaluate(grid, maximizing_player) ``` que retorna '+10' se o maximizing_player vence e '-10' se o adversário venceu. Em caso neutro retorna '0'. Isso é utilizado para determinar o minimax para calcular os scores.
    - A função ``` TicTacToeOracle.dfs(...) ```, explicada acima.
    - A ``` função TicTacToeOracle.fund_best_mode(grid, plater) ``` que itera pelas células vazias, simula as jogadas, chama ``` dfs(..., is_maximizing=False) ``` e restaura a célula, após isso mantém a melhor jogada em ``` self.best_move ```.