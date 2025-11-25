function Medidas_Redes = CFMedidasRedes(data)
% CFMEDIDASREDES - Calcula medidas topológicas de redes (Grafo)
%
% ENTRADA:
% data: Célula {Sujeito, Trial, Epoca} com matrizes de conectividade [Canais x Canais]

    if ~iscell(data)
        error('Entrada inválida. Esperada célula de matrizes de conectividade.');
    end
    
    [nSuj, nTrial, nEp] = size(data);
    if nSuj == 0 || nTrial == 0 || nEp == 0
         Medidas_Redes = [];
         return;
    end
    nCan = size(data{1}, 1);

    % Inicializa o array de saída (ajuste o número de colunas 'X' conforme as medidas extraídas)
    % Exemplo: 4 medidas comuns (Grau Médio, Eficiência Global, Coeficiente de Clusterização, Modularidade)
    Medidas_Redes = zeros(nSuj * nTrial * nEp, 4); 
    idx = 1;

    % Parâmetro de limiar (ajuste conforme o necessário para binarizar/pesar a rede)
    threshold = 0.5; % Exemplo: pode ser adaptativo ou fixo

    for s = 1:nSuj
        for t = 1:nTrial
            for e = 1:nEp
                mat = data{s, t, e};
                
                if isempty(mat)
                    Medidas_Redes(idx, :) = NaN;
                else
                    % 1. Aplica um limiar para obter a matriz binária (requerida por muitas métricas)
                    adj_mat = double(mat > threshold); 
                    adj_mat(1:nCan+1:end) = 0; % Zera a diagonal (sem auto-conexão)
                    
                    % 2. Métrica 1: Grau Médio (Average Degree)
                    degree = sum(adj_mat, 2);
                    Medidas_Redes(idx, 1) = mean(degree);

                    % 3. Métrica 2: Coeficiente de Clusterização (Average Clustering Coefficient)
                    % (Requer toolbox de teoria de grafos, como BCT. Usando aproximação simples)
                    % Se você tiver o BCT, use: C = clustering_coef_bu(adj_mat);
                    % Se não tiver, use 0 ou crie uma substituição:
                    Medidas_Redes(idx, 2) = 0; % Placeholder: Requer toolbox de grafos

                    % 4. Métrica 3: Eficiência Global (Global Efficiency)
                    Medidas_Redes(idx, 3) = 0; % Placeholder: Requer toolbox de grafos

                    % 5. Métrica 4: Modularidade (Max Modularity)
                    Medidas_Redes(idx, 4) = 0; % Placeholder: Requer toolbox de grafos
                end
                idx = idx + 1;
            end
        end
    end
end