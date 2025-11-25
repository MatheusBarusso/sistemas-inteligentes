function Medidas_Stats = CFMedidasEstatisticas(data)
% CFMEDIDASESTATISTICAS - Calcula medidas estatísticas (mean, var, std, etc.)
%
% ENTRADA:
% data: Dados em formato de célula {Sujeito, Trial, Epoca} com matrizes [Canais x Canais] (para conectividade)
%       ou matriz N-dimensional [S x C x N x E x T] (para dados EEG brutos)

    % Assume que a entrada é uma célula de matrizes de conectividade 
    % (formato usado no bloco de agregação PLV)
    if iscell(data)
        [nSuj, nTrial, nEp] = size(data);
        if nSuj == 0 || nTrial == 0 || nEp == 0
             warning('Célula de entrada vazia.');
             Medidas_Stats = [];
             return;
        end
        nCan = size(data{1}, 1); % Assumindo matrizes simétricas [C x C]
        
        % Inicializa o array de saída
        Medidas_Stats = zeros(nSuj * nTrial * nEp, 7); 
        idx = 1;

        for s = 1:nSuj
            for t = 1:nTrial
                for e = 1:nEp
                    mat = data{s, t, e};
                    if isempty(mat)
                        Medidas_Stats(idx, :) = NaN; % Evita erros com épocas ruins
                    else
                        % Converte a matriz de conectividade [C x C] em um vetor de features
                        % Usando apenas a parte triangular superior (excluindo a diagonal)
                        upper_tri = mat(triu(true(nCan), 1)); 

                        Medidas_Stats(idx, 1) = mean(upper_tri(:));
                        Medidas_Stats(idx, 2) = var(upper_tri(:));
                        Medidas_Stats(idx, 3) = std(upper_tri(:));
                        Medidas_Stats(idx, 4) = max(upper_tri(:)) - min(upper_tri(:)); % PTP
                        Medidas_Stats(idx, 5) = sqrt(mean(upper_tri(:).^2)); % RMS
                        Medidas_Stats(idx, 6) = kurtosis(upper_tri(:));
                        Medidas_Stats(idx, 7) = skewness(upper_tri(:));
                    end
                    idx = idx + 1;
                end
            end
        end
        
    else
        % Caso a função seja chamada com dados EEG [Suj x Canal x Amostras x Época x Trial]
        % (Mantido para compatibilidade, embora o foco atual seja conectividade)
        % Note: Esta parte deve refletir o código original que gerou Medidas_Stats/Medidas_hjorth
        % No contexto atual, esta função é chamada com CÉLULAS de conectividade.
        error('Formato de entrada não suportado para dados brutos EEG. Esperada célula de conectividade.');
    end
end