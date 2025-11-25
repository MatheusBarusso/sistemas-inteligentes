function corr_mats = CFCorrelacao(EEG_epochs)
    % Entrada esperada: [Sujeito x Canal x Amostras x Epoca x Trial]
    [nSuj, nCanais, nAmostras, nEpocas, nTrials] = size(EEG_epochs);

    % A saída deve ser [Sujeito, Trial, Epoca]
    corr_mats = cell(nSuj, nTrials, nEpocas);

    for s = 1:nSuj
        fprintf('Correlação: Processando Sujeito %d/%d...\n', s, nSuj);
        for t = 1:nTrials
            for e = 1:nEpocas
                % Extrai o sinal [Canais x Amostras]
                % A ordem (s, :, :, e, t) é crucial para pegar a época correta
                sinal = squeeze(EEG_epochs(s, :, :, e, t));
                
                % Calcula correlação (retorna [Canais x Canais])
                corr_mats{s, t, e} = corrcoef(sinal'); 
            end
        end
    end
end