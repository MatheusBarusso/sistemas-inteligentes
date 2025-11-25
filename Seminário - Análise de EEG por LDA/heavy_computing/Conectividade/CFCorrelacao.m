function corr_mats = CFCorrelacao(EEG_data)
    % Dimensões dos dados
    nSuj = size(EEG_data, 1);
    nTrials = size(EEG_data, 2);
    nEpocas = size(EEG_data, 3);
    nCanais = size(EEG_data, 4);

    corr_mats = cell(nSuj, nTrials, nEpocas);

    for s = 1:nSuj
        fprintf('Processando Sujeito %d/%d...\n', s, nSuj);

        for t = 1:nTrials
            for e = 1:nEpocas

                % Dados: [Canais × Amostras]
                sinal = squeeze(EEG_data(s, t, e, :, :));   

                % Matriz de correlação entre canais
                corrMat = corrcoef(sinal');   % transposto para [Amostras × Canais]

                % Guardar
                corr_mats{s, t, e} = corrMat;
            end
        end
    end

    fprintf('Finalizado: Matrizes de correlação geradas!\n');
end