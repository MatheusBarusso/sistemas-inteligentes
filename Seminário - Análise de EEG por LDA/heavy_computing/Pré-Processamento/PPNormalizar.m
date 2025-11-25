function EEG_data = PPnormalizar(EEG_data, metodo)
    [NumSujeito, NumCan, ~, NumTrial] = size(EEG_data);

    for s = 1:NumSujeito
        for ch = 1:NumCan
            for tr = 1:NumTrial

                sinal = squeeze(EEG_data(s,ch,:,tr));

                switch lower(metodo)
                    case 'zscore'
                        mu = mean(sinal);
                        sigma = std(sinal);
                        if sigma ~= 0
                            sinal = (sinal - mu) / sigma;
                        else
                            sinal = sinal - mu; % Evita divisão por zero
                        end
                    otherwise
                        error('Método inválido. Use: zscore.');
                end

                EEG_data(s,ch,:,tr) = sinal;

            end
        end
    end

end