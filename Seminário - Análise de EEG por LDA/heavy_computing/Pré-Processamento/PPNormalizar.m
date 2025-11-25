function EEG_data = PPNormalizar(EEG_data, metodo)

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

                    case 'minmax'
                        x_min = min(sinal);
                        x_max = max(sinal);
                        if x_max ~= x_min
                            sinal = (sinal - x_min) / (x_max - x_min);
                        else
                            sinal = zeros(size(sinal)); % Evita NaN
                        end

                    case 'robust'
                        med = median(sinal);
                        IQRv = iqr(sinal); % Interquartile Range
                        if IQRv ~= 0
                            sinal = (sinal - med) / IQRv;
                        else
                            sinal = sinal - med;
                        end

                    otherwise
                        error('Método inválido. Use: zscore, minmax ou robust');
                end

                EEG_data(s,ch,:,tr) = sinal;

            end
        end
    end

end