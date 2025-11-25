function EEG_data = PPTendencia(EEG_data)
[NumSujeito, NumCan, ~, NumTrial] = size(EEG_data);

for s = 1:NumSujeito
    for ch = 1:NumCan
        for tr = 1:NumTrial
            
            % Sinal atual
            sinal = squeeze(EEG_data(s,ch,:,tr));

            % Remover tendência linear (detrend central)
            sinal = detrend(sinal, 'linear');   % ou apenas detrend(sinal)

            % Atualiza sinal
            EEG_data(s,ch,:,tr) = sinal;
        end
    end
end