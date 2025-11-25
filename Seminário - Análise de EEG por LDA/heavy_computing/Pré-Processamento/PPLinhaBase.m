function EEG_corr = PPLinhaBase(EEG_data, fs, tipo_baseline, dur_base)

[NumSujeito, NumCan, ~, NumTrial] = size(EEG_data);
EEG_corr = EEG_data;

for s = 1:NumSujeito
    for ch = 1:NumCan
        for tr = 1:NumTrial

            sinal = squeeze(EEG_data(s,ch,:,tr));

            if strcmp(tipo_baseline,'prestim')
                Nbase = dur_base * fs;
                base = mean(sinal(1:Nbase));  % pré-estímulo
            else
                base = mean(sinal);          % baseline global
            end

            EEG_corr(s,ch,:,tr) = sinal - base;

        end
    end
end

end
