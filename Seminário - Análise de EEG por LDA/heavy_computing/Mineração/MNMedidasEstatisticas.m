function features = MNMedidasEstatisticas(EEG_epochs)

[S, C, N, E, T] = size(EEG_epochs);

% Inicializar arrays
features.mean = zeros(S,C,E,T);
features.var  = zeros(S,C,E,T);
features.std  = zeros(S,C,E,T);
features.ptp  = zeros(S,C,E,T);
features.rms  = zeros(S,C,E,T);
features.kurt = zeros(S,C,E,T);
features.skew = zeros(S,C,E,T);

% Loop para cada dimensão
for s = 1:S
    for ch = 1:C
        for tr = 1:T
            for e = 1:E
                epoch = squeeze(EEG_epochs(s,ch,:,e,tr));
                features.mean(s,ch,e,tr) = mean(epoch);
                features.var(s,ch,e,tr)  = var(epoch);
                features.std(s,ch,e,tr)  = std(epoch);
                features.ptp(s,ch,e,tr)  = max(epoch) - min(epoch);
                features.rms(s,ch,e,tr)  = rms(epoch);
                features.kurt(s,ch,e,tr) = kurtosis(epoch);
                features.skew(s,ch,e,tr) = skewness(epoch);
            end
        end
    end
end

end