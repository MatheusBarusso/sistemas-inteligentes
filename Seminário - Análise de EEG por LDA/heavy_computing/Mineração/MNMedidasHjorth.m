function hjorth = MNMedidasHjorth(EEG_epochs)
[S, C, N, E, T] = size(EEG_epochs);

% Inicializar arrays
hjorth.activity   = zeros(S,C,E,T);
hjorth.mobility   = zeros(S,C,E,T);
hjorth.complexity = zeros(S,C,E,T);

for s = 1:S
    for ch = 1:C
        for tr = 1:T
            for e = 1:E
                epoch = squeeze(EEG_epochs(s,ch,:,e,tr));

                % Activity: variância do sinal
                var_x = var(epoch);
                hjorth.activity(s,ch,e,tr) = var_x;

                % Mobility: sqrt(var(x')/var(x))
                dx = diff(epoch);
                var_dx = var(dx);
                hjorth.mobility(s,ch,e,tr) = sqrt(var_dx / var_x);

                % Complexity: sqrt(var(dx')/var(dx)) / mobility
                ddx = diff(dx);
                var_ddx = var(ddx);
                hjorth.complexity(s,ch,e,tr) = sqrt(var_ddx / var_dx) / hjorth.mobility(s,ch,e,tr);
            end
        end
    end
end

end