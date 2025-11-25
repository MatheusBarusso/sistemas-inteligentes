function bandpower_features = MNPotenciaBandas(EEG_epochs, fs, chan_names, plot_flag)

% Definir bandas
bands = struct('delta',[0.5 4], 'theta',[4 8], 'alpha',[8 13], 'beta',[13 30], 'gamma',[30 45]);

[S, C, N, E, T] = size(EEG_epochs);

% Inicializar arrays
bandpower_features.delta = zeros(S,C,E,T);
bandpower_features.theta = zeros(S,C,E,T);
bandpower_features.alpha = zeros(S,C,E,T);
bandpower_features.beta  = zeros(S,C,E,T);
bandpower_features.gamma = zeros(S,C,E,T);

% Calcular bandpower
for s = 1:S
    for ch = 1:C
        for tr = 1:T
            for e = 1:E
                epoch = squeeze(EEG_epochs(s,ch,:,e,tr));
                
                bandpower_features.delta(s,ch,e,tr) = bandpower(epoch, fs, bands.delta);
                bandpower_features.theta(s,ch,e,tr) = bandpower(epoch, fs, bands.theta);
                bandpower_features.alpha(s,ch,e,tr) = bandpower(epoch, fs, bands.alpha);
                bandpower_features.beta(s,ch,e,tr)  = bandpower(epoch, fs, bands.beta);
                bandpower_features.gamma(s,ch,e,tr) = bandpower(epoch, fs, bands.gamma);
            end
        end
    end
end

% Plotar média por canal
if plot_flag
    figure;
    for ch = 1:C
        % Média sobre sujeitos, epochs e trials
        mean_delta = mean(bandpower_features.delta(:,ch,:,:), [1 3 4]);
        mean_theta = mean(bandpower_features.theta(:,ch,:,:), [1 3 4]);
        mean_alpha = mean(bandpower_features.alpha(:,ch,:,:), [1 3 4]);
        mean_beta  = mean(bandpower_features.beta(:,ch,:,:),  [1 3 4]);
        mean_gamma = mean(bandpower_features.gamma(:,ch,:,:), [1 3 4]);
        
        subplot(ceil(C/4),4,ch);
        bar([mean_delta mean_theta mean_alpha mean_beta mean_gamma]);
        set(gca,'XTickLabel',{'δ','θ','α','β','γ'});
        ylabel('Power (\muV^2)');
        title(chan_names{ch}, 'Interpreter','none');
    end
    sgtitle('Bandpower médio por canal');
end

end