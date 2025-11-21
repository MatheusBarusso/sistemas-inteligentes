%% Carregando os dados......................................................
clear; clc; close all

pathdataset = './dataset/';
nome_input = 'EEG_Data_BCI_IV_2a.mat';
nome_output = 'BCI_IV2a_preprocessed.mat';
load([pathdataset nome_input]);

%..........................................................................

%% Visualização -> Variáveis Sujeito, Época e Canal
Sujeito = 1;
Epoca = 1;
Canal = 1;
%Canais selecionados para função de plot de diferentes classes
canais = [8 12];

PlotCanal(EEG_data, chan_names, fs, Sujeito, Canal, Epoca);
PlotAll(EEG_data, chan_names, fs, Sujeito, Epoca);
PlotDiffCanal(EEG_data, labels, chan_names, fs, Sujeito, canais);
PlotTopologia(EEG_data, chan_names, chan_coords, fs, Sujeito, Epoca);
PlotPSDCanal(EEG_data, chan_names, fs, Sujeito, Epoca);

%..........................................................................

%% Pré-processamento
%Filtragem
EEG_data = PPFiltro(EEG_data,fs);

%Remover tendências
EEG_data = PPTendencia(EEG_data);

%Correção da linha base
tipo_baseline = 'prestim'; %Pré-estimulação
dur_base = 1;
EEG_data = PPLinhaBase(EEG_data, fs, tipo_baseline, dur_base);

%Detectar Artefatos
LimiarAmplitude = 100;
[bad_epoch, EEG_clean] = PPArtefatos(EEG_data, LimiarAmplitude);
fprintf('Total de trials ruins: %d\n', sum(bad_epoch(:)));

%Remoção de artefatos
%[EEG_clean, S_all, W_all, removed_all] = PPRemoveArtefatosICA(EEG_data);
%save([pathdataset 'EEG_Data_SemArtefatos.mat'],'EEG_clean','S_all','W_all','removed_all','-v7.3');

load([pathdataset 'EEG_Data_SemArtefatos.mat']); %-> Load p/ teste

EEG_data = EEG_clean; 
clear EEG_clean

%Reamostragem
fs2 = 128;
[EEG_data,fs] = PPReamostragem(EEG_data, fs, fs2);

%Normalização
MetodoNormalizacao = 'zscore'; %Para [0,1]: 'minmax'; Para Z-score: 'zscore'; Para Mediana/IQR:'robust' 
EEG_data = PPNormalizar(EEG_data, MetodoNormalizacao);

%Re-referenciação
TipoReRef = 'CAR'; %Para Média comum:'CAR',Sagital:'Cz', Mastoides: 'Mastoides', Laplace: 'Laplaciano'
EEG_data = PPReRe(EEG_data, chan_names, chan_coords, TipoReRef);

%Segmentação em épocas
window_sec = 1;      % 1 segundo por janela
overlap_sec = 0.5;   % 50% de sobreposição
[EEG_data,NumEpocas] = PPSegmentar(EEG_data, fs, window_sec, overlap_sec);

%..........................................................................

%% Mineração
%Estatísica
%% Seção de Funções
% Visualização -> Prefixo Plot.............................................
function PlotCanal(EEG_data,chan_names,fs,Sujeito,Canal,Epoca)
[~,~,NumAmo,~] = size(EEG_data);
sinal = squeeze(EEG_data(Sujeito,Canal, :,Epoca));
    t = (0:NumAmo-1) / fs;
figure;
plot(t,sinal,'k');
title(sprintf('Sujeito %d - Canal %s - Época %d',Sujeito,chan_names{Canal},Epoca));
xlabel('Tempo (s)');
ylabel('Amplitude (µV)');
grid on;
end

function PlotAll(EEG_data,chan_names,fs,Sujeito,Epoca)
[~,NumCan,NumAmo,~] = size(EEG_data);
sinal = squeeze(EEG_data(Sujeito,:,:,Epoca));
    t = (0:NumAmo-1) / fs;

figure;
for ch = 1:NumCan
    subplot(NumCan,1, ch);
    plot(t, sinal(ch,:), 'k');
    ylabel(chan_names{ch}, 'Interpreter','none');

    % Remove ticks do eixo y
    set(gca,'YTick',[]);
    

    % Remove ticks do eixo x para todos, exceto última linha
    [row, ~] = ind2sub([NumCan,NumAmo], ch);
    if row < NumAmo
        set(gca,'XTick',[]);
    end
end
% Limites do eixo x
xlim([t(1) t(end)]);
grid on;
xlabel('Tempo (s)');

end

function PlotDiffCanal(EEG_data, labels, chan_names, fs, Sujeito, canais_selecionados)
if nargin < 6
    canais_selecionados = 1:size(EEG_data,2); % todos os canais
end

classes = unique(labels(Sujeito,:)); % pega classes do sujeito
num_classes = length(classes);
figure('Name', sprintf('Sujeito %d - Comparação de Classes', Sujeito), 'NumberTitle','off');

for c = 1:num_classes
    cls = classes(c);

    % encontra os trials deste sujeito para a classe atual
    idx_trials = find(labels(Sujeito,:) == cls);
    if isempty(idx_trials)
        warning('Nenhum trial encontrado para Classe %d do Sujeito %d.', cls, Sujeito);
        continue;
    end

    % pega o primeiro trial válido
    trial = idx_trials(1);
    signal = squeeze(EEG_data(Sujeito, canais_selecionados, :, trial));
    t = (0:size(signal,2)-1)/fs;
    subplot(num_classes,1,c);
    plot(t, signal');
    title(sprintf('Classe %d - Trial %d', cls, trial));
    xlabel('Tempo (s)');
    ylabel('µV');
    grid on;
end
sgtitle(sprintf('Sujeito %d - Comparação entre Classes', Sujeito),'FontSize',14);
end

function PlotTopologia(EEG_data, chan_names, chan_coords, fs, Sujeito, Trial)
signal = squeeze(EEG_data(Sujeito, :, :, Trial)); % canais x amostras

% Calcula amplitude média por canal
amp_mean = mean(signal,2); % média ao longo do tempo

% Geração de grade para interpolação
x = chan_coords(:,1);
y = chan_coords(:,2);

grid_res = 100; % resolução do grid
[xq, yq] = meshgrid(linspace(min(x)-0.1,max(x)+0.1,grid_res), ...
                    linspace(min(y)-0.1,max(y)+0.1,grid_res));

% Interpolação dos dados
vq = griddata(x, y, amp_mean, xq, yq, 'cubic');

figure('Name', sprintf('Topoplot - Sujeito %d Trial %d', Sujeito, Trial), 'NumberTitle','off');
contourf(xq, yq, vq, 50, 'LineColor', 'none'); % mapa contínuo
colorbar;
colormap jet;
axis equal;
hold on;
scatter(x, y, 100, 'k', 'filled');
text(x+0.01, y+0.01, chan_names, 'FontSize', 10, 'Interpreter','none');
title(sprintf('Topoplot 2D - Sujeito %d - Trial %d', Sujeito, Trial));
axis off;
hold off;
end

function PlotPSDCanal(EEG_data, chan_names, fs, Sujeito, Trial)

[nSubjects, nChannels, nSamples, nTrials] = size(EEG_data);

if Sujeito < 1 || Sujeito > nSubjects || Trial < 1 || Trial > nTrials
    error('Sujeito ou Trial fora do intervalo.');
end

% Extrair sinal do trial
sig = squeeze(EEG_data(Sujeito, :, :, Trial)); % nChannels x nSamples

% Parâmetros PSD
window = 256;       % tamanho da janela (ajustável)
noverlap = 128;     % sobreposição
nfft = 512;         % número de pontos FFT

figure('Name',sprintf('PSD - Suj %d Trial %d', Sujeito, Trial), 'NumberTitle','off');

for ch = 1:nChannels
    subplot(ceil(nChannels/4),4,ch);
    [Pxx,F] = pwelch(sig(ch,:), window, noverlap, nfft, fs);
    plot(F,10*log10(Pxx),'k','LineWidth',1.2); % PSD em dB/Hz
    xlim([0 40]);   % exibir até 40 Hz (Delta a Beta)
    xlabel('Freq (Hz)');
    ylabel('PSD (dB/Hz)');
    title(chan_names{ch}, 'Interpreter','none', 'FontSize',9);
    grid on;
end

sgtitle(sprintf('PSD - Sujeito %d, Trial %d', Sujeito, Trial), 'FontSize',14);

end
%..........................................................................

% Pré-Processamento -> Prefixo PP..........................................
function EEG_data = PPFiltro(EEG_data,fs)
%Sem reamostragem
[NumSujeito, NumCan, ~, NumTrial] = size(EEG_data);

% --- Definição dos filtros (antes do loop) ---
f_low = 0.5; f_high = 30; order = 100;
b_bp = fir1(order, [f_low f_high]/(fs/2), 'bandpass');

f0 = 60; Q = 30;
[b_notch, a_notch] = iirnotch(f0/(fs/2), f0/(fs/2)/Q);

f_highpass = 0.5; order_hp = 50;
b_hp = fir1(order_hp, f_highpass/(fs/2), 'high');

% --- Loop diretamente modificando EEG_data ---
for s = 1:NumSujeito
    for ch = 1:NumCan
        for tr = 1:NumTrial

            % Extrair o sinal
            sinal = squeeze(EEG_data(s,ch,:,tr));

            % 1️⃣ Passa-banda 0.5–30 Hz
            sinal = filtfilt(b_bp, 1, sinal);

            % 2️⃣ Notch 60 Hz
            sinal = filtfilt(b_notch, a_notch, sinal);

            % 3️⃣ Passa-alta 0.5 Hz
            sinal = filtfilt(b_hp, 1, sinal);

            % 🔁 Atualizar diretamente na matriz original
            EEG_data(s,ch,:,tr) = sinal;
        end
    end
end


end

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

end

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

function [bad_epoch, EEG_clean] = PPArtefatos(EEG_data, ampThreshold)
[NumSujeito, ~, ~, NumTrial] = size(EEG_data);

bad_epoch = false(NumSujeito, NumTrial); % inicializa matriz de artefatos
EEG_clean = EEG_data;                     % cria cópia dos dados

for s = 1:NumSujeito
    for tr = 1:NumTrial
        sinal = squeeze(EEG_data(s,:,:,tr)); % C x N
        % cálculo peak-to-peak por canal
        p2p = max(sinal,[],2) - min(sinal,[],2);
        if any(p2p > ampThreshold)
            bad_epoch(s,tr) = true;           % marca trial como artefato
            EEG_clean(s,:,:,tr) = NaN;        % substitui por NaN
        end
    end
end

end

function [EEG_clean, S_all, W_all, removed_all] = PPRemoveArtefatosICA(EEG_data)

[NumSujeito, NumCan, NumAmostras, NumTrial] = size(EEG_data);

% Inicializações
EEG_clean = zeros(size(EEG_data));
S_all = cell(NumSujeito, NumTrial);
W_all = cell(NumSujeito, NumTrial);
removed_all = cell(NumSujeito, NumTrial);

fprintf('\n=== Iniciando ICA puro para remoção de artefatos ===\n');

for s = 1:NumSujeito
    fprintf('\nSujeito %d/%d', s, NumSujeito);
    
    for tr = 1:NumTrial
        fprintf('.');
        
        % --- 1. Extrair trial (C x N)
        X = squeeze(EEG_data(s, :, :, tr));
        
        % Verifica se o sinal é válido
        if any(isnan(X(:))) || all(X(:) == 0)
            EEG_clean(s,:,:,tr) = 0;
            continue;
        end

        % --- 2. Centralizar dados (zero mean por canal)
        X = X - mean(X,2);

        % --- 3. Whitening
        [C, N] = size(X);
        E = cov(X');
        [U, D] = eig(E);
        D_inv_sqrt = diag(1./sqrt(diag(D) + eps));  % evitar divisão por zero
        X_white = D_inv_sqrt * U' * X;

        % --- 4. ICA via maximização da kurtosis
        W = randn(C,C);
        maxIter = 500; tol = 1e-6;
        for iter = 1:maxIter
            W_old = W;
            Y = W*X_white;
            gY = Y.^3; % não-linearidade para super-gaussianas
            W = (gY*Y')/N - 3*eye(C)*W;
            % Ortogonalização
            [Uo, ~, Vo] = svd(W);
            W = Uo*Vo';
            % Convergência
            if max(abs(abs(diag(W*W_old'))-1)) < tol
                break;
            end
        end

        % --- 5. Componentes independentes
        S = W * X_white;

        % --- 6. Detecção automática de artefatos (desvio padrão alto)
        std_IC = std(S,0,2);
        threshold = mean(std_IC) + 2*std(std_IC);
        removed_components = find(std_IC > threshold);

        % --- 7. Zerar componentes artefatuais
        S_clean = S;
        S_clean(removed_components,:) = 0;

        % --- 8. Reconstrução do EEG limpo
        EEG_clean(s,:,:,tr) = pinv(W) * S_clean;

        % --- 9. Armazenar resultados
        S_all{s,tr} = S;
        W_all{s,tr} = W;
        removed_all{s,tr} = removed_components;
    end
end

fprintf('\n\n=== ICA puro concluído com sucesso ===\n');
end

function [EEG_data_fs2,fs2] = PPReamostragem(EEG_data, fs, fs2)
[NumSujeito,NumCan,NumAmo,NumTrial] = size(EEG_data);

      NumAmo2 = round(NumAmo * fs2/fs);
EEG_data_fs2 = zeros(NumSujeito, NumCan, NumAmo2,NumTrial);

for s = 1:NumSujeito
  for tr = 1:NumTrial
    for ch = 1:NumCan
     EEG_data_fs2(s,ch,:,tr) = resample(squeeze(EEG_data(s,ch,:,tr)), fs2, fs);
    end
  end
end

fs = fs2;

end

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

function EEG_ref = PPReRe(EEG_data, chan_names, chan_coords, method)
[S, C, N, T] = size(EEG_data);
EEG_ref = EEG_data; % inicialização

switch lower(method)
    case 'car'
        % Média Comum (CAR)
        mean_ch = mean(EEG_data, 2); % média por canal
        for ch = 1:C
            EEG_ref(:,ch,:,:) = EEG_data(:,ch,:,:) - mean_ch;
        end

    case 'cz'
        % Referência em Cz
        idxCz = find(strcmpi(chan_names,'Cz'),1);
        if isempty(idxCz)
            error('Canal Cz não encontrado.');
        end
        Cz_sig = EEG_data(:,idxCz,:,:);
        for ch = 1:C
            EEG_ref(:,ch,:,:) = EEG_data(:,ch,:,:) - Cz_sig;
        end

    case 'mastoides'
        % Referência A1/A2
        idxA1 = find(strcmpi(chan_names,'A1'),1);
        idxA2 = find(strcmpi(chan_names,'A2'),1);
        if isempty(idxA1) || isempty(idxA2)
            error('Canais A1/A2 não encontrados.');
        end
        ref_sig = (EEG_data(:,idxA1,:,:) + EEG_data(:,idxA2,:,:))/2;
        for ch = 1:C
            EEG_ref(:,ch,:,:) = EEG_data(:,ch,:,:) - ref_sig;
        end

    case 'laplaciano'
        % Laplaciano espacial (canais vizinhos)
        if isempty(chan_coords)
            error('Para Laplaciano, chan_coords deve ser fornecido.');
        end
        % Distância euclidiana entre canais
        dist = pdist2(chan_coords, chan_coords);
        for ch = 1:C
            % vizinhos próximos (dist < threshold)
            threshold = 0.2; % ajustar conforme escala
            neighbors = find(dist(ch,:) > 0 & dist(ch,:) <= threshold);
            if isempty(neighbors)
                continue;
            end
            mean_neighbors = mean(EEG_data(:,neighbors,:,:),2);
            EEG_ref(:,ch,:,:) = EEG_data(:,ch,:,:) - mean_neighbors;
        end

    otherwise
        error('Método desconhecido. Opções: CAR, Cz, Mastoides, Laplaciano.');
end

end

function [EEG_epochs,num_epochs] = PPSegmentar(EEG_data, fs, window_sec, overlap_sec)

[S, C, N, T] = size(EEG_data);
win_len = round(window_sec * fs);
step = round((window_sec - overlap_sec) * fs);
num_epochs = floor((N - win_len)/step) + 1;

EEG_epochs = zeros(S, C, win_len, num_epochs, T);

for s = 1:S
    for tr = 1:T
        start_idx = 1;
        for e = 1:num_epochs
            idx = start_idx:(start_idx + win_len - 1);
            EEG_epochs(s,:,:,e,tr) = EEG_data(s,:,idx,tr);
            start_idx = start_idx + step;
        end
    end
end

end

