%% Carregando os dados.....................................................
% Statistics and Machine Learning Toolbox necessário para funcionamento
clear all; clc; close all

pathdataset = './dataset/';
nome_input = 'EEG_Data_BCI_IV_2a.mat';
nome_output = 'BCI_IV2a_preprocessed.mat';
load([pathdataset nome_input]);

%Carregar parte das funções -> Quando aninhadas estouravam a memória
%Prioridade para funções com maior complexidade computacional
baseDir = fileparts(matlab.desktop.editor.getActiveFilename);
funcDir = fullfile(baseDir, 'heavy_computing');
addpath(genpath(funcDir));
baseDir = pwd;
plv_dir_name = 'PLVs';
path_plv_results = fullfile(baseDir, plv_dir_name);

disp('Etapa de Carregamento Finalizada')
disp('-------------------------------------------------------------------')
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

disp('Etapa de visualização Finalizada');
disp('-------------------------------------------------------------------')
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

%Remoção de artefatos -> Descomentar as linhas abaixo se for processar
[EEG_clean, S_all, W_all, removed_all] = PPRemoveArtefatosICA(EEG_data);
save([pathdataset 'EEG_Data_SemArtefatos.mat'],'EEG_clean','S_all','W_all','removed_all','-v7.3');

%Comentar a linha abaixo se for rodar a remoção de artefatos
%load([pathdataset 'EEG_Data_SemArtefatos.mat']); %-> Load p/ teste

EEG_data = EEG_clean; 
clear EEG_clean

%Reamostragem
fs2 = 128;
[EEG_data,fs] = PPReamostragem(EEG_data, fs, fs2);

%Normalização
MetodoNormalizacao = 'zscore';
EEG_data = PPNormalizar(EEG_data, MetodoNormalizacao);

%Re-referenciação
TipoReRef = 'CAR'; 
EEG_data = PPReRe(EEG_data, chan_names, chan_coords, TipoReRef);

%Segmentação em épocas
window_sec = 1;      % 1 segundo por janela
overlap_sec = 0.5;   % 50% de sobreposição
[EEG_epochs, NumEpocas] = PPSegmentar(EEG_data, fs, window_sec, overlap_sec);

disp('Etapa de Pré-Processamento finalizada');
disp('-------------------------------------------------------------------')
%..........................................................................


%% Mineração
%Estatísica
Medidas_estatisticas = MNMedidasEstatisticas(EEG_epochs);

%Parâmetros de Hjorth
Medidas_hjorth = MNMedidasHjorth(EEG_epochs);

%Potência das bandas -> Descomentar linhas abaixo para rodar mineração
fs = 128;
plot_flag = 0;
band_features = MNPotenciaBandas(EEG_epochs, fs, chan_names, plot_flag);

%Comentar a linha abaixo se for rodar a mineração
%load([pathdataset 'Medidas_Bandas_Potencia.mat']); % -> Load p/ testes

disp('Etapa de Mineração Finalizada');
disp('-------------------------------------------------------------------')
%..........................................................................


%% Conectividade Funcional
%Correlação
correlacao = CFCorrelacao(EEG_epochs);

%% Phase-Locking-Value -> "Sincronização de sinais EEG"
%plv = CFPLV(EEG_data, fs);

[NumSujeito, ~, NumAmo, NumEpocas, NumTrial] = size(EEG_data);
for s = 1:NumSujeito
    fprintf('\n PLV: Processando Sujeito %d/%d (Batching)...\n', s, NumSujeito);
    dados_suj = squeeze(EEG_epochs(s, :, :, :, :));
    EEG_sujeito_atual = permute(dados_suj, [4 3 1 2]);
    plv_sujeito = CFPLV_SingleSubject(EEG_sujeito_atual, fs);
    nome_arquivo = fullfile(path_plv_results, sprintf('plv_sujeito_%d.mat', s));
    save(nome_arquivo, 'plv_sujeito', '-v7.3');
    clear plv_sujeito EEG_sujeito_atual dados_suj;
end

%% Medidas Estastísticas
Medidas_correlacao_estastisticas = CFMedidasEstatisticas(correlacao);
Medidas_correlacao_redes = CFMedidasRedes(correlacao);

%% Medidas Topológicas
% Medidas_PLV_estatisticas = CFMedidasEstatisticas(plv);
% Medidas_PLV_Redes = CFMedidasRedes(plv);

if ~exist('NumSujeito', 'var')
    NumSujeito = size(EEG_data, 1);
end

[Medidas_PLV_estatisticas, Medidas_PLV_Redes] = CFMedidasAgregadas_PLV(NumSujeito, path_plv_results);
clear PLV_agregado;

disp('Etapa de Conectividade Funcional Finalizada');
disp('-------------------------------------------------------------------')
%..........................................................................


%% Matriz de Medidas
%Montagem da Matriz
[X, feature_names] = MDMMontarMatriz(Medidas_estatisticas, Medidas_hjorth, band_features, ...
                                    Medidas_correlacao_estastisticas, Medidas_correlacao_redes, ...
                                    Medidas_PLV_estatisticas, Medidas_PLV_Redes);

%Normalização da Matriz com Medidas Concatenadas
desvio_padrao = std(X, [], 1, 'omitnan'); % Calcula desvio ignorando NaNs
cols_validas = desvio_padrao > 1e-6; % Mantém apenas colunas com variação
if sum(~cols_validas) > 0
    fprintf('Aviso: Removendo %d features constantes (sem variação).\n', sum(~cols_validas));
    X = X(:, cols_validas);
    feature_names = feature_names(cols_validas);
end

% 2. Tratamento de NaNs (Substituição pela média da coluna)
% Isso resolve o problema dos trials de artefato que viraram NaN
col_mean = mean(X, 1, 'omitnan');
for i = 1:size(X, 2)
    col = X(:, i);
    idx_nan = isnan(col);
    if any(idx_nan)
        col(idx_nan) = col_mean(i); % Substitui NaN pela média daquela feature
        X(:, i) = col;
    end
end

% Verificação final de segurança
X(isnan(X)) = 0; 
X(isinf(X)) = 0;
% ----------------------------------------
X_norm = MDMNormalizar(X, 'zscore');

disp('Etapa de Matriz de Medidas Finalizada');
disp('-------------------------------------------------------------------')
%..........................................................................


%% Redução de Dimensionalidade
%Reducao de Dimensionalidade
[X_pca, V, lambda, explained] = RDPCA(X_norm, 'var', 95);

%Plotar PCA
RDPlotar(explained, lambda);

disp('Etapa de Redução de Dimensionalidade Finalizada');
disp('-------------------------------------------------------------------')
%..........................................................................


%% Classificação
%Vetor das classes
[NumSujeito,NumCan,NumAmo,NumEpocas,NumTrial] = size(EEG_epochs);
Y = CLMontarVetor(NumSujeito, NumTrial, NumEpocas, labels);

%Validação Cruzada -> k-fold
folds = CLKFold(X, Y, 'KFold', [], 5);

%Classificar -> KFold ou LOSO // LDA
metodo_validacao = 'KFold';
classificador = 'LDA';
resultados = CLTreinamento(X, Y, metodo_validacao, classificador);

disp('Etapa de Classificação Finalizada');
disp('-------------------------------------------------------------------')
%..........................................................................


%% Análise de Desempenho
%Análise Final
Acuracia = resultados.acuracia;
MatrizConfusao = resultados.cm;
ADResultados(MatrizConfusao);
%..........................................................................

%%
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %


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
S
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
%..........................................................................


% Mineração -> Prefixo MN..................................................
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
%..........................................................................


function features = CFMedidasEstatisticas(conn_all, thresh)
if nargin < 2
    thresh = 0.5; % limiar padrão para grau
end

NumSuj = size(conn_all,1);
NumTr  = size(conn_all,2);
NumEp  = size(conn_all,3);

% Exemplo: armazenar em célula ou matriz
features = [];  % acumulador de vetores

for s = 1:NumSuj
    for tr = 1:NumTr
        for ep = 1:NumEp
            
            M = conn_all{s,tr,ep};   % matriz NxN de conectividade
            
            if isempty(M)
                continue;
            end
            
            %===== 1) MEDIDAS GLOBAIS =====
            mean_conn = mean(M(:));
            var_conn  = var(M(:));

            % Evitar log(0) na entropia
            M = M(:);
            M = M - min(M);         % remove valores negativos, se existirem
            M = M + eps;            % evita zeros

            % Normalização correta
            M = M / sum(M);

            % Entropia de Shannon
            entropy_M = -sum(M .* log(M));


            % --- Outras métricas com a matriz original M (sem alterar!)
            max_conn = max(M(:));
            %min_conn = min(M(:));

            %..Maxímo e mínimo
            max_conn = max(M(:));
            min_conn = min(M(:)); %..Está dando 0, problemático com PCA
            
            
            %===== 3) ORGANIZAR VETOR DE FEATURES =====
            feat_vec = [mean_conn, var_conn, entropy_M, ...
                        max_conn, min_conn];
            
            % Concatenar no dataset final
            features = [features; feat_vec];
            
        end
    end
end

end

function features = CFMedidasRedes(conn_all)

if nargin < 2
    thresh = 0.5; % limiar padrão para grau
end

NumSuj = size(conn_all,1);
NumTr  = size(conn_all,2);
NumEp  = size(conn_all,3);

% Exemplo: armazenar em célula ou matriz
features = [];  % acumulador de vetores

for s = 1:NumSuj
    for tr = 1:NumTr
        for ep = 1:NumEp
            
            M = conn_all{s,tr,ep};   % matriz NxN de conectividade
            
            if isempty(M)
                continue;
            end
            
            %===== 2) MEDIDAS LOCAIS =====
            degree   = sum(M > thresh, 2); % vetor Nx1
            strength = sum(M, 2);          % vetor Nx1
            
            mean_degree   = mean(degree);
            mean_strength = mean(strength);
            
            
            %===== 3) ORGANIZAR VETOR DE FEATURES =====
            feat_vec = [mean_degree, mean_strength];
            
            % Concatenar no dataset final
            features = [features; feat_vec];
            
        end
    end
end

end
%..........................................................................


% Matriz de Medidas -> Prefixo MDM.........................................

function X_norm = MDMNormalizar(X, method)

switch lower(method)
    case 'zscore'
        X_norm = (X - mean(X,1)) ./ std(X,[],1);
    case 'minmax'
        X_min = min(X,[],1);
        X_max = max(X,[],1);
        X_norm = (X - X_min) ./ (X_max - X_min);
    case 'robust'
        X_med = median(X,1);
        X_iqr = iqr(X,1);
        X_norm = (X - X_med) ./ X_iqr;
    otherwise
        error('Método de normalização inválido. Use "zscore", "minmax" ou "robust".');
end
end
%..........................................................................


% Redução de Dimensionalidade -> Prefixo RD................................
function [Z, V, lambda, explained] = RDPCA(X, varargin)
 % média móvel, para evitar valores com NaN
 for i = 1:size(X,2)
    col = X(:, i);
    col(isnan(col)) = mean(col(~isnan(col)));
    X(:, i) = col;
 end
    %Centralização dos dados
    X = double(X);                 % Garante formato double
    mu = mean(X, 1);               % Média de cada coluna (feature)
    Xc = X - mu;                   % Dados centralizados

    %Matriz de covariância
    C = cov(Xc);


    %Autovalores e Autovetores
    [V, D] = eig(C);               % Decomposição espectral
    [lambda, idx] = sort(diag(D), 'descend');  % Ordena autovalores
    V = V(:, idx);                 % Reorganiza autovetores

    %Variância explicada
    totalVar = sum(lambda);
    explained = cumsum(lambda) / totalVar * 100;

    %Seleção do número de componentes (k)
    k = length(lambda);  % Padrão: mantém tudo
    if ~isempty(varargin)
        if strcmpi(varargin{1}, 'k')
            k = varargin{2};
        elseif strcmpi(varargin{1}, 'var')
            k = find(explained >= varargin{2}, 1, 'first');
        end
    end

    V = V(:, 1:k);            % Mantém os k componentes principais
    Z = Xc * V;               % Projeção dos dados

end

function RDPlotar(explained, lambda)
    figure;
    
    %Plotar variância explicada
    subplot(1, 2, 1);
    bar(explained, 'FaceAlpha', 0.7);
    xlabel('Componente Principal');
    ylabel('Variância Explicada (%)');
    title('Variância Explicada por Componente');
    grid on;

    %Curve Scree Plot (autovalores)
    if nargin > 1
        subplot(1, 2, 2);
        plot(lambda, '-o', 'LineWidth', 1.5);
        xlabel('Componente Principal');
        ylabel('Autovalor');
        title('Scree Plot (Autovalores)');
        grid on;
    end

end
%..........................................................................


% Classificação -> Prefixo CL..............................................
function Y = CLMontarVetor(NumSujeito, NumTrial, NumEpocas, labels_trials)
    Y = []; % Vetor final de rótulos

    for s = 1:NumSujeito
        for tr = 1:NumTrial
            label_tr = labels_trials(s, tr);  % Classe do trial (1 a 4)
            % Repete o rótulo para cada época do trial
            Y = [Y; repmat(label_tr, NumEpocas, 1)];
        end
    end
end

function folds = CLKFold(X, Y, method, info, k)
    if nargin < 3
        error('É necessário especificar o método de validação.');
    end

    switch upper(method)

        case 'KFOLD'
            if nargin < 5
                error('Para KFold, forneça o número de folds k.');
            end
            cv = cvpartition(Y, 'KFold', k);
            for i = 1:k
                folds(i).trainIdx = training(cv, i);
                folds(i).testIdx = test(cv, i);
            end

        case 'LOSO'
            if nargin < 4 || isempty(info)
                error('Para LOSO, forneça o vetor com IDs dos sujeitos.');
            end
            uniqueSubjects = unique(info);
            for i = 1:length(uniqueSubjects)
                testSubj = uniqueSubjects(i);
                folds(i).testIdx  = (info == testSubj);
                folds(i).trainIdx = ~folds(i).testIdx;
            end

        otherwise
            error('Método não reconhecido. Use ''KFold'' ou ''LOSO''.');
    end
end

function resultados = CLTreinamento(X, Y, metodo_validacao, classificador)

    % Criação do modelo
    switch classificador
        case 'LDA'
            modelo = fitcdiscr(X, Y);
        case 'SVM'
            modelo = fitcsvm(X, Y, 'KernelFunction', 'rbf');
        case 'KNN'
            modelo = fitcknn(X, Y, 'NumNeighbors',5);
        otherwise
            error('Classificador não reconhecido.');
    end

    % Validação cruzada
    switch metodo_validacao
        case 'KFold'
            cv = crossval(modelo, 'KFold', 5);
        case 'LOSO'
            cv = crossval(modelo, 'Leaveout', 'on'); % Simula LOSO (ajustaremos depois)
        otherwise
            error('Método de validação não reconhecido.');
    end

    % Acurácia média
    resultados.acuracia = 1 - kfoldLoss(cv, 'LossFun', 'ClassifError');
    resultados.cm = confusionmat(Y, kfoldPredict(cv));

    fprintf('Acurácia: %.2f %%\n', resultados.acuracia * 100);
    disp('Matriz de Confusão:');
    disp(resultados.cm);
end
%..........................................................................


% Análise de Desempenho -> Prefixo AD......................................
function ADResultados(cm)

% Número de classes
K = size(cm,1);

% Nomes das classes (se não fornecido)
class_names = arrayfun(@num2str, 1:K, 'UniformOutput', false);

% Normalizar por linha (percentual)
cm_percent = 100*cm ./ sum(cm,2);

% Plotar matriz de confusão
figure;
imagesc(cm_percent);
colormap(jet);
colorbar;
xlabel('Classe Prevista'); ylabel('Classe Real');
title('Matriz de Confusão (%)');
axis square;
set(gca,'XTick',1:K,'XTickLabel',class_names);
set(gca,'YTick',1:K,'YTickLabel',class_names);

% Adicionar valores no gráfico
textStrings = num2str(cm_percent(:),'%0.1f');
textStrings = strtrim(cellstr(textStrings));
[x,y] = meshgrid(1:K);
hStrings = text(x(:),y(:),textStrings(:), 'HorizontalAlignment','center', 'Color','w');

% Métricas de desempenho
accuracy = sum(diag(cm))/sum(cm(:));
precision = diag(cm)./sum(cm,1)';  % TP / (TP+FP)
recall    = diag(cm)./sum(cm,2);   % TP / (TP+FN)
F1        = 2*(precision.*recall)./(precision+recall);

% MCC (Matthews Correlation Coefficient) - multiclass
t_sum = sum(cm,2);  % total por classe real
p_sum = sum(cm,1)'; % total por classe prevista
c = trace(cm);
s = sum(cm(:));
MCC_num = c*s - t_sum'*p_sum;
MCC_den = sqrt( (s^2 - p_sum'*p_sum) * (s^2 - t_sum'*t_sum) );
MCC = MCC_num / MCC_den;

% Exibir métricas
fprintf('Acurácia global: %.2f %%\n', accuracy*100);
for k = 1:K
    fprintf('Classe %s -> Precision: %.2f %% | Recall: %.2f %% | F1-score: %.2f %%\n', ...
        class_names{k}, precision(k)*100, recall(k)*100, F1(k)*100);
end
fprintf('MCC global: %.3f\n', MCC);
end
%..........................................................................

