%% Etapa de Carregamento...................................................
clear; clc; close all;

pathdataset = './dataset/';
nome_input = 'EEG_Data_BCI_IV_2a.mat';
nome_output = 'BCI_IV2a_preprocessed.mat';
baseDir = fileparts(matlab.desktop.editor.getActiveFilename);
funcDir = fullfile(baseDir, 'heavy_computing');
addpath(genpath(funcDir));
load([pathdataset nome_input]);

disp('Etapa de Carregamento Finalizada');
disp('------------------------------------------------------------------');
%..........................................................................

%% Etapa de Visualização -> Variáveis Sujeito, Época e Canal
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

%% Etapa de Pré-Processamento
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
%[EEG_clean, S_all, W_all, removed_all] = PPRemoveArtefatosICA(EEG_data);
%save([pathdataset 'EEG_Data_SemArtefatos.mat'],'EEG_clean','S_all','W_all','removed_all','-v7.3');

%Comentar a linha abaixo se for rodar a remoção de artefatos
load([pathdataset 'EEG_Data_SemArtefatos.mat']); %-> Load p/ teste
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
Medidas_estatisticas = MNMedidasEstatisticas(EEG_data); %talvez mudar EEG_data?


%Parâmetros de Hjorth
Medidas_hjorth = MNMedidasHjorth(EEG_data); %talvez mudar EEG_data?


%Potência das bandas -> Descomentar linhas abaixo para rodar mineração
%fs = 128;
%plot_flag = 0;
%band_features = MNPotenciaBandas(EEG_epochs, fs, chan_names, plot_flag);

%Comentar a linha abaixo se for rodar a mineração
load([pathdataset 'Medidas_Bandas_Potencia.mat']); % -> Load p/ testes


disp('Etapa de Mineração Finalizada');
disp('-------------------------------------------------------------------')
%..........................................................................


%% Conectividade Funcional
%Correlação
correlacao = CFCorrelacao(EEG_data);  %talvez mudar EEG_data?


%% Phase-Locking-Value -> "Sincronização de sinais EEG"
plv = CFPLV(EEG_data, fs);


%% Medidas Estastísticas
Medidas_correlacao_estastisticas = CFMedidasEstatisticas(correlacao);
Medidas_correlacao_redes = CFMedidasRedes(correlacao);

%% Medidas Topológicas
Medidas_PLV_estatisticas = CFMedidasEstatisticas(plv);
Medidas_PLV_Redes = CFMedidasRedes(plv);


disp('Etapa de Conectividade Funcional Finalizada');
disp('-------------------------------------------------------------------')
%..........................................................................


%% Matriz de Medidas
%Montagem da Matriz
[X, feature_names] = MDMMontarMatriz(Medidas_estatisticas,Medidas_hjorth);


%Normalização
X_norm = MDMNormalizar(X, 'zscore');


disp('Etapa de Matriz de Medidas Finalizada');
disp('-------------------------------------------------------------------')
%..........................................................................


%% Redução de Dimensionalidade
%Reducao de Dimensionalidade
[X_pca, V, lambda, explained] = RDPCA(X_norm, 'var', 95);
RDPlotar(explained, lambda);


disp('Etapa de Redução de Dimensionalidade Finalizada');
disp('-------------------------------------------------------------------')
%..........................................................................


%% Classificação
%Vetor das classes
[NumSujeito,NumCan,NumAmo,NumEpocas,NumTrial] = size(EEG_data);
Y = CLMontarVetor(NumSujeito, NumTrial, NumEpocas, labels);


%K-Fold
folds = CLKFold(X, Y, 'KFold', [], 5)


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

%% SESSÃO DE FUNÇÕES DE BAIXA COMPLEXIDADE
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

% Plot Pós Redução de Dimensionalidade por PCA
function RDPlotar(explained, lambda)
    figure;
    
    % -------------------------------
    % 1) Plotar variância explicada
    % -------------------------------
    subplot(1, 2, 1);
    bar(explained, 'FaceAlpha', 0.7);
    xlabel('Componente Principal');
    ylabel('Variância Explicada (%)');
    title('Variância Explicada por Componente');
    grid on;

    % -------------------------------
    % 2) Curve Scree Plot (autovalores)
    % -------------------------------
    if nargin > 1
        subplot(1, 2, 2);
        plot(lambda, '-o', 'LineWidth', 1.5);
        xlabel('Componente Principal');
        ylabel('Autovalor');
        title('Scree Plot (Autovalores)');
        grid on;
    end

end


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

