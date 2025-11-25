function [Medidas_PLV_Stats_Agg, Medidas_PLV_Redes_Agg] = CFMedidasAgregadas_PLV(NumSujeito, pathdataset)
% CFMEDIDASAGREGADAS_PLV - Lê os arquivos PLV salvos no disco (batching),
% calcula as medidas estatísticas e de rede, e retorna os arrays finais.
%
% ENTRADAS:
% NumSujeito  : Número total de sujeitos.
% pathdataset : Caminho para a pasta onde os arquivos 'plv_sujeito_X.mat' estão salvos.
%
% SAÍDAS:
% Medidas_PLV_Stats_Agg : Array numérico com todas as medidas estatísticas (concatenadas).
% Medidas_PLV_Redes_Agg : Array numérico com todas as medidas de rede (concatenadas).

    % Inicializa os arrays de agregação
    Medidas_PLV_Stats_Agg = [];
    Medidas_PLV_Redes_Agg = [];

    fprintf('\n--- Agregação de Medidas PLV iniciada (Lendo do Disco) ---\n');

    for s = 1:NumSujeito
        fprintf('  Lendo e processando Sujeito %d/%d...\n', s, NumSujeito);
        
        % 1. Constrói o nome do arquivo para o sujeito atual
        nome_arquivo = fullfile(pathdataset, sprintf('plv_sujeito_%d.mat', s));
        
        % Verifica se o arquivo existe antes de carregar
        if ~exist(nome_arquivo, 'file')
            warning('Arquivo não encontrado: %s. Pulando este sujeito.', nome_arquivo);
            continue;
        end
        
        % 2. Carregar a célula PLV do disco.
        % Variável carregada será 'plv_sujeito' de tamanho [Trials x Epocas]
        load(nome_arquivo, 'plv_sujeito'); 
        
        % 3. Adaptar o formato: As funções originais (CFMedidasEstatisticas e CFMedidasRedes)
        % esperam a célula no formato [Sujeito x Trial x Epoca].
        nTrials = size(plv_sujeito, 1);
        nEpocas = size(plv_sujeito, 2);
        
        % Cria uma célula 1xTRIALxEPOCA
        plv_celula_3D = cell(1, nTrials, nEpocas); 
        plv_celula_3D(1, :, :) = plv_sujeito; % Atribui os dados do sujeito à dimensão 1
        
        % 4. Calcular as medidas usando as funções originais do seu código
        % As funções CFMedidasEstatisticas/CFMedidasRedes AGORA recebem apenas
        % a célula pequena (1 x Trial x Epoca), o que é seguro para a memória.
        Medidas_Stats = CFMedidasEstatisticas(plv_celula_3D);
        Medidas_Redes = CFMedidasRedes(plv_celula_3D);
        
        % 5. Concatenar (agregar) os resultados do sujeito atual aos arrays finais
        Medidas_PLV_Stats_Agg = [Medidas_PLV_Stats_Agg; Medidas_Stats];
        Medidas_PLV_Redes_Agg = [Medidas_PLV_Redes_Agg; Medidas_Redes];
        
        % 6. Limpar variáveis grandes temporárias para liberar RAM
        clear plv_sujeito plv_celula_3D;
    end
    
    fprintf('--- Agregação de Medidas PLV finalizada com sucesso! ---\n');
end