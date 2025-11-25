function plv_suj = CFPLV_SingleSubject(EEG_suj_data, fs)

[nTrials, nEp, nCan, nAmo] = size(EEG_suj_data);
plv_suj = cell(nTrials, nEp);

for t = 1:nTrials
    for e = 1:nEp
        sinal = squeeze(EEG_suj_data(t, e, :, :));
        phase = angle(hilbert(sinal')');
        plvMat = zeros(nCan, nCan);

        for i = 1:nCan
            for j = i:nCan
                deltaPhi = phase(i,:) - phase(j,:);
                plvMat(i,j) = abs(mean(exp(1j*deltaPhi)));
                plvMat(j,i) = plvMat(i,j);
            end
        end
        plv_suj{t,e} = plvMat;
    end
end
% fprintf('PLV calculado para todos os sujeitos, trials e épocas.\n');
end

%[nSuj, nTrials, nEp, nCan, nAmo] = size(EEG_data);
%plv_mats = cell(nSuj, nTrials, nEp);
% for s = 1:nSuj
%     fprintf('Processando Sujeito %d/%d...\n', s, nSuj);
%     for t = 1:nTrials
%         for e = 1:nEp
%             % Sinal do trial/época atual [Canais x Amostras]
%             sinal = squeeze(EEG_data(s,t,e,:,:));
%             % Transformada de Hilbert para fase
%             phase = angle(hilbert(sinal')');  % transpor e retornar a [Canais x Amostras]
%             % Inicializar matriz PLV
%             plvMat = zeros(nCan, nCan);
%             % Calcular PLV para cada par de canais
%             for i = 1:nCan
%                 for j = i:nCan
%                     deltaPhi = phase(i,:) - phase(j,:);
%                     plvMat(i,j) = abs(mean(exp(1j*deltaPhi)));
%                     plvMat(j,i) = plvMat(i,j); % simétrica
%                 end
%             end
%             plv_mats{s,t,e} = plvMat;
%         end
%     end
% end
% fprintf('PLV calculado para todos os sujeitos, trials e épocas.\n');
% end