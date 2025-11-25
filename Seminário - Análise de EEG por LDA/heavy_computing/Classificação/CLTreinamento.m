function resultados = CLTreinamento(X, Y, metodo_validacao, classificador)

    % Criação do modelo
    switch classificador
        case 'LDA'
            modelo = fitcdiscr(X, Y);
        otherwise
            error('Classificador não reconhecido.');
    end

    % Validação cruzada
    switch metodo_validacao
        case 'KFold'
            cv = crossval(modelo, 'KFold', 5);
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