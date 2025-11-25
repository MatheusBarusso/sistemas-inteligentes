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
        otherwise
            error('Método não reconhecido. Use "KFold".');
    end
end