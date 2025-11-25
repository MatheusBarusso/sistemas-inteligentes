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