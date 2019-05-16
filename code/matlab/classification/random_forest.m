%% load data
load train
%% random forest classifier
rng(1)
Mdl = TreeBagger(1000,feature,label,'OOBPrediction','On',...
    'Method','classification','OOBPredictorImportance','On');

% training error
figure
oobErrorBaggedEnsemble = oobError(Mdl);
plot(oobErrorBaggedEnsemble)
xlabel 'Number of grown trees';
ylabel 'Out-of-bag classification error';

% feature importance
figure
bar(Mdl.OOBPermutedPredictorDeltaError)
xlabel('Feature Index')
ylabel('Out-of-Bag Feature Importance')
ylabel('Out-of-Bag Feature Importance')

% select effective features
idxvar = find(Mdl.OOBPermutedPredictorDeltaError>0.325);
classifier = TreeBagger(500,feature(:,idxvar),label_name,'OOBPredictorImportance','off','OOBPrediction','on');
figure
plot(oobError(classifier))
xlabel('Number of Grown Trees')
ylabel('Out-of-Bag Classification Error')

%% on training set
% predict
label_pred = predict(classifier,feature(:,idxvar));

% calculate accuracy
C = confusionmat(label_name,label_pred);
confusionchart(C,class);

TP = 0;
for i = 1:size(label_pred,1)
    if strcmp(label_pred{i}, label_name{i})
        TP = TP + 1;
    end
end
accuracy = TP/size(label_pred,1)
%% on validation set
load validation

% predict
label_pred = predict(classifier,feature(:,idxvar));

% calculate accuracy
C = confusionmat(label_name,label_pred);
confusionchart(C,class);

TP = 0;
for i = 1:size(label_pred,1)
    if strcmp(label_pred{i}, label_name{i})
        TP = TP + 1;
    end
end
accuracy = TP/size(label_pred,1)