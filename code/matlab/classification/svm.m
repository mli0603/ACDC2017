%% load data
load feature_vec
load feature_vec_val
%% SVM classifier
class = {'MINF','DCM','HCM','RV','NOR'};

rng(1); % For reproducibility
t = templateSVM('Standardize',true, 'KernelFunction','rbf','KernelScale','auto');
Mdl = fitcecoc(feature_vec,Y,'Learners',t,...
    'ClassNames',class,'Verbose',2);
error = resubLoss(Mdl)