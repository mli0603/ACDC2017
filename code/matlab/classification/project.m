clear; clc; close all

%% initialize mapping between class and index
% map = containers.Map;
class = {'MINF','DCM','HCM','RV','NOR'};
ind = [1,2,3,4,5];
map = containers.Map(class,ind);

%% build a distribution of features
prefix = '/home/mli/OneDrive/Medical Image Analysis/Project/Cardiac/matlab/dataset/train/';
patients_folder = dir(strcat(prefix,'patient*'));
patients_names = {patients_folder.name};

ejection_fraction = cell(5,1);
I_ED_rv = cell(5,1);
weight = cell(5,1);
height = cell(5,1);

for i = 1:size(patients_names,2)
    name = patients_names{i};
    info = ParseInfo(prefix,name);
    ind = map(info{3});
    
    height{ind} = [height{ind};info{4}];
    weight{ind} = [weight{ind};info{6}];
    V_ED = niftiread(strcat(prefix,name,'/',name,'_frame',num2str(info{1},'%02d'),'.nii'));
    V_ED_gt = niftiread(strcat(prefix,name,'/',name,'_frame',num2str(info{1},'%02d'),'_gt.nii'));
    V_ES = niftiread(strcat(prefix,name,'/',name,'_frame',num2str(info{2},'%02d'),'.nii'));
    V_ES_gt = niftiread(strcat(prefix,name,'/',name,'_frame',num2str(info{2},'%02d'),'_gt.nii'));
        
    subplot(1,2,1)
    imshow(V_ED(:,:,3),[])
    title('Cardiac Image Slice')
    subplot(1,2,2)
    imshow(V_ED_gt(:,:,3),[])
    title('Segmentation Mask Slice')
    
    % normalize images
    V_ED = V_ED/max(V_ED(:));
    V_ES = V_ES/max(V_ES(:));
    
    % problem 1, find mean intensity of rv
    rv_region = V_ED_gt == 1; % rv has label of 1
    I_ED_rv{ind} = [I_ED_rv{ind};mean(V_ED(rv_region))];
    
    % problem 2, find the ejectin fraction
    lv_ED_region = V_ED_gt == 3; % lv has label of 3
    lv_ES_region = V_ES_gt == 3; % lv has label of 3
    ejection_fraction{ind} = [ejection_fraction{ind};(sum(lv_ED_region(:))-sum(lv_ES_region(:)))/sum(lv_ED_region(:))];
end
%%
color = {'c','b','r','g','y'};

figure
for j = 1:5
    subplot(5,1,j)
    histogram(I_ED_rv{j},linspace(0.05,0.95,10),'FaceColor',color{j},'FaceAlpha',.5,'EdgeColor','none');
    ylabel('Count')
    legend(class{j})
end
xlabel('Mean Intensity of RV')
xlim([0,1])
saveas(gcf,'intensity.png')

figure
for j = 1:5
    subplot(5,1,j)
    histogram(ejection_fraction{j},linspace(0.05,0.95,10),'FaceColor',color{j},'FaceAlpha',.5,'EdgeColor','none');
    ylabel('Count')
    legend(class{j})
end
xlabel('Ejection fraction of LV')
xlim([0,1])
saveas(gcf,'ejection_fraction.png')

figure
for j = 1:5
    subplot(5,1,j)
    histogram(weight{j},linspace(30,120,10),'FaceColor',color{j},'FaceAlpha',.5,'EdgeColor','none');
    ylabel('Count')
    legend(class{j})
end
xlabel('Weight')
saveas(gcf,'weight.png')

figure
for j = 1:5
    subplot(5,1,j)
    histogram(height{j},linspace(140,200,10),'FaceColor',color{j},'FaceAlpha',.5,'EdgeColor','none');
    ylabel('Count')
    legend(class{j})
end
xlabel('Height')
saveas(gcf,'height.png')

%% build a feature vector
prefix = '/home/mli/OneDrive/Medical Image Analysis/Project/Cardiac/matlab/dataset/train/';
patients_folder = dir(strcat(prefix,'patient*'));
patients_names = {patients_folder.name};

% height, weight, [V_LV, V_RV, V_myocardial at ED and ES (in ml)], [the LV and RV ejection fraction],
% [the ratio between RV and LV volume at ED and ES, the ratio between myocardial and LV volume at ED and ES]
feature_vec = zeros(50,14);
Y = cell(50,1);
for i = 1:size(patients_names,2)
    name = patients_names{i};
    info = ParseInfo(prefix,name);
    ind = map(info{3});
    
    V_ED_gt = niftiread(strcat(prefix,name,'/',name,'_frame',num2str(info{1},'%02d'),'_gt.nii'));
    V_ES_gt = niftiread(strcat(prefix,name,'/',name,'_frame',num2str(info{2},'%02d'),'_gt.nii'));
    
    % patient-based
    height = info{4};
    weight = info{6};
    % volume
    lv_ED_region = V_ED_gt == 3; % lv has label of 3
    my_ED_region = V_ED_gt == 2; % myocardium has label of 2
    rv_ED_region = V_ED_gt == 1; % rv has label of 1
    lv_ES_region = V_ES_gt == 3;
    rv_ES_region = V_ES_gt == 2;
    my_ES_region = V_ES_gt == 1;
    
    lv_ED_volume = sum(lv_ED_region(:));
    my_ED_volume = sum(my_ED_region(:));
    rv_ED_volume = sum(rv_ED_region(:));
    
    lv_ES_volume = sum(lv_ES_region(:));
    my_ES_volume = sum(my_ES_region(:));
    rv_ES_volume = sum(rv_ES_region(:));
    % ejection fraction
    lv_ejection_fraction = (lv_ED_volume-lv_ES_volume)/lv_ED_volume;
    rv_ejection_fraction = (rv_ED_volume-rv_ES_volume)/rv_ED_volume;
    % ratio of volume
    ed_rvlv_ratio = rv_ED_volume * 1.0 / lv_ED_volume;
    es_rvlv_ratio = rv_ES_volume * 1.0 / lv_ES_volume;
    ed_mylv_ratio = my_ED_volume * 1.0 / lv_ED_volume;
    es_mylv_ratio = my_ES_volume * 1.0 / lv_ES_volume;
    
    % build feature vec
    Y{i} = info{3};
    feature_vec(i,:) = [height,weight,lv_ED_volume,my_ED_volume,rv_ED_volume,...
        lv_ES_volume,my_ES_volume,rv_ES_volume,lv_ejection_fraction,rv_ejection_fraction...
        ed_rvlv_ratio,es_rvlv_ratio,ed_mylv_ratio,es_mylv_ratio]; 
end
%% split between training and validation
train = randperm(50,40);
val = setdiff(linspace(1,50,50),train);
feature_train = feature_vec(train,:);
feature_val = feature_vec(val,:);
Y_train = Y(train,:);
Y_val = Y(val,:);
%% random forest classifier
Mdl = TreeBagger(50,feature_train,Y_train,'OOBPrediction','On',...
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

% select most important 5 features and train
idxvar = find(Mdl.OOBPermutedPredictorDeltaError>0.6);
b5v = TreeBagger(100,feature_train(:,idxvar),Y_train,'OOBPredictorImportance','off','OOBPrediction','on');
figure
plot(oobError(b5v))
xlabel('Number of Grown Trees')
ylabel('Out-of-Bag Classification Error')

% predict
Y_pred = predict(b5v,feature_val(:,idxvar));

% calculate accuracy
C = confusionmat(Y_val,Y_pred);
confusionchart(C,class);
saveas(gcf,'SVM_confusion.png');

TP = 0;
for i = 1:size(Y_pred,1)
    if strcmp(Y_pred{i}, Y_val{i})
        TP = TP + 1;
    end
end
accuracy = TP/size(Y_pred,1)