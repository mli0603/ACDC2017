clear; clc; close all
load train
%% initialize mapping between class and index
% map = containers.Map;
class = {'MINF','DCM','HCM','RV','NOR'};
ind = [1,2,3,4,5];
map = containers.Map(class,ind);

%% build a feature vector
prefix = 'C:\Users\Maxwell\OneDrive - Johns Hopkins University\Medical Image Analysis\Project\Cardiac\matlab\dataset\validation\';
patients_folder = dir(strcat(prefix,'patient*'));
patients_names = {patients_folder.name};

feature = zeros(50,62);
label_name = cell(50,1);
for i = 1:size(patients_names,2)
    name = patients_names{i};
    info = ParseInfo(prefix,name);
    ind = map(info{3});
    
    height = info{4};
    weight = info{6};
    V_ED_gt = niftiread(strcat(prefix,name,'/',name,'_frame',num2str(info{1},'%02d'),'_gt.nii.gz'));
    V_ES_gt = niftiread(strcat(prefix,name,'/',name,'_frame',num2str(info{2},'%02d'),'_gt.nii.gz'));    
%     Mosteller =  sqrt(double(height)*weight/3600);
%    
%     % ED features
%     ED_features = ExtractInstantFeature(V_ED_gt,Mosteller);
%     % ES features
%     ES_features = ExtractInstantFeature(V_ES_gt,Mosteller);
%     % dynamic features
%     dynamic_features = ExtractDynamicFeatures(ED_features, ES_features);
    
    % build feature vec
    label_name{i} = info{3};
    label(i) = map(info{3}); 
%     feature(i,:) = [ED_features,ES_features,dynamic_features,weight];
    feature(i,:) = BuildFeatureVector(V_ED_gt,V_ES_gt,height,weight,tmp_mean,tmp_std);
end
label = label';
%%
save validation feature label_name label