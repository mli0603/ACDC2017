clear; clc; close all
load('../s.mat')
%% initialize mapping between class and index
% map = containers.Map;
class = {'MINF','DCM','HCM','RV','NOR'};
ind = [1,2,3,4,5];
map = containers.Map(class,ind);

%% build a feature vector
% prefix = 'C:\Users\Maxwell\OneDrive - Johns Hopkins University\Medical Image Analysis\Project\Cardiac\matlab\dataset\train\';
% patients_folder = dir(strcat(prefix,'patient*'));
% patients_names = {patients_folder.name};

feature = zeros(50,62);
label_name = cell(50,1);
for i = 1:size(s,2)
%     name = patients_names{i};
%     info = ParseInfo(prefix,name);
%     ind = map(info{3});
    
%     height = info{4};
%     weight = info{6};
    SegED = s(i).SegED;
    SegES = s(i).SegES; 
    
    figure(1)
    imshow(SegED(:,:,3),[]);
    
    % build feature vec
    label(i) = s(i).Group; 
    
    switch uint8(label(i))
        case 2
            label_name(i) = cellstr('DCM');
        case 1
            label_name(i) = cellstr('MINF');
        case 3
            label_name(i) = cellstr('HCM');
        case 4
            label_name(i) = cellstr('RV');
        otherwise
            label_name(i) = cellstr('NOR');
    end

    feature(i,:) = BuildFeatureVector(SegED,SegES,s(i).Height,s(i).Weight,0,1);
end
label = label';
tmp_mean = mean(feature);
tmp_std = std(feature);
feature = (feature - tmp_mean)./tmp_std;
%%
save train feature label_name label tmp_mean tmp_std