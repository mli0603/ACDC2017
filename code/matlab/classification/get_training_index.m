prefix = 'C:\Users\Maxwell\OneDrive - Johns Hopkins University\Medical Image Analysis\Project\Cardiac\matlab\dataset\train\';
patients_folder = dir(strcat(prefix,'patient*'));
patients_names = {patients_folder.name};

file_train = fopen('train_data.txt','w');
file_validation = fopen('validation_data.txt','w');

for i = 1:size(patients_names,2)
    name = char(patients_names(i));
    idx = str2num(erase(name,'patient'));
    fprintf(file_train,'%d\n',idx);
    train_idx(i,1) = idx;
end

for i = 1:100
    if ~any(train_idx(:) == i)
        fprintf(file_validation,'%d\n',i);
    end
end

fclose(file_train);
fclose(file_validation);
