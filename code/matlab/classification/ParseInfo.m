function info = ParseInfo(prefix, patient_name)
    % function to parse patient infomation
    % param
    % prefix: file directory
    % patient_name: patient name for data subfolder
    % output
    % info: information of patient, including ED/ES frame#, group, height,
    % total number of frames in the 3D volume and weight
    
    filename = strcat(prefix,patient_name,'/Info.cfg');
    fid = fopen(filename);
    
    %read the first line (ED)
    line1 = fgetl(fid);
    ED = ParseLine(line1);
    
    %read the second line (ES)
    line2 = fgetl(fid);
    ES = ParseLine(line2);
    
    %read the third line (Group)
    line3 = fgetl(fid);
    cell = strsplit(line3,':');
    Group = strtrim(cell{2});
    
    %read the fourth line (Height)
    line4 = fgetl(fid);
    Height = ParseLine(line4);
    
    %read the fifth line (NbFrame)
    line5 = fgetl(fid);
    NbFrame = ParseLine(line5);
    
    %read the fifth line (Weight)
    line5 = fgetl(fid);
    Weight = ParseLine(line5);
    
    info = {ED,ES,Group,Height,NbFrame,Weight};
end