function num = ParseLine(line)
    % param: 
    % line: a line of string from the txt file
    % output
    % num: a number parsed from the line
    
    cell = strsplit(line,':');
    
    num = str2num(cell2mat(cell(2))); 
end
