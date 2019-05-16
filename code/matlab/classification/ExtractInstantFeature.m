function features = ExtractInstantFeature(volume,Mosteller)
    % function to extract cardiac features from volume
    % rv has label of 1
    % myocardium has label of 2
    % lv has label of 3
    
    % params
        % volume: labeled image of the heart
        % Mosteller: body surface
        % weight: mass of patient
    % output
        % features: array containing extracted features
        
    for i = 1:size(volume,3)
        curr_stats = regionprops('struct',volume(:,:,i),'Area','Circularity',...
        'MajorAxisLength','MinorAxisLength','Perimeter','Centroid');
        
        if size(curr_stats,1) > 0
            % RV 
            rv_thickness(i) = curr_stats(2).MinorAxisLength;
            rv_circularity(i) = curr_stats(2).Circularity;
            rv_circumference(i) = curr_stats(2).Perimeter;
            rv_area(i) = curr_stats(2).Area;
        end
        if size(curr_stats,1) > 1
            % myocardium 
            myo_thickness(i) = curr_stats(1).MinorAxisLength; % assuming thickness is the length of minoraxis
            myo_circularity(i) = curr_stats(1).Circularity;
            myo_circumference(i) = curr_stats(1).Perimeter; % assume circumference is the perimeter
            myo_area(i) = curr_stats(1).Area;
        end 
        if size(curr_stats,1) > 2
        % LV 
            lv_thickness(i) = curr_stats(3).MinorAxisLength;
            lv_circularity(i) = curr_stats(3).Circularity;
            lv_circumference(i) = curr_stats(3).Perimeter;
            lv_area(i) = curr_stats(3).Area;
        end
    end
    
    % clean
    myo_thickness = nonzeros(myo_thickness)';
    myo_circularity = nonzeros(rmmissing(myo_circularity))';
    myo_circularity(find(myo_circularity==Inf)) = [];
    myo_circumference = nonzeros(myo_circumference)';
    myo_area = nonzeros(myo_area)';
    
    rv_thickness = nonzeros(rv_thickness)';
    rv_circularity = nonzeros(rmmissing(rv_circularity)');
    rv_circumference = nonzeros(rv_circumference)';
    rv_area = nonzeros(rv_area)';
    
    lv_thickness = nonzeros(lv_thickness)';
    lv_circularity = nonzeros(rmmissing(lv_circularity)');
    lv_circumference = nonzeros(lv_circumference)';
    lv_area = nonzeros(lv_area)';
    
    % find instant sub features
    myo_instant_subfeatures = ExtractInstantSubFeatures(myo_thickness,myo_circularity,myo_circumference,myo_area,Mosteller);
    rv_instant_subfeatures = ExtractInstantSubFeatures(rv_thickness,rv_circularity,rv_circumference,rv_area,Mosteller);
    lv_instant_subfeatures = ExtractInstantSubFeatures(lv_thickness,lv_circularity,lv_circumference,lv_area,Mosteller);

    features = [myo_instant_subfeatures,rv_instant_subfeatures,lv_instant_subfeatures];
end

function subfeature = ExtractInstantSubFeatures(thickness,circularity,circumference,area,Mosteller)
    % helper function to extract subfeatures, such as min, max, mean
    max_thickness = max(thickness);
    min_thickness = min(thickness);
    std_thickness = std(thickness);
    mean_thickness = mean(thickness);
    
    mean_circularity = mean(circularity);
    
    max_circumference = max(circumference);
    mean_circumference = mean(circumference);
      
    volume = sum(area);
    volume_per_surface = sum(area)/Mosteller;
    
    subfeature = [max_thickness,min_thickness,std_thickness,mean_thickness,...
        mean_circularity,max_circumference,mean_circumference,volume,...
        volume_per_surface];
end