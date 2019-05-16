function features = ExtractDynamicFeatures(ED_features, ES_features)
    % function to find the dynamic features, i.e., between ED and ES
    % rv has label of 1
    % myocardium has label of 2
    % lv has label of 3
    % params:
        % ED_features, ES_features from ExtractInstantFeature function
    % output:
        % features: dynamic features
    
    volume_idx = 8;
    
    % find volume
    rv_volume_ED = ED_features(volume_idx);
    rv_volume_ES = ES_features(volume_idx);
    
    myo_volume_ED = ED_features(volume_idx + size(ED_features,2)/3);
    myo_volume_ES = ES_features(volume_idx + size(ES_features,2)/3);
    
    lv_volume_ED = ED_features(volume_idx + size(ED_features,2)/3*2);
    lv_volume_ES = ES_features(volume_idx + size(ED_features,2)/3*2);

    % ejection fraction
    rv_ejection_fraction = (rv_volume_ED-rv_volume_ES)/rv_volume_ED;
    myo_ejection_fraction = (myo_volume_ED-myo_volume_ES)/myo_volume_ED;
    lv_ejection_fraction = (lv_volume_ED-lv_volume_ES)/lv_volume_ED;
    
    % ratio of volume
    rvlv_ratio_ED = double(rv_volume_ED) / lv_volume_ED;
    rvlv_ratio_ES = double(rv_volume_ES) / lv_volume_ES;
    myolv_ratio_ED = double(myo_volume_ED) / lv_volume_ED;
    myolv_ratio_ES = double(myo_volume_ES) / lv_volume_ES;
    
    features = [rv_ejection_fraction,myo_ejection_fraction,lv_ejection_fraction...
        rvlv_ratio_ED,rvlv_ratio_ES,myolv_ratio_ED,myolv_ratio_ES];
end