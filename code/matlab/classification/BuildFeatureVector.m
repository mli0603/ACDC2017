function feature = BuildFeatureVector(ED,ES,height,weight,mean,std)
    Mosteller =  sqrt(double(double(height)*weight/3600.0));

    % ED features
    ED_features = ExtractInstantFeature(ED,Mosteller);
    % ES features
    ES_features = ExtractInstantFeature(ES,Mosteller);
    % dynamic features
    dynamic_features = ExtractDynamicFeatures(ED_features, ES_features);
    
    % build feature vec
    feature = [ED_features,ES_features,dynamic_features,double(weight)];
    feature = (feature-mean)./std;
end