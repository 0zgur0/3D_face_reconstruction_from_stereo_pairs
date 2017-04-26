function [pixel_labels] = k_means(img,nColors)
%%%%%%%%%%%%%%%%%
cform = makecform('srgb2lab');
lab_he = applycform(img,cform);

ab = double(lab_he(:,:,2:3));
nrows = size(ab,1);
ncols = size(ab,2);
ab = reshape(ab,nrows*ncols,2);

%nColors = 3;
% repeat the clustering 3 times to avoid local minima
[cluster_idx, cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean', ...
'Replicates',3);

pixel_labels = reshape(cluster_idx,nrows,ncols);

end