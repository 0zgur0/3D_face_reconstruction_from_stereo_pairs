function [ foreground ] = face_mask_extraction( I , which_image )
img = imcrop(I,[1 1 size(I,1) size(I,2)/2-1]);
%img = imresize(img,0.5);
[m,n,x] = size(img);

% k-means for half part of the imput image
pixel_labels = k_means(img,3);
%figure();imshow(pixel_labels,[]);

%Expand the image 
pixel_labels_extended = zeros(2*m,n);
pixel_labels_extended(1:m,:) = pixel_labels(:,:);                                

%Labels
if which_image=='r'
    label_back = round(mean(mean((pixel_labels(1:end,1:round((n/5)))))));
elseif  which_image=='l'
    label_back = round(mean(mean((pixel_labels(1:end,round(4*n/5):end)))));
else
    label_back = round(mean(mean((pixel_labels(1:end,1:round((n/5)))))));
end
    
labeled_pixels = pixel_labels_extended ~= label_back;
labeled_pixels(m+1:2*m,:) = 0;
%figure();imshow(labeled_pixels,[]);

% k-means for entire image
pixel_labels2 = k_means(I,3);

%Labels
if which_image=='r'
    label_face = round(mean(mean((pixel_labels2(round(2*m/5):round(3*m/5),round(4*n/5):end)))));
elseif  which_image=='l'
    label_face = round(mean(mean((pixel_labels2(round(2*m/5):round(3*m/5),1:round(1*n/5))))));
else
    label_face = round(mean(mean((pixel_labels2(round(2*m/5):round(3*m/5),round(2*n/5):round(3*n/5))))));
end

labeled_pixels2 = pixel_labels2 == label_face;

foreground = (labeled_pixels | labeled_pixels2);
%figure();imshow(foreground,[]);title('foreground');

%Deleting small details
SE = strel('disk', 12);
seed = imerode(foreground,SE);
foreground = imreconstruct(seed,foreground);
%figure();imshow(foreground,[]);title('delete small details');


%% Region Growing
% foreground = region_growing(I,foreground,30);
% figure();imshow(foreground,[]);title('region growing');

% %Deleting small details
% SE = strel('disk', 12);
% seed = imerode(foreground,SE);
% foreground = imreconstruct(seed,foreground);
% figure();imshow(foreground,[]);title('delete small details 2');

%Region-filling
SE2 = strel('disk',8);
SE3 = strel('disk',30);
foreground = imopen(foreground,SE2);
foreground = imclose(foreground,SE3);
%figure();imshow(foreground,[]);title('region filling');


%Deleting small details
SE = strel('disk', 24);
seed = imerode(foreground,SE);
foreground = imreconstruct(seed,foreground);
%figure();imshow(foreground,[]);title('delete small details 3');

end