function [ disp_int, disp_int_only_face ] = interF (dispartity_ref,disparityMapBM,mask_middle,features_m,disp_FP,disparity_feature_points, ... 
    disparityRange, thDisp,face_lower_th,face_upper_th)

win_size=199;       %size of the mask which is used for Gaussian kernels
[M, N]=size(dispartity_ref);  %size of the disparity map
w=20;   % (2*w+1) is the size of the mask which is used for interpolation
%face_lower_th = 20; 
%face_upper_th = 100;

%% Displaying feature points depths
for i=1:68
    x=uint32(features_m(i,1));
    y=uint32(features_m(i,2));
    disp_FP(y-5:y+5,x-5:x+5) =   disp_FP(y,x);
end
%figure();imshow(disp_FP,disparityRange);colormap jet;colorbar;

%% Putting Gaussians on the face 
sigma_initial = 20*ones(68,1);
F = face(sigma_initial,M,N,win_size,features_m,disparity_feature_points,dispartity_ref,mask_middle);
%figure();imshow(F,disparityRange);colormap jet;colorbar;

%% Get rid of outliers 
inliers = abs(dispartity_ref-F)<thDisp;
dispartity2_ref_out=dispartity_ref.*inliers;
%figure();imshow(dispartity2_ref_out,disparityRange);colormap jet;colorbar;

%% Adding valid BM (block matching) disparities
inliersBM = abs(disparityMapBM-F)<thDisp;
dispartity2_ref_out_bm = dispartity2_ref_out + (dispartity2_ref_out==0).*disparityMapBM.*inliersBM;
%figure();imshow(dispartity2_ref_out_bm,disparityRange);colormap jet;colorbar;


%% Feature Mask 
%Take only part of the head where feature points exists (get rid of upper part of head and neck)
f_y_max = features_m(features_m==max(features_m(:,2)));
f_y_min = features_m(features_m==min(features_m(:,2)));
f_y_max=f_y_max(1);f_y_min=f_y_min(1);
mask_middle_croped_face=zeros(M,N);
mask_middle_croped=zeros(M,N);
for i = 1:M
    for j=1:N
        if mask_middle(i,j)==1 && i< f_y_max+face_lower_th && i>f_y_min-face_upper_th
            mask_middle_croped_face(i,j)=1;
        end
        if  mask_middle(i,j)==1 && i< f_y_max+face_lower_th 
            mask_middle_croped(i,j)=1;
        end
    end
end

%figure();imshow(mask_middle_croped,[]);

%% Filling missing pixels in the disparity map --interpolation
disp_int(:,:) = dispartity2_ref_out_bm(:,:).*mask_middle_croped_face ...
+ dispartity_ref.*(mask_middle_croped-mask_middle_croped_face);
mask=fspecial('gauss',2*w+1,5);

for i=1:M
    for j=1:N
        if mask_middle_croped(i,j)==1 && disp_int(i,j)==0
            win = disp_int(i-w:i+w,j-w:j+w);
            nonZeros= win~=0;
            win=mask.*win;
            if sum(sum(mask.*nonZeros)) ~=0
                disp_int(i,j) = sum(sum(win.*nonZeros))/sum(sum(mask.*nonZeros));
            end
        end
    end      
end

disp_int_only_face = disp_int.*mask_middle_croped_face;
%figure();imshow(disp_int,disparityRange);colormap jet;colorbar;
%figure();imshow(disp_int_only_face,disparityRange);colormap jet;colorbar;


end

