clear all; 
%close all; clc;
%% 3D Face Reconstruction
% Load camera params
load('stereoParams_mr.mat')
load('stereoParams_ml.mat')

%Subject name and image pair
sName = '2';
pNamme = '1';

% Read facial feature points middle-right pair
features_m = csvread(strcat('s',num2str(sName),'/m',num2str(pNamme),'.csv'));
features_r = csvread(strcat('s',num2str(sName),'/r',num2str(pNamme),'.csv'));
% Read facial feature points middle-left pair
features_m2 = csvread(strcat('s',num2str(sName),'/m2_',num2str(pNamme),'.csv'));
features_l = csvread(strcat('s',num2str(sName),'/l',num2str(pNamme),'.csv'));

% Read images
img_middle = imread(strcat('s',num2str(sName),'/subject',num2str(sName),'_Middle_',num2str(pNamme),'_e1.png'));
img_middle2 = imread(strcat('s',num2str(sName),'/subject',num2str(sName),'_Middle_',num2str(pNamme),'_e2.png'));
img_right = imread(strcat('s',num2str(sName),'/subject',num2str(sName),'_Right_',num2str(pNamme),'_e.png'));
img_left = imread(strcat('s',num2str(sName),'/subject',num2str(sName),'_Left_',num2str(pNamme),'_e.png')); 

% img_middle1 = imread('m11.png');
% img_right1 = imread('r11.png');

%Convert uint8 to double
img_middle = im2double(img_middle);
img_middle2 = im2double(img_middle2);
img_right = im2double(img_right);
img_left = im2double(img_left);

% figure();imshow(img_middle,[]);
% figure();imshow(img_right,[]);

%% Face Segmentation 
mask_middle = face_mask_extraction(img_middle,'m');
mask_middle2 = mask_middle;
mask_right = face_mask_extraction(img_right,'r');
mask_left = face_mask_extraction(img_left,'l');

%% Stereo Rectification 
%--------------------------m-r pair--------------------------------
%Rectify face images 
[img_middle_rec,img_right_rec] = rectifyStereoImages(img_middle,img_right,stereoParams_mr, ...
    'OutputView','full');
%figure();imshowpair(img_middle_rec,img_right_rec,'montage');

%Rectify mask images
[mask_middle,mask_right] = rectifyStereoImages(mask_middle,mask_right,stereoParams_mr, ...
    'OutputView','full');
% figure();imshowpair(img_middle_rec,img_right_rec,'montage');

[M,N,dummy] = size(img_middle_rec);  %size of the rectified images

%--------------------------m-l pair--------------------------------
%Rectify face images 
[img_middle_rec2,img_left_rec] = rectifyStereoImages(img_middle2,img_left,stereoParams_ml, ...
    'OutputView','full');
%figure();imshowpair(img_middle_rec,img_right_rec,'montage');

%Rectify mask images
[mask_middle2,mask_left] = rectifyStereoImages(mask_middle2,mask_left,stereoParams_ml, ...
    'OutputView','full');
% figure();imshowpair(img_middle_rec,img_right_rec,'montage');

[M2,N2,dummy] = size(img_middle_rec2);  %size of the rectified images

%% Facial Feature Points 
xydif = abs(features_m-features_r);
disparity_feature_points = sqrt(xydif(:,1).^2 + xydif(:,2).^2);

xydif = abs(features_m2-features_l);
disparity_feature_points2 = -sqrt(xydif(:,1).^2 + xydif(:,2).^2);

max_disp_FP = round(max(disparity_feature_points));
min_disp_FP = round(min(disparity_feature_points));

max_disp_FP2 = round(max(disparity_feature_points2));
min_disp_FP2 = round(min(disparity_feature_points2));


%Disparty map for facial feature points
disp_FP = zeros(M,N);
disp_FP2 = zeros(M2,N2);
for i=1:68
    disp_FP(uint32(features_m(i,2)),uint32(features_m(i,1))) =  disparity_feature_points(i);
    disp_FP2(uint32(features_m2(i,2)),uint32(features_m2(i,1))) =  disparity_feature_points2(i);
end

%Determine disparity ranges according to featrue points disparities
disparityRange = disparityRangeEstimate( max_disp_FP,min_disp_FP );
disparityRange2 = disparityRangeEstimate( max_disp_FP2,min_disp_FP2 );

%% Disparity MAP

%Convert RGB image gray-level image
gray_img_m = rgb2gray(img_middle_rec);
gray_img_r = rgb2gray(img_right_rec);

% gray_img_m = img_middle_rec;
% gray_img_r = img_right_rec;

gray_img_m2 = rgb2gray(img_middle_rec2);
gray_img_l = rgb2gray(img_left_rec);

% %%Image smoothing
% h=fspecial('gaussian',5,1);
% gray_img_m = imfilter(gray_img_m,h);
% gray_img_r = imfilter(gray_img_r,h);
% gray_img_m2 = imfilter(gray_img_m2,h);
% gray_img_l = imfilter(gray_img_l,h);


%Disparity PARAMS
%disparityRange = [276-6,344+6];
bs = 15;        %defauld bs=15
cTH = 0.7;      %default 0.5
uTH = 15;       %default 15
tTH = 0.0000;   %default 0.0002 only applies if method is blockmatching
dTH = 15;       %default []

%--------------------------m-r pair-------------------------------------
%SGBM method
disparityMap1 = disparity(gray_img_m,gray_img_r,'DisparityRange',disparityRange, ...
    'ContrastThreshold',cTH, 'UniquenessThreshold',uTH, 'DistanceThreshold',dTH,'BlockSize',bs);

%Block match method
disparityMapBM1 = disparity(gray_img_m,gray_img_r,'DisparityRange',disparityRange, ...
    'ContrastThreshold',cTH, 'UniquenessThreshold',uTH, 'DistanceThreshold',dTH,'BlockSize',bs, ...
    'Method','BlockMatching');

%Visulizing disparity maps
%figure();imshow(disparityMap1,disparityRange);colormap jet;colorbar;
%figure();imshow(disparityMapBM1,disparityRange);colormap jet;colorbar;

%--------------------------m-l pair-------------------------------------
%SGBM method
disparityMap2 = disparity(gray_img_m2,gray_img_l,'DisparityRange',disparityRange2, ...
    'ContrastThreshold',cTH, 'UniquenessThreshold',uTH, 'DistanceThreshold',dTH,'BlockSize',bs);

%Block match method
disparityMapBM2 = disparity(gray_img_m2,gray_img_l,'DisparityRange',disparityRange2, ...
    'ContrastThreshold',cTH, 'UniquenessThreshold',uTH, 'DistanceThreshold',dTH,'BlockSize',bs, ...
    'Method','BlockMatching');

%Visulizing disparity maps
%figure();imshow(disparityMap2,disparityRange2);colormap jet;colorbar;
%figure();imshow(disparityMapBM2,disparityRange2);colormap jet;colorbar;


%% Unreliable Points
unreliable1 = disparityMap1 < -1e+12;
%mask_middle_rec = 1-(rgb2gray(img_middle_rec)>0);
unreliable1 = unreliable1 | (1-mask_middle);
%figure();imshow(unreliable1,[]);

unreliable2 = disparityMap2 < -1e+12;
unreliable2 = unreliable2 | (1-mask_middle2);
%figure();imshow(unreliable2,[]);


%Get rid of unrelible pixels
dispartity1_ref = disparityMap1.*(1-unreliable1); 
%figure();imshow(dispartity1_ref,disparityRange);colormap jet;colorbar;

dispartity2_ref = disparityMap2.*(1-unreliable2); 
%figure();imshow(dispartity2_ref,disparityRange2);colormap jet;colorbar;

%% Get rid of unrelible pixels usnig feature points and interpolation
face_lower_th = 20; %Threshold for lowest point of face from lowest feature point (in y-direction)
face_upper_th = 100; %Threshold for highest point of face from highest feature point (in y-direction)
thDisp1 = 15; %Threshold for unrelibility, increasing threshold reduces #unrelible points
[disp_int1, disp_int_only_face1] = interF (dispartity1_ref,disparityMapBM1,mask_middle, ... 
    features_m,disp_FP,disparity_feature_points, disparityRange, thDisp1,face_lower_th,face_upper_th);

face_upper_th = 100;
thDisp2 = 20; %Threshold for unrelibility, increasing threshold reduces #unrelible points
[disp_int2, disp_int_only_face2] = interF (dispartity2_ref,disparityMapBM2,mask_middle2, ... 
    features_m2,disp_FP2,disparity_feature_points2, disparityRange2, thDisp2,face_lower_th,face_upper_th);

%% Median filtering
dsp1 = disp_int1;
%Applying median filter to depth-image (disparity map)
dsp_med_filt1 =  medfilt2(dsp1,[50 50]);
% dsp_med_filt= mask_middle_croped_erode.*dsp_med_filt;
unreliable_out1 = 1-(dsp_med_filt1~=0);

dsp2 = disp_int2;
dsp_med_filt2 =  medfilt2(dsp2,[50 50]);
unreliable_out2 = 1-(dsp_med_filt2~=0);

%% Smoothing disparity maps by Gaussian filter
h=fspecial('gaussian', [10 10], 2);
dsp_gauss1 = imfilter(dsp_med_filt1,h);
dsp_gauss2 = imfilter(dsp_med_filt2,h);

%% Generate Point Clouds
xyzPoints1 = reconstructScene(dsp_gauss1,stereoParams_mr);
xyzPoints2 = reconstructScene(dsp_gauss2,stereoParams_ml);

xyzPoints11 = reconstructScene(dsp_med_filt1,stereoParams_mr);
xyzPoints22 = reconstructScene(dsp_med_filt2,stereoParams_ml);

%% Generate 3D face meshes 
mesh_create_func( img_middle_rec, dsp_gauss1, xyzPoints1, unreliable_out1);
mesh_create_func( img_middle_rec2, dsp_gauss2, xyzPoints2, unreliable_out2);

%% Merging 2 point clouds and estimate error (ICP algoritm is used)
%Create point cloud object
ptCloud1 = pointCloud(xyzPoints11);
ptCloud2 = pointCloud(xyzPoints22);

%Subsample point clouds by scale
scale= 1;
xyzPoints1_down = pcdownsample(ptCloud1,'random',scale);
xyzPoints2_down = pcdownsample(ptCloud2,'random',scale);

%Obtain rotation matrix from stereoParams
R1 = stereoParams_mr.RotationOfCamera2;
R2 = stereoParams_ml.RotationOfCamera2;

%Define Initilial Transform
tformI =  affine3d();
tformI.T(1:3,1:3) = R2*inv(R1); %R1 is 3x3 matrix so using inv() is okay!

%Transform the point cloud 1 such a way that overlay point cloud 2
[tform,movingReg,rmse,squaredError] = pcregrigid(xyzPoints1_down,xyzPoints2_down,'MaxIterations',10, ...
   'InitialTransform', tformI);

%Visiluze point clouds
% figure();pcshow(xyzPoints1_down);
% figure();pcshow(xyzPoints2_down);

%ptCloudAligned = pctransform(xyzPoints1_down,tform);

%Merge point clouds
ptCloudOut = pcmerge(movingReg,xyzPoints2_down,1);
%figure();pcshow(ptCloudOut);

%% Accuracy estimation
th_dr = 3.3;
th_dr2 = 3.3/2;
dr = sqrt(squaredError);
acc = sum(dr<th_dr)/length(dr)
acc2 = sum(dr<th_dr2)/length(dr)


