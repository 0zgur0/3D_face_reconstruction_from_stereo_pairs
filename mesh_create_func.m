function [] = mesh_create_func( img, disparityMap, xyzPoints, unreliable)
%% create a connectivity structure
[M, N] = size(disparityMap);        %get image size
res = 2;                            %resolution of mesh   
[nI,mI] = meshgrid(1:res:N,1:res:M); %create a 2D meshgrid of pixels, thus defining a resolution grid
TRI = delaunay(nI(:),mI(:));       %create a triangle connectivity list
indI = sub2ind([M,N],mI(:),nI(:));  %cast grid points to linear indices

%% linearize the arrays and adapt to chosen resolution
pcl = reshape(xyzPoints,N*M,3);
% reshape to (N*M)x3
J1l = reshape(img,N*M,3);
% reshape to (N*M)x3
pcl = pcl(indI,:);
% select 3D points that are on resolution grid
J1l = J1l(indI,:);
% select pixels that are on the resolution grid
%% remove the unreliable points and the associated triangles
ind_unreliable = find(unreliable(indI));% get the linear indices of unreliable 3D points
imem = ismember(TRI(:),ind_unreliable); % find indices of references to unreliable points
[ir,~] = ind2sub(size(TRI),find(imem)); % get the indices of rows with refs to unreliable points.
TRI(ir,:) = [];
% dispose them
iused = unique(TRI(:));
% find the ind's of vertices that are in use
used = zeros(length(pcl),1);
% pre-allocate
used(iused) = 1;
% create a map of used vertices
map2used = cumsum(used);
% conversion table from indices of old vertices to the new one
pcl = pcl(iused,:);
% remove the unused vertices
J1l = J1l(iused,:);
TRI = map2used(TRI);
% update the ind's of vertices
%% create the 3D mesh
TR = triangulation(TRI,double(pcl));
%% visualize
figure;
TM = trisurf(TR);
set(TM,'FaceVertexCData',J1l);
set(TM,'Facecolor','interp');
% set(TM,'FaceColor','red');
set(TM,'EdgeColor','none');
xlabel('x (mm)')
ylabel('y (mm)')
zlabel('z (mm)')
axis([-250 250 -250 250 400 900])
set(gca,'xdir','reverse')
set(gca,'zdir','reverse')
daspect([1,1,1])
axis tight
% create the object
% plot the mesh
% set colors to input image
% if you want a colored surface
% suppress the edges
end