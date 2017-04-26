function [F] = face( sigma,M,N,win_size,features_m,disparity_feature_points,dispartity1_ref,mask_middle)

w=(win_size-1)/2;

gaussians = zeros(M,N,74);
for i=1:68
    x=uint32(features_m(i,1));
    y=uint32(features_m(i,2));
    gaussians(y-w:y+w,x-w:x+w,i) = fspecial('gauss',win_size,20);
end

c1 = uint32(features_m(2,:) + (features_m(30,:)-features_m(2,:))/3);
c2 = uint32(features_m(4,:) + (features_m(32,:)-features_m(4,:))/2);
c3 = uint32(features_m(36,:) + (features_m(14,:)-features_m(36,:))/2);
c4 = uint32(features_m(30,:) + (features_m(16,:)-features_m(30,:))/3);
c5 = uint32(features_m(2,:) + 2*(features_m(30,:)-features_m(2,:))/3);
c6 = uint32(features_m(30,:) + 2*(features_m(16,:)-features_m(30,:))/3);

gaussians(c1(2)-w:c1(2)+w,c1(1)-w:c1(1)+w,69) = fspecial('gauss',win_size,sigma(1,1));
gaussians(c2(2)-w:c2(2)+w,c2(1)-w:c2(1)+w,70) = fspecial('gauss',win_size,sigma(1,1));
gaussians(c3(2)-w:c3(2)+w,c3(1)-w:c3(1)+w,71) = fspecial('gauss',win_size,sigma(1,1));
gaussians(c4(2)-w:c4(2)+w,c4(1)-w:c4(1)+w,72) = fspecial('gauss',win_size,sigma(1,1));
gaussians(c5(2)-w:c5(2)+w,c5(1)-w:c5(1)+w,73) = fspecial('gauss',win_size,sigma(1,1));
gaussians(c6(2)-w:c6(2)+w,c6(1)-w:c6(1)+w,74) = fspecial('gauss',win_size,sigma(1,1));

disp=zeros(74,1);
disp(1:68,1) = disparity_feature_points;
w1=dispartity1_ref(c1(2)-w:c1(2)+w,c1(1)-w:c1(1)+w);
w2=dispartity1_ref(c2(2)-w:c2(2)+w,c2(1)-w:c2(1)+w);
w3=dispartity1_ref(c3(2)-w:c3(2)+w,c3(1)-w:c3(1)+w);
w4=dispartity1_ref(c4(2)-w:c4(2)+w,c4(1)-w:c4(1)+w);
w5=dispartity1_ref(c5(2)-w:c5(2)+w,c5(1)-w:c5(1)+w);
w6=dispartity1_ref(c6(2)-w:c6(2)+w,c6(1)-w:c6(1)+w);

disp(69,1) = sum(sum(w1))/sum(sum(w1~=0));
disp(70,1) = sum(sum(w2))/sum(sum(w2~=0));
disp(71,1) = sum(sum(w3))/sum(sum(w3~=0));
disp(72,1) = sum(sum(w4))/sum(sum(w4~=0));
disp(73,1) = sum(sum(w5))/sum(sum(w5~=0));
disp(74,1) = sum(sum(w6))/sum(sum(w6~=0));


tot=sum(gaussians,3);
for i=1:74 
    gaussians(:,:,i)=gaussians(:,:,i)./tot;
end

F =  zeros(M,N);
fo=zeros(74,1);
for k=1:M
    for l=1:N
        if mask_middle(k,l)==1
            fo(:,1)=gaussians(k,l,:);
            F(k,l) = sum(fo.*disp);
        end
    end 
end

%Smooth 
% h=fspecial('gauss',100,20);
% F=imfilter(F,h);

%Cost function
%J = sum(sum((dispartity1_ref_out - F).^2));

end

