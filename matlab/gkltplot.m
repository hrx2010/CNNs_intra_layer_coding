clear all;
close all;

isgklt = true;
rng(0);

% hidden  parameters
s = [1,3];
H = s.*randn(512,2);
theta = 0.6*pi/4;
R = [cos(theta),sin(theta);-sin(theta),cos(theta)];
H = (R'*H')';

s = [1.7,0.8]*0.8;
X = s.*randn(512,2);
theta = 0.2*pi/4;
R = [cos(theta),sin(theta);-sin(theta),cos(theta)];
X = (R'*X')';

% empirical statitistics
Chh = cov(H);
Cxx = 0.5*(cov(X)+cov(X)');
[V,D] = eig(Chh);
[U,L] = eig(Cxx);
hmax = 2*ceil(sqrt(max(D(:))));
xmax = 2*ceil(sqrt(max(L(:))));
gmax = max(hmax,xmax);

if isgklt
    [V,D] = eig(Chh,0.5*(inv(Cxx)+inv(Cxx)'));
end

[h_1,h_2] = meshgrid(-gmax:0.1:gmax);
% h12 = [h_1(:),h_2(:)];
[xmids,ymids] = meshgrid(-2*gmax:1:2*gmax);
mids = [xmids(:),ymids(:)];
map = 1-gray;
map = map(1:end-32,:);

figure(1);
colormap(map);
dist = reshape(mvnpdf([h_1(:),h_2(:)],[0,0],Chh),size(h_1));
contourf(h_1,h_2,dist,0.08:-0.01:0.000,'LineColor',0.25*[1,1,1]);
hold on;
Y = H;
%s = scatter(Y(:,1),Y(:,2),1,'r','filled','SizeData',1.5);
%alpha(s,0.5);
plotv(4*inv(V'),'k');
% xymids = (V'*mids')';
% [vx,vy] = voronoi(xymids(:,1),xymids(:,2));
% plot(vx,vy,'Color',0.7*[1,1,1]);
hold off;
axis equal;
axis([-hmax,+hmax,-hmax,+hmax]);
% xlabel('$h_1$');
% ylabel('$h_2$');
xticks(-6:3:6);
yticks(-6:3:6);
pdfprint('temp.pdf','Height',9,'Width',9,'Position',[1.5,1.5,7,7]);

figure(2);
colormap(map);
dist = reshape(mvnpdf([h_1(:),h_2(:)],[0,0],Cxx),size(h_1));
contourf(h_1,h_2,dist,[0.16:-0.04:0.04,0.02],'LineColor',0.25*[1,1,1]);
hold on;
Y = X;
%s = scatter(Y(:,1),Y(:,2),1,'b','filled','SizeData',1.5);
%alpha(s,0.5);
plotv(4*inv(U'),'k');
% xymids = (V'*mids')';
% [vx,vy] = voronoi(xymids(:,1),xymids(:,2));
% plot(vx,vy,'Color',0.7*[1,1,1]);
hold off;
axis equal;
axis([-xmax,+xmax,-xmax,+xmax]);
xlabel('$x_1$');
ylabel('$x_2$');
xticks(-4:2:4);
yticks(-4:2:4);
pdfprint('temp.pdf','Height',9,'Width',9,'Position',[1.5,1.5,7,7]);

figure(2);
colormap(map);
dist = reshape(mvnpdf([h_1(:),h_2(:)],[0,0],Chh),size(h_1));
h_p = reshape((V'*[h_1(:),h_2(:)]')',[size(h_1),2]);
contourf(h_p(:,:,1),h_p(:,:,2),dist,0.08:-0.01:0.000, ...
         'LineColor',0.25*[1,1,1]);
hold on;
Y = (V'*H')';
%s = scatter(Y(:,1),Y(:,2),1,'r','filled','SizeData',1.5);
%alpha(s,0.5);
plotv(4*inv(V')*V','k');
xymids = mids;
[vx,vy] = voronoi(xymids(:,1),xymids(:,2));
p = plot(vx,vy,'Color',0.8*[1,1,1]);
hold off;
axis equal;
axis([-hmax,+hmax,-hmax,+hmax]);
xlabel('$w_1$');
ylabel('$w_2$');
xticks(-6:3:6);
yticks(-6:3:6);
pdfprint('temp.pdf','Height',9,'Width',9,'Position',[1.5,1.5,7,7]);

figure(3);
colormap(map);
dist = reshape(mvnpdf([h_1(:),h_2(:)],[0,0],Chh),size(h_1));
contourf(h_1,h_2,dist,0.08:-0.01:0.000,'LineColor',0.25*[1,1,1]);
hold on;
Y = H;
%s = scatter(Y(:,1),Y(:,2),1,'r','filled','SizeData',1.5);
%alpha(s,0.5);
plotv(4*inv(V'),'k');
xymids = mids;%(inv(V')*mids')';
[vx,vy] = voronoi(xymids(:,1),xymids(:,2));
v1 = inv(V')*[vx(1,:);vy(1,:)];
v2 = inv(V')*[vx(2,:);vy(2,:)];
vx = [v1(1,:);v2(1,:)];
vy = [v1(2,:);v2(2,:)];
plot(vx,vy,'Color',0.8*[1,1,1]);
axis equal;
axis([-hmax,+hmax,-hmax,+hmax]);
xlabel('$h_1$');
ylabel('$h_2$');
xticks(-6:3:6);
yticks(-6:3:6);
pdfprint('temp.pdf','Height',9,'Width',9,'Position',[1.5,1.5,7,7]);

figure(4);
colormap(map);
dist = reshape(mvnpdf([h_1(:),h_2(:)],[0,0],Chh),size(h_1));
h_p = reshape((sqrt(L)*U'*[h_1(:),h_2(:)]')',[size(h_1),2]);
contourf(h_p(:,:,1),h_p(:,:,2),dist,0.08:-0.01:0.000,'LineColor',0.25*[1,1,1]);
hold on;
Y = (sqrt(L)*U'*H')';
% s = scatter(Y(:,1),Y(:,2),1,'m','filled','SizeData',1.5);
% alpha(s,0.5);
plotv(4*sqrt(L)*U'*inv(V'),'k');
xymids = mids;%(inv(V')*mids')';
[vx,vy] = voronoi(xymids(:,1),xymids(:,2));
v1 = sqrt(L)*U'*inv(V')*[vx(1,:);vy(1,:)];
v2 = sqrt(L)*U'*inv(V')*[vx(2,:);vy(2,:)];
vx = [v1(1,:);v2(1,:)];
vy = [v1(2,:);v2(2,:)];
plot(vx,vy,'Color',0.7*[1,1,1]);
hold off;
axis equal;

axis([-hmax,+hmax,-hmax,+hmax]);
xlabel('$y_1$');
ylabel('$y_2$');
xticks(-6:3:6);
yticks(-6:3:6);
pdfprint('temp.pdf','Height',9,'Width',9,'Position',[1.5,1.5,7,7]);

% figure(3);
% colormap(map);
% dist = reshape(mvnpdf(h12,[0,0],Cxx),size(h_1));
% contourf(h_1,h_2,dist,0.08:-0.01:0.000,'LineColor','black');
% hold on;
% Y = X;
% scatter(Y(:,1),Y(:,2),1,'blue');
% plotv(4*U,'b');
% xymids = (U'*mids')';
% % [ux,uy] = voronoi(xymids(:,1),xymids(:,2));
% % plot(ux,uy,'Color',0.7*[1,1,1]);
% axis equal;
% axis([-xmax,+xmax,-xmax,+xmax]);
% xlabel('$x_1$');
% ylabel('$x_2$');

% figure(2);
% colormap(map);





