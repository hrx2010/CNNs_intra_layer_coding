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

s = [1.7,0.6]*0.8;
X = s.*randn(512,2);
theta = 0.2*pi/4;
R = [cos(theta),sin(theta);-sin(theta),cos(theta)];
X = (R'*X')';

% empirical statitistics
Chh = cov(H);
Cxx = 0.5*(cov(X)+cov(X)');
[V,D] = eig(Chh);
% [U,L] = eig(Cxx);
hmax = 2*ceil(sqrt(max(D(:))));
xmax = 2*ceil(sqrt(max(L(:))));
gmax = max(hmax,xmax);

if isgklt
    [V,D] = eig(Chh,0.5*(inv(Cxx)+inv(Cxx)'));
    V = inv(V');
    V = inv(V./sqrt(sum(V.^2)))';
end

[h_1,h_2] = meshgrid(-gmax:0.1:gmax);
% h12 = [h_1(:),h_2(:)];
[xmids,ymids] = meshgrid(-2*gmax:1:2*gmax);
mids = [xmids(:),ymids(:)];
map = 1-gray;
map = map(1:end-32,:);

figure(1);
set(gcf,'Color','none');
colormap(map);
dist = reshape(mvnpdf([h_1(:),h_2(:)],[0,0],Chh),size(h_1));
contourf(h_1,h_2,dist,0.08:-0.01:0.000,'LineColor',0.25*[1,1,1]);
hold on;
Y = H;
s = scatter(Y(:,1),Y(:,2),1,0.25*[1,1,1],'filled','SizeData',1.5);
alpha(s,0.5);
plotv(6*inv(V'),'r');
% xymids = (V'*mids')';
% [vx,vy] = voronoi(xymids(:,1),xymids(:,2));
% plot(vx,vy,'Color',0.7*[1,1,1]);
hold off;
axis equal;
axis([-hmax,+hmax,-hmax,+hmax]);
xticks(-6:3:6);
labels = xticklabels;
labels{end} = '$\theta_1$'
xticklabels(labels);
yticks(-6:3:6);
labels{end} = '$\theta_2$'
yticklabels(labels);
pdfprint('temp.pdf','Height',8.5,'Width',8.5,'Position',[1.5,1.5,6.5,6.5]);

figure(2);
set(gcf,'Color','none');
colormap(map);
dist = reshape(mvnpdf([h_1(:),h_2(:)],[0,0],Cxx),size(h_1));
contourf(h_1,h_2,dist,[0.16:-0.04:0.04,0.02],'LineColor',0.25*[1,1,1]);
hold on;
Y = X;
s = scatter(Y(:,1),Y(:,2),1,0.25*[1,1,1],'filled','SizeData',1.5);
alpha(s,0.5);
plotv(4*V,'r');
% xymids = (V'*mids')';
% [vx,vy] = voronoi(xymids(:,1),xymids(:,2));
% plot(vx,vy,'Color',0.7*[1,1,1]);
hold off;
axis equal;
axis([-xmax,+xmax,-xmax,+xmax]);
xlabel('$x_1$');
ylabel('$x_2$');
xticks(-4:2:4);
labels = xticklabels;
labels{end} = '$\gamma_1$'
xticklabels(labels);
yticks(-4:2:4);
labels = yticklabels;
labels{end} = '$\gamma_2$'
yticklabels(labels);
pdfprint('temp.pdf','Height',8.5,'Width',8.5,'Position',[1.5,1.5,6.5,6.5]);

figure(2);
set(gcf,'Color','none');
colormap(map);
dist = reshape(mvnpdf([h_1(:),h_2(:)],[0,0],Chh),size(h_1));
h_p = reshape((V'*[h_1(:),h_2(:)]')',[size(h_1),2]);
contourf(h_p(:,:,1),h_p(:,:,2),dist,0.08:-0.01:0.000, ...
         'LineColor',0.25*[1,1,1]);
hold on;
Y = round((V'*H')');
s = scatter(Y(:,1),Y(:,2),1,0.25*[1,1,1],'filled','SizeData',1.5);
alpha(s,0.5);
xymids = mids;
[vx,vy] = voronoi(xymids(:,1),xymids(:,2));
p = plot(vx,vy,'Color',0.8*[1,1,1]);
 plotv(5*inv(V')*V','r');
hold off;
axis equal;
axis([-hmax,+hmax,-hmax,+hmax]);
xticks(-6:3:+6);
labels = xticklabels;
labels{end} = '$t_1$';
xticklabels(labels);
yticks(-6:3:+6);
labels = yticklabels;
labels{end} = '$t_2$';
yticklabels(labels);
% xlabel('$w_1$');
% ylabel('$w_2$');
xticks(-6:3:6);
yticks(-6:3:6);
pdfprint('temp.pdf','Height',8.5,'Width',8.5,'Position',[1.5,1.5,6.5,6.5]);

figure(3);
set(gcf,'Color','none');
colormap(map);
dist = reshape(mvnpdf([h_1(:),h_2(:)],[0,0],Chh),size(h_1));
contourf(h_1,h_2,dist,0.08:-0.01:0.000,'LineColor',0.25*[1,1,1]);
hold on;
Y = (inv(V')*round(V'*H'))';
s = scatter(Y(:,1),Y(:,2),1,0.25*[1,1,1],'filled','SizeData',1.5);
alpha(s,0.5);
plotv(6*inv(V'),'r');
xymids = mids;%(inv(V')*mids')';
[vx,vy] = voronoi(xymids(:,1),xymids(:,2));
v1 = inv(V')*[vx(1,:);vy(1,:)];
v2 = inv(V')*[vx(2,:);vy(2,:)];
vx = [v1(1,:);v2(1,:)];
vy = [v1(2,:);v2(2,:)];
plot(vx,vy,'Color',0.8*[1,1,1]);
axis equal;
axis([-hmax,+hmax,-hmax,+hmax]);
xticks(-6:3:6);
labels = xticklabels;
labels{end} = '$\theta_1$'
xticklabels(labels);
yticks(-6:3:6);
labels{end} = '$\theta_2$'
yticklabels(labels);
pdfprint('temp.pdf','Height',8.5,'Width',8.5,'Position',[1.5,1.5,6.5,6.5]);

figure(4);
set(gcf,'Color','none');
colormap(map);
dist = reshape(mvnpdf([h_1(:),h_2(:)],[0,0],Chh),size(h_1));
h_p = reshape((sqrt(L)*U'*[h_1(:),h_2(:)]')',[size(h_1),2]);
contourf(h_p(:,:,1),h_p(:,:,2),dist,0.08:-0.01:0.000,'LineColor',0.25*[1,1,1]);
hold on;
%(inv(V')*round(V'*H'))'
Y = (sqrt(L)*U'*inv(V')*round(V'*H'))';
s = scatter(Y(:,1),Y(:,2),1,0.25*[1,1,1],'filled','SizeData',1.5);
alpha(s,0.5);
xymids = mids;%(inv(V')*mids')';
[vx,vy] = voronoi(xymids(:,1),xymids(:,2));
v1 = sqrt(L)*U'*inv(V')*[vx(1,:);vy(1,:)];
v2 = sqrt(L)*U'*inv(V')*[vx(2,:);vy(2,:)];
vx = [v1(1,:);v2(1,:)];
vy = [v1(2,:);v2(2,:)];
plot(vx,vy,'Color',0.7*[1,1,1]);
plotv(5*sqrt(L)*U'*inv(V'),'r');
hold off;
axis equal;

axis([-4,+4,-4,+4]);
xticks(-4:2:4);
labels = xticklabels;
labels{end} = '$y_1$'
xticklabels(labels);
yticks(-4:2:4);
labels{end} = '$y_2$'
yticklabels(labels);
pdfprint('temp.pdf','Height',8.5,'Width',8.5,'Position',[1.5,1.5,6.5,6.5]);

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





