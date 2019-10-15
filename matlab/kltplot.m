clear all;
close all;

rng(0);

% hidden  parameters
s = [1,3];
H = s.*randn(256,2);
theta = 0.4*pi/4;
R = [cos(theta),sin(theta);-sin(theta),cos(theta)];
H = (R'*H')';

s = [4,0.4];
X = s.*randn(256,2);
theta = 1.0*pi/4;
R = [cos(theta),sin(theta);-sin(theta),cos(theta)];
X = (R'*X')';

% empirical statitistics
Chh = cov(H);
Cxx = cov(X);
[V,D] = eig(Chh);%,inv(Cxx));
[U,L] = eig(Cxx);

hmax = 2*ceil(sqrt(max(D(:))));
xmax = 2*ceil(sqrt(max(L(:))));
gmax = max(hmax,xmax);
[h1grid,h2grid] = meshgrid(-gmax:0.1:gmax);
grid = [h1grid(:),h2grid(:)];
[xmids,ymids] = meshgrid(-2*gmax:1:2*gmax);
mids = [xmids(:),ymids(:)];
map = 1-gray;
map = map(1:end-32,:);

figure(1);
colormap(map);
dist = reshape(mvnpdf(grid,[0,0],Chh),size(h1grid));
contourf(h1grid,h2grid,dist,0.08:-0.01:0.000,'LineColor','black');
hold on;
Y = H;
scatter(Y(:,1),Y(:,2),1,'red');
plotv(4*V,'r');
% xymids = (V'*mids')';
% [vx,vy] = voronoi(xymids(:,1),xymids(:,2));
% plot(vx,vy,'Color',0.7*[1,1,1]);
axis equal;
axis([-hmax,+hmax,-hmax,+hmax]);
xlabel('$h_1$');
ylabel('$h_2$');

figure(2);
colormap(map);
dist = reshape(mvnpdf((V'*grid')',[0,0],Chh),size(h1grid));
contourf(h1grid,h2grid,dist,0.08:-0.01:0.000,'LineColor','black');
hold on;
Y = (V'*H')';
scatter(Y(:,1),Y(:,2),1,'red');
plotv(4*inv(V')*V','r');
xymids = mids;
[vx,vy] = voronoi(xymids(:,1),xymids(:,2));
plot(vx,vy,'Color',0.7*[1,1,1]);
axis equal;
axis([-hmax,+hmax,-hmax,+hmax]);
xlabel('$c_1$');
ylabel('$c_2$');

figure(3);
colormap(map);
dist = reshape(mvnpdf(grid,[0,0],Chh),size(h1grid));
contourf(h1grid,h2grid,dist,0.08:-0.01:0.000,'LineColor','black');
hold on;
Y = H;
scatter(Y(:,1),Y(:,2),1,'red');
plotv(4*V,'r');
xymids = (V*mids')';
[vx,vy] = voronoi(xymids(:,1),xymids(:,2));
plot(vx,vy,'Color',0.7*[1,1,1]);
axis equal;
axis([-hmax,+hmax,-hmax,+hmax]);
xlabel('$h_1$');
ylabel('$h_2$');



figure(3);
colormap(map);
dist = reshape(mvnpdf(grid,[0,0],Cxx),size(h1grid));
contourf(h1grid,h2grid,dist,0.08:-0.01:0.000,'LineColor','black');
hold on;
Y = X;
scatter(Y(:,1),Y(:,2),1,'blue');
plotv(4*U,'b');
xymids = (U'*mids')';
% [ux,uy] = voronoi(xymids(:,1),xymids(:,2));
% plot(ux,uy,'Color',0.7*[1,1,1]);
axis equal;
axis([-xmax,+xmax,-xmax,+xmax]);
xlabel('$x_1$');
ylabel('$x_2$');

figure(2);
colormap(map);




