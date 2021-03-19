clearvars; close all; clc; 
%%
% create true velocity model for immersed disk test case using 
% the level-set FWI method 
vp = zeros(150,150)+2.0;
[xg,yg]=meshgrid(linspace(-1.5,0,150), linspace(0,1.5,150));

xc = -0.75; 
yc = 0.75; 
r = 0.40; 

p = [xg(:),yg(:)]; 
d=sqrt((p(:,1)-xc).^2/1.0.^2+(p(:,2)-yc).^2)-r;
d = reshape(d,150,150);

vp(d < 0) = 4.5;

figure; pcolor(xg,yg,vp); shading interp
axis equal; xlabel('x-position (km)'); ylabel('y-position (km)'); 
cb=colorbar; ylabel(cb,'p-wave velocity (km/s)')
title('True velocity model'); 

% dependency: segymat
WriteSegy('immersed_disk_true_vp.segy',vp); 

%%
% create guess velocity model for immersed disk test case using 
% the level-set FWI method 
vp = zeros(150,150)+2.0;
[xg,yg]=meshgrid(linspace(-1.5,0,150), linspace(0,1.5,150));

xc = -0.75; 
yc = 0.75; 
r = 0.35; 

p = [xg(:),yg(:)]; 
d=sqrt((p(:,1)-xc).^2+(p(:,2)-yc).^2/1.0.^2)-r;
d = reshape(d,150,150);

vp(d < 0) = 4.5;

figure; pcolor(xg,yg,vp); shading interp
axis equal; xlabel('x-position (km)'); ylabel('y-position (km)'); 
cb=colorbar; ylabel(cb,'p-wave velocity (km/s)')
title('Guess velocity model'); 

WriteSegy('immersed_disk_guess_vp.segy',vp); 
