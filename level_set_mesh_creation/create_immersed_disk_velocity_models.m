clearvars; close all; clc; 
%%
% create true velocity model for immersed disk test case using 
% the level-set FWI method 
%vp = zeros(150,150)+2.0; 


% make piecewise smooth

[xg,yg]=meshgrid(linspace(0,1.0,100),linspace(-0.65,0,65));

C0 = 1.5; 
M = 4.0; 
vp = C0 + M*abs(linspace(-0.65,0,65)); 
vp=repmat(vp',[1,100]);


xc = 0.50; 
yc = -0.325; 
r = 0.15; 

p = [xg(:),yg(:)]; 
d=sqrt((p(:,1)-xc).^2/1.0.^2+(p(:,2)-yc).^2)-r;
d = reshape(d,65,100);

vp(d < 0) = 4.5;

figure; pcolor(xg,yg,vp); shading interp
axis equal; xlabel('x-position (km)'); ylabel('y-position (km)'); 
cb=colorbar; ylabel(cb,'p-wave velocity (km/s)')
title('True velocity model'); 

% dependency: segymat
WriteSegy('immersed_disk_true_vp.segy',flipud(vp)); 

%%
% create guess velocity model for immersed disk test case using 
% the level-set FWI method 
%vp = zeros(150,150)+2.0;
%vp=repmat(linspace(1.5,4.0,65)',[1,100]);

[xg,yg]=meshgrid(linspace(0,1.0,100),linspace(-0.65,0,65));

C0 = 1.5; 
M = 4.0; 
vp = C0 + M*abs(linspace(-0.65,0,65)); 
vp=repmat(vp',[1,100]);

xc = 0.50; 
yc = -0.325; 
r = 0.10; 

p = [xg(:),yg(:)]; 
d=sqrt((p(:,1)-xc).^2/2.0+(p(:,2)-yc).^2/1.0.^2)-r;
d = reshape(d,65,100);

vp(d < 0) = 4.5;

figure; pcolor(xg,yg,vp); shading interp
axis equal; xlabel('x-position (km)'); ylabel('y-position (km)'); 
cb=colorbar; ylabel(cb,'p-wave velocity (km/s)')
title('Guess velocity model'); 

WriteSegy('immersed_disk_guess_vp.segy',flipud(vp)); 
