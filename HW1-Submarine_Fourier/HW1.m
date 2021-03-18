clear; close all; clc;

load subdata.mat;
L = 10;
n = 64;
x2 = linspace(-L,L,n+1);
x = x2(1:n);
y = x;
z = x;
k = (2*pi/(2*L))*[0:(n/2-1), -n/2:-1];
ks = fftshift(k);

[X,Y,Z]=meshgrid(x,y,z);
[Kx,Ky,Kz]=meshgrid(ks,ks,ks);

Rt = cell(1,49);
ave = zeros(n,n,n);
for j=1:49
    Un(:,:,:) = reshape(subdata(:,j),n,n,n);
    Unt = fftshift(fftn(Un));
    ave = ave + Unt;
    Rt{j} = Unt;
end
ave = abs(ave)/49;
[M,I] = max(ave,[],'all','linear');
[Ix,Iy,Iz] = ind2sub(size(ave),I);
Kx(Ix,Iy,Iz)
Ky(Ix,Iy,Iz)
Kz(Ix,Iy,Iz)

a = 0.5;
filterx = exp(-a.*(Kx-Kx(Ix,Iy,Iz)).^2);
filtery = exp(-a.*(Ky-Ky(Ix,Iy,Iz)).^2);
filterz = exp(-a.*(Kz-Kz(Ix,Iy,Iz)).^2);
filter = filterx.*filtery.*filterz;

locs = zeros(3,49);
for j=1:49
    Rfn = abs(ifftn(Rt{j}.*filter));
    [M,I] = max(Rfn,[],'all','linear');
    [Ix,Iy,Iz] = ind2sub(size(Rfn),I);
    locs(:,j) = [X(Ix,Iy,Iz),Y(Ix,Iy,Iz),Z(Ix,Iy,Iz)];
end

writematrix(locs,'locations.csv')
plot3(locs(1,:),locs(2,:),locs(3,:));
xlim([-10,10]); ylim([-10,10]); zlim([-10,10]);
xlabel("x"); ylabel("y"); zlabel("z");
grid on;
hold on;
plot3(10*ones(size(locs(1,:))),locs(2,:),locs(3,:));
plot3(locs(1,:),10*ones(size(locs(2,:))),locs(3,:));
plot3(locs(1,:),locs(2,:),-10*ones(size(locs(3,:))));
legend("Submarine Trajectory", "y-z projection", "x-z projection", "x-y projection");
title("Submarine Trajectory")