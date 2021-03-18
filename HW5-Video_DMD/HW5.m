clear all; close all; clc;

%% Load and Prepare Videos
% obj = VideoReader('monte_carlo_low.mp4');
obj = VideoReader('ski_drop_low.mp4');
frameRate = obj.FrameRate;
duration = obj.Duration;
video = read(obj);

X = zeros(size(video,1),size(video,2),size(video,4));
for i = 1:size(X,3)
    X(:,:,i) = rgb2gray(im2double(video(:,:,:,i)));
end
dim = size(X);
X = reshape(X,[size(X,1)*size(X,2),size(X,3)])';
X = unique(X,'stable','rows');
X = X';
dim(3) = size(X,2);
clear obj; clear video;

%% Perform Low Rank DMD
approx = 1;
t = 0:1/frameRate:(dim(3)-1)/frameRate;
dt = t(2)-t(1);
X1 = X(:,1:end-1);
X2 = X(:,2:end);

%%
[U,Sig,V] = svd(X1,'econ');
STilde = U'*X2*V*diag(1./diag(Sig));
[eVec,eVal] = eig(STilde);
mu = diag(eVal);
clear X2; clear V; clear Sig; clear eVal; clear STilde;
omega = log(mu)/dt;
clear mu;
ind = (abs(omega) < approx);
omega = omega(ind);
phi = U*eVec;
y0 = phi\X1(:,1);
y0 = y0(ind);

u_modes = zeros(length(y0),length(t));
for j = 1:length(t)
   u_modes(:,j) = y0.*exp(omega*t(j)); 
end
clear mu; clear omega; clear U; clear V; clear X1; clear X2; clear eVec;
X_lowrank = phi(:,ind)*u_modes;
clear phi; clear u_modes; clear y0;

X_sparse = X - abs(X_lowrank);
clear X;

R = X_sparse.*(X_sparse < 0);
X_lowrank = R + abs(X_lowrank);
X_sparse = X_sparse - R;
X_lowrank = reshape(X_lowrank,dim);
X_sparse = reshape(X_sparse,dim);

%% View Videos
implay(X_sparse);
