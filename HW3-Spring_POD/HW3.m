%% Load Test 1
clear; close all; clc;
load('cam1_1.mat');
load('cam2_1.mat');
load('cam3_1.mat');

vids = cell(1,3); 
vids{1} = vidFrames1_1(:,:,:,11:211);
vids{2} = vidFrames2_1(:,:,:,20:220);
vids{3} = vidFrames3_1(:,:,:,11:211);
numFrames = size(vids{1},4)-1;

%% Analyse Test 1
pos = zeros(6,numFrames);
x = 1:640;
y = 1:480;
filter = exp(-0.000006*(y' - 240).^2)*exp(-0.000006*(x - 320).^2);

for i = 1:3
    for j = 1:numFrames
        X = im2double(rgb2gray(vids{i}(:,:,:,j+1)));
        X = conv2(X.*filter,ones(15)*(1/15^2));
        [M,I] = max(X,[],'all','linear');
        [row,col] = ind2sub(size(X),I);
        if M > 0.3
            pos(i*2-1:i*2,j) = [row;col];
        elseif j>1
            pos(i*2-1:i*2,j) = pos(i*2-1:i*2,j-1);
        else
            pos(i*2-1:i*2,j) = [0;0];
        end
    end
end

%% Plot Test 1
figure(1)
subplot(3,1,1)
plot(1:numFrames,pos(1,:));
title('Position of Mass (Test 1)');
hold on;
plot(1:numFrames,pos(2,:));
xlabel('time (frames)');
ylabel('position (camera 1)');
legend('x position','y position');
subplot(3,1,2)
plot(1:numFrames,pos(3,:));
hold on;
plot(1:numFrames,pos(4,:));
xlabel('time (frames)');
ylabel('position (camera 2)');
subplot(3,1,3)
plot(1:numFrames,pos(5,:));
hold on;
plot(1:numFrames,pos(6,:));
xlabel('time (frames)');
ylabel('position (camera 3)');
print('tracking_1','-dpng');

[U,S,V] = svd(pos,'econ');

sig = diag(S);
figure(2);
subplot(2,2,1);
plot(sig,'ko','Linewidth',2);
title('Singular Values and Corresponding Energy (Test 1)');
ylabel('Singular Value');
subplot(2,2,2);
semilogy(sig,'ko','Linewidth',2)
ylabel('Singular Value (log scale)');
subplot(2,2,3);
plot(sig.^2/sum(sig.^2),'ko','Linewidth',2);
xlabel('Singular Value');
ylabel('Energy');
subplot(2,2,4);
semilogy(sig.^2/sum(sig.^2),'ko','Linewidth',2);
ylabel('Energy (log scale)');
xlabel('Singular Value');
print('svals_1','-dpng');

f = 1:6;
t = 1:numFrames;
figure(3);
subplot(2,1,1);
plot(f,U(:,1),'b',f,U(:,2),'--r',f,U(:,3),':k','Linewidth',2);
title('POD Modes (Test 1)');
xlabel('x');
legend('mode 1','mode 2','mode 3','Location','northwest');
subplot(2,1,2);
plot(t,V(:,1),'b',t,V(:,2),'--r',t,V(:,3),':k','Linewidth',2);
xlabel('t');
ylim([-0.1 0]);
legend('v_1','v_2','v_3','Location','northwest');
print('modes_1','-dpng');

f_rank1 = U(:,1)*S(1,1)*V(:,1)';
figure(4);
plot(1:numFrames,f_rank1(1,:));
title('Rank 1 Approximation (Test 1)');
xlabel('time (frames)');
ylabel('position');
print('projection_1','-dpng');

%% Load Test 2
clear; close all; clc;
load('cam1_2.mat');
load('cam2_2.mat');
load('cam3_2.mat');

vids = cell(1,3); 
vids{1} = vidFrames1_2(:,:,:,14:314);
vids{2} = vidFrames2_2(:,:,:,2:302);
vids{3} = vidFrames3_2(:,:,:,18:318);
numFrames = size(vids{1},4)-1;

%% Analyse Test 2
pos = zeros(6,numFrames);
x = 1:640;
y = 1:480;
filters = cell(1,3);
filters{1} = exp(-0.00001*(y' - 300).^2)*exp(-0.00008*(x - 350).^2);
filters{2} = exp(-0.00001*(y' - 250).^2)*exp(-0.00001*(x - 320).^2);
filters{3} = exp(-0.00008*(y' - 250).^2)*exp(-0.00001*(x - 300).^2);

for i = 1:3
    for j = 1:numFrames
        X = im2double(rgb2gray(vids{i}(:,:,:,j+1)));
        X = conv2(X.*filters{i},ones(15)*(1/15^2));
        [M,I] = max(X,[],'all','linear');
        [row,col] = ind2sub(size(X),I);
        if M > 0.3
            pos(i*2-1:i*2,j) = [row;col];
        elseif j>1
            pos(i*2-1:i*2,j) = pos(i*2-1:i*2,j-1);
        else
            pos(i*2-1:i*2,j) = [0;0];
        end
    end
end

%% Plot Test 2
figure(1)
subplot(3,1,1)
plot(1:numFrames,pos(1,:));
title('Position of Mass (Test 2)');
hold on;
plot(1:numFrames,pos(2,:));
xlabel('time (frames)');
ylabel('position (camera 1)');
legend('x position','y position');
subplot(3,1,2)
plot(1:numFrames,pos(3,:));
hold on;
plot(1:numFrames,pos(4,:));
xlabel('time (frames)');
ylabel('position (camera 2)');
subplot(3,1,3)
plot(1:numFrames,pos(5,:));
hold on;
plot(1:numFrames,pos(6,:));
xlabel('time (frames)');
ylabel('position (camera 3)');
print('tracking_2','-dpng');

[U,S,V] = svd(pos,'econ');

sig = diag(S);
figure(2);
subplot(2,2,1);
plot(sig,'ko','Linewidth',2);
title('Singular Values and Corresponding Energy (Test 2)');
ylabel('Singular Value');
subplot(2,2,2);
semilogy(sig,'ko','Linewidth',2)
ylabel('Singular Value (log scale)');
subplot(2,2,3);
plot(sig.^2/sum(sig.^2),'ko','Linewidth',2);
xlabel('Singular Value');
ylabel('Energy');
subplot(2,2,4);
semilogy(sig.^2/sum(sig.^2),'ko','Linewidth',2);
ylabel('Energy (log scale)');
xlabel('Singular Value');
print('svals_2','-dpng');

f = 1:6;
t = 1:numFrames;
figure(3);
subplot(2,1,1);
plot(f,U(:,1),'b',f,U(:,2),'--r',f,U(:,3),':k','Linewidth',2);
title('POD Modes (Test 2)');
xlabel('x');
legend('mode 1','mode 2','mode 3','Location','northwest');
subplot(2,1,2);
plot(t,V(:,1),'b',t,V(:,2),'--r',t,V(:,3),':k','Linewidth',2);
xlabel('t');
ylim([-0.075 -0.025]);
legend('v_1','v_2','v_3','Location','northwest');
print('modes_2','-dpng');

f_rank1 = U(:,1)*S(1,1)*V(:,1)';
figure(4);
plot(1:numFrames,f_rank1(1,:));
title('Rank 1 Approximation (Test 2)');
xlabel('time (frames)');
ylabel('position');
print('projection_2','-dpng');

%% Load Test 3
clear; close all; clc;
load('cam1_3.mat');
load('cam2_3.mat');
load('cam3_3.mat');

vids = cell(1,3); 
vids{1} = vidFrames1_3(:,:,:,38:238);
vids{2} = vidFrames2_3(:,:,:,25:225);
vids{3} = vidFrames3_3(:,:,:,35:235);
numFrames = size(vids{1},4)-1;

%% Analyse Test 3
pos = zeros(6,numFrames);
x = 1:640;
y = 1:480;
filter = exp(-0.000005*(y' - 240).^2)*exp(-0.000005*(x - 320).^2);

for i = 1:3
    for j = 1:numFrames
        X = im2double(rgb2gray(vids{i}(:,:,:,j+1)));
        X = conv2(X.*filter,ones(15)*(1/15^2));
        [M,I] = max(X,[],'all','linear');
        [row,col] = ind2sub(size(X),I);
        if M > 0.3
            pos(i*2-1:i*2,j) = [row;col];
        elseif j>1
            pos(i*2-1:i*2,j) = pos(i*2-1:i*2,j-1);
        else
            pos(i*2-1:i*2,j) = [0;0];
        end
    end
end

%% Plot Test 3
figure(1)
subplot(3,1,1)
plot(1:numFrames,pos(1,:));
title('Position of Mass (Test 3)');
hold on;
plot(1:numFrames,pos(2,:));
xlabel('time (frames)');
ylabel('position (camera 1)');
legend('x position','y position');
subplot(3,1,2)
plot(1:numFrames,pos(3,:));
hold on;
plot(1:numFrames,pos(4,:));
xlabel('time (frames)');
ylabel('position (camera 2)');
subplot(3,1,3)
plot(1:numFrames,pos(5,:));
hold on;
plot(1:numFrames,pos(6,:));
xlabel('time (frames)');
ylabel('position (camera 3)');
print('tracking_3','-dpng');

[U,S,V] = svd(pos,'econ');

sig = diag(S);
figure(2);
subplot(2,2,1);
plot(sig,'ko','Linewidth',2);
title('Singular Values and Corresponding Energy (Test 3)');
ylabel('Singular Value');
subplot(2,2,2);
semilogy(sig,'ko','Linewidth',2)
ylabel('Singular Value (log scale)');
subplot(2,2,3);
plot(sig.^2/sum(sig.^2),'ko','Linewidth',2);
xlabel('Singular Value');
ylabel('Energy');
subplot(2,2,4);
semilogy(sig.^2/sum(sig.^2),'ko','Linewidth',2);
ylabel('Energy (log scale)');
xlabel('Singular Value');
print('svals_3','-dpng');

f = 1:6;
t = 1:numFrames;
figure(3);
subplot(2,1,1);
plot(f,U(:,1),'b',f,U(:,2),'--r',f,U(:,3),':k','Linewidth',2);
title('POD Modes (Test 3)');
xlabel('x');
legend('mode 1','mode 2','mode 3','Location','northwest');
subplot(2,1,2);
plot(t,V(:,1),'b',t,V(:,2),'--r',t,V(:,3),':k','Linewidth',2);
xlabel('t');
ylim([-0.1 0]);
legend('v_1','v_2','v_3','Location','northwest');
print('modes_3','-dpng');

f_rank1 = U(:,1)*S(1,1)*V(:,1)';
figure(4);
plot(1:numFrames,f_rank1(1,:));
title('Rank 1 Approximation (Test 3)');
xlabel('time (frames)');
ylabel('position');
print('projection_3','-dpng');

%% Load Test 4
clear; close all; clc;
load('cam1_4.mat');
load('cam2_4.mat');
load('cam3_4.mat');

vids = cell(1,3); 
vids{1} = vidFrames1_4(:,:,:,38:238);
vids{2} = vidFrames2_4(:,:,:,25:225);
vids{3} = vidFrames3_4(:,:,:,35:235);
numFrames = size(vids{1},4)-1;

%% Analyse Test 4
pos = zeros(6,numFrames);
x = 1:640;
y = 1:480;
filter = exp(-0.000005*(y' - 240).^2)*exp(-0.000005*(x - 320).^2);

for i = 1:3
    for j = 1:numFrames
        X = im2double(rgb2gray(vids{i}(:,:,:,j+1)));
        X = conv2(X.*filter,ones(15)*(1/15^2));
        [M,I] = max(X,[],'all','linear');
        [row,col] = ind2sub(size(X),I);
        if M > 0.3
            pos(i*2-1:i*2,j) = [row;col];
        elseif j>1
            pos(i*2-1:i*2,j) = pos(i*2-1:i*2,j-1);
        else
            pos(i*2-1:i*2,j) = [0;0];
        end
    end
end

%% Plot Test 4
figure(1)
subplot(3,1,1)
plot(1:numFrames,pos(1,:));
title('Position of Mass (Test 4)');
hold on;
plot(1:numFrames,pos(2,:));
xlabel('time (frames)');
ylabel('position (camera 1)');
legend('x position','y position');
subplot(3,1,2)
plot(1:numFrames,pos(3,:));
hold on;
plot(1:numFrames,pos(4,:));
xlabel('time (frames)');
ylabel('position (camera 2)');
subplot(3,1,3)
plot(1:numFrames,pos(5,:));
hold on;
plot(1:numFrames,pos(6,:));
xlabel('time (frames)');
ylabel('position (camera 3)');
print('tracking_4','-dpng');

[U,S,V] = svd(pos,'econ');

sig = diag(S);
figure(2);
subplot(2,2,1);
plot(sig,'ko','Linewidth',2);
title('Singular Values and Corresponding Energy (Test 4)');
ylabel('Singular Value');
subplot(2,2,2);
semilogy(sig,'ko','Linewidth',2)
ylabel('Singular Value (log scale)');
subplot(2,2,3);
plot(sig.^2/sum(sig.^2),'ko','Linewidth',2);
xlabel('Singular Value');
ylabel('Energy');
subplot(2,2,4);
semilogy(sig.^2/sum(sig.^2),'ko','Linewidth',2);
ylabel('Energy (log scale)');
xlabel('Singular Value');
print('svals_4','-dpng');

f = 1:6;
t = 1:numFrames;
figure(3);
subplot(2,1,1);
plot(f,U(:,1),'b',f,U(:,2),'--r',f,U(:,3),':k','Linewidth',2);
title('POD Modes (Test 4)');
xlabel('x');
legend('mode 1','mode 2','mode 3','Location','northwest');
subplot(2,1,2);
plot(t,V(:,1),'b',t,V(:,2),'--r',t,V(:,3),':k','Linewidth',2);
xlabel('t');
ylim([-0.09 -0.05]);
legend('v_1','v_2','v_3','Location','northwest');
print('modes_4','-dpng');

f_rank1 = U(:,1)*S(1,1)*V(:,1)';
figure(4);
plot(1:numFrames,f_rank1(1,:));
title('Rank 1 Approximation (Test 4)');
xlabel('time (frames)');
ylabel('position');
print('projection_4','-dpng');
