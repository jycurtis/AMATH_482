clear; close all; clc;

%% Prepare Training Data
[trainImg, trainLbl] = mnist_parse('train-images.idx3-ubyte', 'train-labels.idx1-ubyte');
trainImg = reshape(trainImg,[size(trainImg,1)*size(trainImg,2),size(trainImg,3)]);

numTrain = size(trainImg,2);
for i = 1:numTrain
    trainImg(:,i) = trainImg(:,i) - mean(trainImg(:,i));
end
trainImg = im2double(trainImg);

%% Perform SVD
[U,S,V] = svd(trainImg,'econ');
sig = diag(S);
trainProj = S*V';

%% Plot Singular Values
x = 1:length(sig);
t = 1:numTrain;
figure(1);
subplot(1,2,1);
plot(diag(S),'bo');
title('Singular Values');
ylabel('Singular Values (linear scale)');
subplot(1,2,2);
semilogy(diag(S),'bo');
title('Singular Values');
ylabel('Singular Values (log scale)');
print('svals','-dpng');

figure(2);
for i = 1:6
    subplot(2,3,i);
    imshow(rescale(abs(reshape(U(:,i),28,28))));
    title(['mode' num2str(i)]);
end
print('pca_modes','-dpng');

figure(3);
proj = trainProj([2,3,5],:);
CM = jet(10);
for i = 0:9
    plotting = proj(:,(trainLbl==i));
    scatter3(plotting(1,:),plotting(2,:),plotting(3,:),5,CM(i+1,:),'filled');
    hold on;
end
title('Scatter Plot of Training Data Projected onto Modes 2,3,5');
xlabel('Mode 2'); ylabel('Mode 3'); zlabel('Mode 5');
legend('0','1','2','3','4','5','6','7','8','9');
print('proj_scatter','-dpng');

feature = 20;

%% Prepare Test Data
[testImg, testLbl] = mnist_parse('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte');
testImg = reshape(testImg,[size(testImg,1)*size(testImg,2),size(testImg,3)]);

numTest = size(testImg,2);
for i = 1:numTest
    testImg(:,i) = testImg(:,i) - mean(testImg(:,i));
end
testImg = im2double(testImg);

%% Perform LDA With 2 Digits
digits = {2,7};
dA = trainProj(1:feature,(trainLbl==digits{1}));
dB = trainProj(1:feature,(trainLbl==digits{2}));
[thresh,w,sortV,order] = digit_trainer({dA,dB});

figure(4);
subplot(2,2,1);
histogram(sortV{1},30); hold on, plot([thresh{1} thresh{1}], [0 1000],'r');
title(['Training Digit ',num2str(digits{1})]);
xlabel('LDA Projection'); ylabel('Frequency');
subplot(2,2,2);
histogram(sortV{2},30); hold on, plot([thresh{1} thresh{1}], [0 1000],'r');
title(['Training Digit ',num2str(digits{2})]);
xlabel('LDA Projection'); ylabel('Frequency');
trainAcc = (sum(sortV{1}<thresh{1})+sum(sortV{2}>thresh{1}))/(size(dA,2)+size(dB,2))

testProj = U'*testImg;
tA = w'*testProj(1:feature,(testLbl==digits{order(1)}));
tB = w'*testProj(1:feature,(testLbl==digits{order(2)}));
subplot(2,2,3);
histogram(tA,30); hold on, plot([thresh{1} thresh{1}], [0 200],'r');
title(['Test Digit ',num2str(digits{order(1)})]);
xlabel('LDA Projection'); ylabel('Frequency');
subplot(2,2,4);
histogram(tB,30); hold on, plot([thresh{1} thresh{1}], [0 200],'r');
title(['Test Digit ',num2str(digits{order(2)})]);
xlabel('LDA Projection'); ylabel('Frequency');
testAcc = (sum(tA<thresh{1})+sum(tB>thresh{1}))/(size(tA,2)+size(tB,2))
print('lda_hist2','-dpng');

%% Perform LDA With 3 Digits
digits = {2,7,8};
dA = trainProj(1:feature,(trainLbl==digits{1}));
dB = trainProj(1:feature,(trainLbl==digits{2}));
dC = trainProj(1:feature,(trainLbl==digits{3}));
[thresh,w,sortV,order] = digit_trainer({dA,dB,dC});

figure(5);
subplot(2,3,1);
histogram(sortV{1},30);
hold on, plot([thresh{1} thresh{1}], [0 1000],'r'), plot([thresh{2} thresh{2}], [0 1000],'r');
title(['Training Digit ',num2str(digits{1})]);
xlabel('LDA Projection'); ylabel('Frequency');
subplot(2,3,2);
histogram(sortV{2},30);
hold on, plot([thresh{1} thresh{1}], [0 1000],'r'), plot([thresh{2} thresh{2}], [0 1000],'r');
title(['Training Digit ',num2str(digits{2})]);
xlabel('LDA Projection'); ylabel('Frequency');
subplot(2,3,3);
histogram(sortV{3},30);
hold on, plot([thresh{1} thresh{1}], [0 1000],'r'), plot([thresh{2} thresh{2}], [0 1000],'r');
title(['Training Digit ',num2str(digits{3})]);
xlabel('LDA Projection'); ylabel('Frequency');
trainAcc = (sum(sortV{1}<thresh{1})+sum(sortV{2}>thresh{1}&sortV{2}<thresh{2})+sum(sortV{3}>thresh{2}))/(size(dA,2)+size(dB,2)+size(dC,2))

testProj = U'*testImg;
tA = w'*testProj(1:feature,(testLbl==digits{order(1)}));
tB = w'*testProj(1:feature,(testLbl==digits{order(2)}));
tC= w'*testProj(1:feature,(testLbl==digits{order(3)}));
subplot(2,3,4);
histogram(tA,30); 
hold on, plot([thresh{1} thresh{1}], [0 200],'r'), plot([thresh{2} thresh{2}], [0 200],'r');
title(['Test Digit ',num2str(digits{order(1)})]);
xlabel('LDA Projection'); ylabel('Frequency');
subplot(2,3,5);
histogram(tB,30);
hold on, plot([thresh{1} thresh{1}], [0 200],'r'), plot([thresh{2} thresh{2}], [0 200],'r');
title(['Test Digit ',num2str(digits{order(2)})]);
xlabel('LDA Projection'); ylabel('Frequency');
subplot(2,3,6);
histogram(tC,30);
hold on, plot([thresh{1} thresh{1}], [0 200],'r'), plot([thresh{2} thresh{2}], [0 200],'r');
title(['Test Digit ',num2str(digits{order(3)})]);
xlabel('LDA Projection'); ylabel('Frequency');
print('lda_hist3','-dpng');
testAcc = (sum(tA<thresh{1})+sum(tB>thresh{1}&tB<thresh{2})+sum(tC>thresh{2}))/(size(tA,2)+size(tB,2)+size(tC,2))

%% Compare Digits
accuracies = zeros(10,10,2);
for A = 0:9
    for B = A+1:9
        digits = {A,B};
        dA = trainProj(1:feature,(trainLbl==A));
        dB = trainProj(1:feature,(trainLbl==B));
        [thresh,w,sortV,order] = digit_trainer({dA,dB});
        tA = w'*testProj(1:feature,(testLbl==digits{order(1)}));
        tB = w'*testProj(1:feature,(testLbl==digits{order(2)}));

        trainAcc = (sum(sortV{1}<thresh{1})+sum(sortV{2}>thresh{1}))/(size(dA,2)+size(dB,2));
        testAcc = (sum(tA<thresh{1})+sum(tB>thresh{1}))/(size(tA,2)+size(tB,2));
        accuracies(A+1,B+1,1) = trainAcc;
        accuracies(A+1,B+1,2) = testAcc;
    end
end
accuracies(accuracies == 0) = NaN;
[accMax,Imax] = max(accuracies(:,:,2),[],'all','linear');
[maxX,maxY] = ind2sub([10,10],Imax);
[accMin,Imin] = min(accuracies(:,:,2),[],'all','linear');
[minX,minY] = ind2sub([10,10],Imin);
maxX=maxX-1; maxY=maxY-1; minX=minX-1; minY=minY-1;

%% Plot Best and Worst Digits
digits = {0,1};
dA = trainProj(1:feature,(trainLbl==digits{1}));
dB = trainProj(1:feature,(trainLbl==digits{2}));
[thresh,w,sortV,order] = digit_trainer({dA,dB});

figure(6);
subplot(2,2,1);
histogram(sortV{1},30); hold on, plot([thresh{1} thresh{1}], [0 1000],'r');
title(['Training Digit ',num2str(digits{1})]);
xlabel('LDA Projection'); ylabel('Frequency');
subplot(2,2,2);
histogram(sortV{2},30); hold on, plot([thresh{1} thresh{1}], [0 1500],'r');
title(['Training Digit ',num2str(digits{2})]);
xlabel('LDA Projection'); ylabel('Frequency');

testProj = U'*testImg;
tA = w'*testProj(1:feature,(testLbl==digits{order(1)}));
tB = w'*testProj(1:feature,(testLbl==digits{order(2)}));
subplot(2,2,3);
histogram(tA,30); hold on, plot([thresh{1} thresh{1}], [0 200],'r');
title(['Test Digit ',num2str(digits{order(1)})]);
xlabel('LDA Projection'); ylabel('Frequency');
subplot(2,2,4);
histogram(tB,30); hold on, plot([thresh{1} thresh{1}], [0 200],'r');
title(['Test Digit ',num2str(digits{order(2)})]);
xlabel('LDA Projection'); ylabel('Frequency');
print('lda_hist_best','-dpng')

digits = {4,9};
dA = trainProj(1:feature,(trainLbl==digits{1}));
dB = trainProj(1:feature,(trainLbl==digits{2}));
[thresh,w,sortV,order] = digit_trainer({dA,dB});

figure(7);
subplot(2,2,1);
histogram(sortV{1},30); hold on, plot([thresh{1} thresh{1}], [0 1000],'r');
title(['Training Digit ',num2str(digits{1})]);
xlabel('LDA Projection'); ylabel('Frequency');
subplot(2,2,2);
histogram(sortV{2},30); hold on, plot([thresh{1} thresh{1}], [0 1500],'r');
title(['Training Digit ',num2str(digits{2})]);
xlabel('LDA Projection'); ylabel('Frequency');

testProj = U'*testImg;
tA = w'*testProj(1:feature,(testLbl==digits{order(1)}));
tB = w'*testProj(1:feature,(testLbl==digits{order(2)}));
subplot(2,2,3);
histogram(tA,30); hold on, plot([thresh{1} thresh{1}], [0 200],'r');
title(['Test Digit ',num2str(digits{order(1)})]);
xlabel('LDA Projection'); ylabel('Frequency');
subplot(2,2,4);
histogram(tB,30); hold on, plot([thresh{1} thresh{1}], [0 200],'r');
title(['Test Digit ',num2str(digits{order(2)})]);
xlabel('LDA Projection'); ylabel('Frequency');
print('lda_hist_worst','-dpng')

%% Predict Using Decision Tree
tree = fitctree(array2table(trainProj(1:feature,:)'),trainLbl');
predicted = predict(tree,testProj(1:feature,:)');
error = sum(predicted == testLbl);
treeAcc = error/length(predicted)

%% Predict Using SVM
Mdl = fitcecoc(array2table(trainProj(1:feature,:)'),trainLbl');
predicted = predict(Mdl,testProj(1:feature,:)');
error = sum(predicted == testLbl);
svmAcc = error/length(predicted)

%% Compare Best and Worst
subsetImg = trainProj(:,(testLbl==0|testLbl==1));
subsetLbl = trainLbl(testLbl==0|testLbl==1);

tree = fitctree(array2table(subsetImg(1:feature,:)'),subsetLbl');
predicted = predict(tree,testProj(1:feature,(testLbl==0|testLbl==1))');
error = sum(predicted == testLbl(testLbl==0|testLbl==1));
treeAccBest = error/length(predicted)

Mdl = fitcecoc(array2table(subsetImg(1:feature,:)'),subsetLbl');
predicted = predict(Mdl,testProj(1:feature,(testLbl==0|testLbl==1))');
error = sum(predicted == testLbl(testLbl==0|testLbl==1));
svmAccBest = error/length(predicted)

subsetImg = trainProj(:,(testLbl==4|testLbl==9));
subsetLbl = trainLbl(testLbl==4|testLbl==9);

tree = fitctree(array2table(subsetImg(1:feature,:)'),subsetLbl');
predicted = predict(tree,testProj(1:feature,(testLbl==0|testLbl==1))');
error = sum(predicted == testLbl(testLbl==0|testLbl==1));
treeAccWorst = error/length(predicted)

Mdl = fitcecoc(array2table(subsetImg(1:feature,:)'),subsetLbl');
predicted = predict(Mdl,testProj(1:feature,(testLbl==0|testLbl==1))');
error = sum(predicted == testLbl(testLbl==0|testLbl==1));
svmAccWorst = error/length(predicted)