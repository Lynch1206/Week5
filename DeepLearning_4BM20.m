clearvars; clc; close all;

%% Load images and labels
ImagePath= './Images/';
load('./Images/Labels.mat');
imds = imageDatastore(ImagePath,'labels',categorical(Labels));

numImage = length(imds.Files);
ImageSize = size(readimage(imds,1));      % readimage(): Read specified image from datastore
numLabels = size(countEachLabel(imds),1); % countEachLabel(): Count files in ImageDatastore labels

%% Show images
figure(1)
random = randperm(numImage,15);
for i = 1:15
    subplot(3,5,i);
    imshow(imds.Files{random(i)});
    title(Labels(random(i)))      % 0 indicates circle; 1 indicates square
end

%% Split the data set into a training set and a validation set
TrainFiles = 0.8;    % the percentage of the files to assign to imds_train
[imdsTrain,imdsTest] = splitEachLabel(imds,TrainFiles,'randomized');

%% Define the network architecture
FilterSize = 3;     % Height and width of the filters
numFilters = 9;     % the number of channels in the output of a convolutional layer.

layers = [ ...
    imageInputLayer(ImageSize)
    convolution2dLayer(FilterSize, numFilters, 'Padding', 'same') 
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    fullyConnectedLayer(numLabels)
    softmaxLayer
    classificationLayer];

%% Specify the training option
% An epoch is the full pass of the training algorithm over the entire training set.

options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.001,  ...    
    'MaxEpochs',100, ...               
    'Shuffle','every-epoch',...
    'ValidationData',imdsTest, ...
    'ValidationFrequency',25, ...      
    'Verbose',false, ...
    'Plots','training-progress');

%% Train the network
net = trainNetwork(imdsTrain,layers,options);

%% Run the trained network on the test set
YClass = classify(net,imdsTest);
YTest = imdsTest.Labels;

%% Calculate the accuracy
%  Accuracy is the fraction of labels that the network predicts correctly
accuracy = sum(YClass == YTest)/numel(YTest)

%% Show results
figure(2)
random = randperm(round(numImage*(1-TrainFiles)-1), 15);
for i = 1:15
    R = random(i);
    subplot(3,5,i);
    imshow(imdsTest.Files{R});
    
    TestResult = YClass(R); 
    Original = imdsTest.Labels(R);
    
    if  TestResult == Original
        title([TestResult,'-',Original]);
    else
        title([TestResult,'-',Original],'color','red');
    end
end

%% Show all wrong results
j = 0; k = [];
for i = 1:200
    if YClass(i) ~= imdsTest.Labels(i)
        j = j+1;
        k(j) = i;
    end
end

figure(3)
for ii = 1:length(k)
    subplot(ceil(length(k)/3),3,ii);
    imshow(imdsTest.Files{k(ii)})
    title([YClass(k(ii)),'-',imdsTest.Labels(k(ii))],'color','red');
end
 %% Softmax Layer
 YPredict = predict(net, imdsTest);
 
 %figure('units','normalized','outerposition',[0 0 1/2 1]);
 figure(4)
 for i = 1:15
     R = random(i);
     subplot(3,5,i);
     imshow(imdsTest.Files{R});
     if YClass(R) == imdsTest.Labels(R)
         title(num2str(max(YPredict(R,:)),'%0.2f'));
     else
         title(num2str(max(YPredict(R,:)),'%0.2f'),'color','red');
     end
 end
 