%clear all;

%  M = [-1 -1 -1 -1 +1 +1 -1 -1 -1 -1;
%       -1 -1 -1 +1 +1 +1 +1 -1 -1 -1;
%       -1 -1 +1 +1 -1 -1 +1 +1 -1 -1;
%       -1 +1 +1 -1 -1 -1 -1 +1 +1 -1;
%       -1 +1 +1 +1 +1 +1 +1 +1 +1 -1;
%       +1 +1 +1 +1 +1 +1 +1 +1 +1 +1;
%       +1 +1 +1 -1 -1 -1 -1 +1 +1 +1;
%       +1 +1 -1 -1 -1 -1 -1 -1 +1 +1;
%       +1 +1 -1 -1 -1 -1 -1 -1 +1 +1;
%       +1 +1 -1 -1 -1 -1 -1 -1 +1 +1];

a = [-1 +1];
[1 1 1 1 1

x = load('letterLearn2.mat');
%y = load('letterTest.mat');
x = x.x;
%y = y.notcorrect;
% 
%x(4,:) = letterH(:);

% data_train(1,:) = x(1,:); % Letter A load learn
% data_train(2,:) = x(2,:); % Letter B load learn
% data_train(3,:) = x(3,:); % Letter C load learn
% data_train(4,:) = x(4,:); % Letter H load learn
% data_train(5,:) = x(5,:); % Letter T load learn
% data_train(6,:) = x(6,:); % Letter I load learn
% data_train(7,:) = x(7,:); % Letter X load learn
% data_train(8,:) = x(8,:); % Letter N load learn
% data_train(9,:) = x(9,:); % Letter Z load learn
% data_train(10,:) = x(10,:); % Letter K load learn
% 

data_train(1,:) = x(1,:); % Letter A load learn
data_train(2,:) = x(2,:); % Letter B load learn
% data_train(3,:) = x(3,:); % Letter C load learn
data_train(3,:) = x(18,:); % patternFive
data_train(4,:) = x(8,:); % Letter N load learn
data_train(5,:) = x(9,:); % Z
% %data_train(7,:) = x(10,:); % K
% % data_train(3,:) = x(5,:); % Letter T load learn
% % %data_train(4,:) = x(7,:); % Letter X load learn
% % data_train(4,:) = x(8,:); % Letter N load learn
% % data_train(5,:) = x(9,:); % Letter Z load learn
% % data_train(6,:) = x(10,:); % Letter K load learn
data_train(6,:) = x(11,:); % patternOne
data_train(7,:) = x(12,:); % patternTwo
data_train(8,:) = x(17,:); % letterO
data_train(9,:) = x(16,:); % patternFour
data_train(1,:) = x(13,:); % patternThree

im = reshape(data_train(1,:),10,10);
imshow(im)

% temp = reshape(data_train(1,:),10,10);
% %imwrite(temp,'patternSqr.jpg')
% imshow(temp,'InitialMagnification', 2500)
% 
% 
% temp2 = round(imresize(temp,[10,10]));
% data_train(1,:) = temp2(:);
% 
% img = imresize(reshape(data_train(1,:),10,10),[10,10]);
% img2 = imresize(reshape(data_train(2,:),10,10),[50,50]);
% img3 = imresize(reshape(data_train(3,:),10,10),[50,50]);
% img4 = imresize(reshape(data_train(4,:),10,10),[50,50]);
% img5 = imresize(reshape(data_train(5,:),10,10),[50,50]);
% img6 = imresize(reshape(data_train(6,:),10,10),[50,50]);
% img7= imresize(reshape(data_train(7,:),10,10),[50,50]);
% img8 = imresize(reshape(data_train(8,:),10,10),[50,50]);
% img9 = imresize(reshape(data_train(9,:),10,10),[50,50]);

%data_train2(1,:) = round(img(:));
% data_train2(2,:) = round(img2(:));
% data_train2(3,:) = round(img3(:));
% data_train2(4,:) = round(img4(:));
% data_train2(5,:) = round(img5(:));
% data_train2(6,:) = round(img6(:));
% data_train2(7,:) = round(img7(:));
% data_train2(8,:) = round(img8(:));
% data_train2(9,:) = round(img9(:));

% figure(1)
% imshow(round(img9))
% data_test(1,:) = y(1,:); % Letter A load test
% data_test(2,:) = y(2,:); % Letter B load test
% data_test(3,:) = y(3,:); % Letter C load test
%outrow = zeros(10,62500);
m = ones(247,247);
n(1,:) = (m(:))';
for i = 1:8
    % damage the data
    img_damage = 40;
    sweeps = 800*247;
    data_damaged = damage(data_train(:,:), data_train(:,:), img_damage);

%     M = reshape(y(1,:),10,10)';

    tic
    Weight = network_train(data_train(:,:));
    toc
    tic
    out = network_test(Weight, data_damaged(9,:), data_damaged(3,:), sweeps);
    toc
    
    outrow(i,:) = out;
    %data_train3 = mat2gray(data_train(4,:),[-1 1]);
     %I = reshape(data_train(1,:),10,10);
% 
%     %I2 = mat2gray(out(1,:),[-1 1]);
%      I2 = reshape(out,10,10);
%      I3 = reshape(data_damaged(1,:),10,10);
 
end
% figure(1)
%  subplot(1,3,1),imshow(I, 'InitialMagnification', 2500)
%  subplot(1,3,2),imshow(I2, 'InitialMagnification', 2500)
%  subplot(1,3,3),imshow(I3, 'InitialMagnification', 2500)
%I = reshape(outrow(1,:),10,10);
% I2 = mat2gray(out(1,:),[-1 1]);
% I2 = reshape(outrow(i,:),10,10);
% I3 = reshape(data_damaged(1,:),10,10);

It1 = reshape(outrow(1,:),10,10);
It2 = reshape(outrow(2,:),10,10);
It3 = reshape(outrow(3,:),10,10);
It4 = reshape(outrow(4,:),10,10);
It5 = reshape(outrow(5,:),10,10);
It6 = reshape(outrow(6,:),10,10);
It7 = reshape(outrow(7,:),10,10);
It8 = reshape(outrow(8,:),10,10);
% It9 = reshape(outrow(9,:),10,10);  
% It10 = reshape(outrow(10,:),10,10);

%figure(1)
% e = reshape(data_damaged(5,:),10,10);
%subplot(1,3,1),imshow(e, 'InitialMagnification', 2500)
%subplot(1,1,1),imshow(I, 'InitialMagnification', 2500)
% subplot(1,3,3),imshow(I2, 'InitialMagnification', 2500)
% 
figure(2)
subplot(1,10,1),imshow(It1, 'InitialMagnification', 2500)
subplot(1,10,2),imshow(It2, 'InitialMagnification', 2500)
subplot(1,10,3),imshow(It3, 'InitialMagnification', 2500)
subplot(1,10,4),imshow(It4, 'InitialMagnification', 2500)
subplot(1,10,5),imshow(It5, 'InitialMagnification', 2500)
subplot(1,10,6),imshow(It6, 'InitialMagnification', 2500)
subplot(1,10,7),imshow(It7, 'InitialMagnification', 2500)
subplot(1,10,8),imshow(It8, 'InitialMagnification', 2500)
% subplot(1,10,9),imshow(It9, 'InitialMagnification', 2500)
% subplot(1,10,10),imshow(It10, 'InitialMagnification', 2500)

% figure(3)
% It11 = reshape(data_train(5,:),10,10);
% imshow(It11,  'InitialMagnification', 2500)

% calculating the weight matrix
function Weight = network_train(M)
    Weight = zeros(size(M,2),size(M,2));
    for i = 1:size(M,2)
        for j = 1:size(M,2)
            J = 0;
            if i ~= j
                for n = 1:size(M,1)
                    J = M(n,i).*M(n,j) + J;
                end
            end
            Weight(i,j) = J;%sign(1/(size(M,1))*J);
        end
    end
end

function data_test = network_test(Weight, data_test, data_train, no_sweeps)
    for n = 1:1:size(data_test,1)

        % recognition
        iteration = 0;
        iterationOfLastChange = 0;
        loop = true;
        for k = 1:no_sweeps
            %output = data_test;
            % Counter
            iteration = iteration + 1;

            % Generate random element for the asynchronous correction
            i = randi([1 size(data_train,2)],1,1);
            Si = 0;
            % calculating the interaction energy
            for j = 1:1:size(data_test,2)
                Si = Si + Weight(i,j) * data_test(n,j);
            end
            
            % setting testing matrix to closest memory
            out = 0;
            changed = 0;
            if (Si ~= 0)
                if (Si < 0) 
                    out = -1;           
                end
                if (Si >= 0)
                    out = 1;
                    %out = exp(-(sum(Si)/1000));           
                end
                if (out ~= data_test(n,i))
                    changed = 1;
                    data_test(n,i) = out;
                end
            end
            % Main condition
            if (changed == 1)
                iterationOfLastChange = iteration;
            end

            % Break condition after 1000 iterations
            if (iteration - iterationOfLastChange > 1000)
                loop = false;
            end
        end
    end
end

function data_damaged = damage(data_train, data_test, damage_limit)

    
    limit = damage_limit*2;
    data_damaged = data_train(:,:);
    f = 0;
    
    while limit > f
        r = randi([1 100],1, 1);
        
        if data_test(1,r) == data_train(1,r)
           data_damaged(1,r) = -1*data_test(1,r);
           f = sum(abs(data_damaged(1,:) - data_train(1,:)));
        end
    end
    
    f = 0;
    while limit > f
        r = randi([1 100],1, 1);
        if data_test(2,r) == data_train(2,r)
           data_damaged(2,r) = -1*data_test(2,r);
           f = sum(abs(data_damaged(2,:) - data_train(2,:)));
        end
    end

    f = 0;
    while limit > f
        r = randi([1 100],1, 1);
        if data_test(3,r) == data_train(3,r)
           data_damaged(3,r) = -1*data_test(3,r);
           f = sum(abs(data_damaged(3,:) - data_train(3,:)));
        end
    end
    f = 0;
    while limit > f
        r = randi([1 100],1, 1);
        if data_test(4,r) == data_train(4,r)
           data_damaged(4,r) = -1*data_test(4,r);
           f = sum(abs(data_damaged(4,:) - data_train(4,:)));
        end
    end
    f = 0;
    while limit > f
        r = randi([1 100],1, 1);
        if data_test(5,r) == data_train(5,r)
           data_damaged(5,r) = -1*data_test(5,r);
           f = sum(abs(data_damaged(5,:) - data_train(5,:)));
        end
    end
    f = 0;
    while limit > f
        r = randi([1 100],1, 1);
        if data_test(6,r) == data_train(6,r)
           data_damaged(6,r) = -1*data_test(6,r);
           f = sum(abs(data_damaged(6,:) - data_train(6,:)));
        end
    end
    f = 0;
    while limit > f
        r = randi([1 100],1, 1);
        if data_test(7,r) == data_train(7,r)
           data_damaged(7,r) = -1*data_test(7,r);
           f = sum(abs(data_damaged(7,:) - data_train(7,:)));
        end
    end
    f = 0;
    while limit > f
        r = randi([1 100],1, 1);
        if data_test(8,r) == data_train(8,r)
           data_damaged(8,r) = -1*data_test(8,r);
           f = sum(abs(data_damaged(8,:) - data_train(8,:)));
        end
    end
    f = 0;
    while limit > f
       r = randi([1 100],1, 1);
       if data_test(9,r) == data_train(9,r)
           data_damaged(9,r) = -1*data_test(9,r);
           f = sum(abs(data_damaged(9,:) - data_train(9,:)));
        end
    end
end

% Test network accuracy
% A_accuracy = (y(1,:) - data_train(1,:));
% B_accuracy = (y(2,:) - data_train(2,:));
% C_accuracy = (y(3,:) - data_train(3,:));
% 
% % How close is the prediction to either the letter A, B or C
% A_accuracy = 100 - length(A_accuracy(A_accuracy < 0)) 
% B_accuracy = 100 - length(B_accuracy(B_accuracy < 0))
% C_accuracy = 100 - length(C_accuracy(C_accuracy < 0))

% results = [A_accuracy; B_accuracy; C_accuracy]
