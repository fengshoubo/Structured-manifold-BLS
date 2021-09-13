clear;
close all;
dbstop if error
warning off all;
format compact;
fix(clock)
Data=load('Lorenz5W.txt');
% Data=awgn(Data,30);
%     Data=load('ChangJiang_MonthlyFlows_1865_1979_1.txt')
%     Data=awgn(Data,10);
Length=50000;
Data_l=Data(1:Length,:);
[Data_n, Max, Min]=normalize(Data_l);
%   Data_n=Data_l;
%% Initialization
% tao, m, L, Start, Pre_Step, Length_of _Train, Length_of_Test, Hide_nodes
m=[20 20 20];
tau=[1 1 1];
Start=max(m.*tau);
Pre_Step=1;                  %预测步长
Max=Max(1:length(m));
Min=Min(1:length(m));
L=0;
for i=1:length(m)       %重构之后的向量长度
    L=L+m(i);
end
U_recon=zeros(Length,L);
clear i;
%% Devide the data into two sections: Train and Test
%Reconstruct
for k=Start:Length-Pre_Step
    X_recon=reconstruct_mul(Data_n(1:k,1:length(m)),m,tau);
    U_recon(k,:)=[X_recon(1,1:m(1)) X_recon(2,1:m(2))  X_recon(3,1:m(3)) ] ;
end
%Devide
Samples_x=[U_recon(Start:end-Pre_Step,:)];
% Samples_x=[Samples_x  0.2*rand(size(Samples_x,1),20)];
Samples_y=[Data_n(Start+Pre_Step:end,1:length(m)) ];

Samples=[Samples_y Samples_x];
Train=Samples(1:fix(0.9*length(Samples)),:);
Test=Samples(fix(0.9*length(Samples))+1:end,:);

train_x=Train(:,length(m)+1:end);
train_y=Train(:,1:length(m));
% train_y=Train(:,2);
test_x=Test(:,length(m)+1:end);
test_y=Test(:,1:length(m));

% train_x = double(train_x/5);
% train_y = double(train_y);
% test_x = double(test_x/5);
% test_y = double(test_y);
% test_y=Test(:,2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i_feature=[20]
    disp('i_feature');
    i_feature
    for i_C=[28 ]
        for i_alpha=[0.8]
            for i_beta=[0.05]
                for iteration=1:1
                    %%%%%%%%%%%%%%%%%%%%This is the model of broad learning sytem with%%%%%%
                    %%%%%%%%%%%%%%%%%%%%one shot structrue%%%%%%%%%%%%%%%%%%%%%%%%
                    C = 2^(-(i_C));
                    % C=100;
                    s = .8;%the l2 regularization parameter and the shrinkage scale of the enhancement nodes
                    N11=i_feature;%feature nodes  per window
                    N2=1;%  onumberf windows of feature nodes
                    N33=300;% number of enhancement nodes
                    N5= 0; %----N5: the layer number of additional enhancement nodes
                    m1 = 0; %m1:number of widow of feature nodes in the increment step
                    m2 = 0;%----m2:number of enhancement nodes related to the incremental feature nodes per increment step
                    epochs=1;
                    N1=N11; N3=N33;
                    for j=1:epochs
%                         [G_W,G_Y]=LapEigenmaps_L21_devide(train_x,i_feature,i_alpha, 0.1);
                        [G_W,G_Y]=LapEigenmaps_L21(train_x,i_feature,i_alpha,i_beta);
%                         load('G_W');
%                         G_Y=[];
%                         G_W=2*rand(60,20)-1;
%                         G_Y=[];
                        [TrainingAccuracy,TestingAccuracy,Training_time,Testing_time] = bls_train_feng_timeseries(train_x,train_y,test_x,test_y,Max, Min,s,C,N1,N2,N3,N3,N5,m1,m2,1,G_W,G_Y);
                        train_err(j,:)=TrainingAccuracy * 100;
                        test_err(j,:)=TestingAccuracy * 100;
                        train_time(j,:)=Training_time;
                        test_time(j,:)=Testing_time;
                    end
                    save ( ['norb_result_oneshot_' num2str(N3)], 'train_err', 'test_err', 'train_time', 'test_time');
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    train_error(iteration,:)=train_err;
                    test_error(iteration,:)=test_err;
                end
                disp('i_feature=');
                i_feature
                disp('i_C=');
                i_C
                disp('i_alpha=');
                i_alpha
                disp('i_beta=');
                i_beta
                disp('test_err=');
                mean(test_error)
            end
        end
        train_error_mean(i_C,:)=mean(train_error);
        test_error_mean(i_C,:)=mean(test_error);
    end
    test(i_feature,:,:)=test_error_mean;
end
% plot(test_error_mean);