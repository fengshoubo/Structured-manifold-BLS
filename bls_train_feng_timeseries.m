function [RMSE_train,RMSE_test,Training_time,Testing_time] = bls_train_feng_timeseries(train_x,train_y,test_x,test_y,Max, Min, s,C,N1,N2,N3,N4,N5,m1,m2,test_epochs,G_W,G_Y)
% Learning Process of the proposed broad learning system 增量式学习
%Input:
%---train_x,test_x : the training data and learning data
%---train_y,test_y : the label
%---We: the randomly generated coefficients of feature nodes
%---wh:the randomly generated coefficients of enhancement nodes
%----s: the shrinkage parameter for enhancement nodes
%----C: the regularization parameter for sparse regualarization
%----N1: the number of feature nodes  per window
%----N2: the number of windows of feature nodes
%----N3: 基本层的 enhancement nodes 的节点数
%----N4: additional enhancement nodes 的节点数
%----N5: the layer number of additional enhancement nodes
% ---m1:number of widow of feature nodes in the increment step
%----m2:number of enhancement nodes related to the incremental feature nodes per increment step
%%%%%%%%%%%%%%feature nodes%%%%%%%%%%%%%%

for i_test_epoch=1:test_epochs
    %                            training stage
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%% mapped feature %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    T=zeros(size(train_x,1), N1*N2+ N3+N4*N5);
    tic;
    H1 = [train_x];        % 输入 24300*2049 往外扩0.1
    T1_o=zeros(size(train_x,1),N2*N1);                     %y 24300*1000，表示feature层节点数 N1 每个窗口特征节点 100  N2 窗口数 10
    for i=1:N2
        we=orth(2*rand(size(train_x,2)+1,N1)-1);               %2049*100
        %         A1 = H1 * we; A1 = mapminmax(A1);              %没有进行特征缩放
        %%%% edit
        beta1=G_W;
        clear we;
        %       beta1  =  sparse_bls(A1,H1,1e-3,50)';               %beta1 2049*100
        %       beta1  =  sparse_bls_L21(A1,H1,1e-3,1e-2,100);
        %       beta1  =  sparse_bls_LF_L21(A1,H1,C,1,50);
        beta11{i}=beta1;                                    %结构体
        T1 = H1 * beta1;                                    % 24300*100
        fprintf(1,'Feature nodes in window %f: Max Val of Output %f Min Val %f\n',i,max(T1(:)),min(T1(:)));
        T1_o(:,N1*(i-1)+1:N1*i)=T1;                            %T1连接上，行表示样本数，列表示节点数  feature层  总体：243000*（N2*N1）（1000）
        clear A1; clear T1;
    end
    wh_random=orth(2*rand(size(H1,2),N3)-1);
    T_random=tansig(H1*wh_random);
    clear H1;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%% mapped feature end %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % output:
    % y: mapped feature 的表征
    % beta11：mapped feature 的随机映射参数，测试时使用
    % ps: 输出表征T1_o 的缩放参数
    
    %%%%%%%%%%%%% enhancement nodes layer 1 H2 T2  start %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    H2 = [T1_o 1 * ones(size(T1_o,1),1)];                    %24300*1001
    if N1*N2>=N3
        wh2=orth(2*rand(N2*N1+1,N3)-1);                 %1001*1000
    else
        wh2=orth(2*rand(N2*N1+1,N3)'-1)';
    end
    T2 = H2 *wh2;
    l2 = max(max(T2));
    l2 = s/l2;
    fprintf(1,'Enhancement nodes: Max Val of Output %f Min Val %f\n',l2,min(T2(:)));
    T2_o = tansig(T2 * l2);
    clear T2；clear H2;
    %%%%%%%%%%%%% enhancement nodes layer 1 H2 T2  end %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %output:
    %T2_o: enhancement feature layer 1 的表征
    %wh2: mapped feature 到 enhancement nodes 的随机映射参数
    %l2: enhancement nodes layer 1 的缩放系数
    
    %%%%%%%%%%%%%%%%Additional enhancement nodes layer N5 start %%%%%%%%%%%%%%%%%%%%%%%%%%%
    T_addition_enhance=zeros(size(train_x,1),N4*N5);
    H = [T2_o 1 * ones(size(T2_o,1),1)];                  %24300*1001
    for i_e_l=1:N5
        %         if N3>=N4
        %              wh=orth(2*rand(N3+1,N4)-1);                 %1001*1000  后面的additional enhancement nodes 的节点数相同
        %         else
        %              wh=orth(2*rand(N3+1,N4)'-1)';
        %         end
        wh=orth(2*rand(N4+1,N4)-1);
        
        T_i = H * wh;
        l = max(max(T_i));
        l = s/l;
        T_o = sigmoid(T_i * l);
        T_addition_enhance(:,((i_e_l-1)*N4)+1:(i_e_l*N4)) = T_o;
        fprintf(1,'%fst Enhancement nodes: Max Val of Output %f Min Val %f\n',i_e_l, l,min(T_o(:)));
        H = [T_o .1 * ones(size(T_o,1),1)];
        L(i_e_l)=l;
        Wh{i_e_l}=wh;
        clear l;clear T_o; clear wh;
    end
    clear H;
    
    %%%%%%%%%%%%%%%%%%%%Compute the output weights%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %     T=[train_x T1_o T2_o T_addition_enhance];
    %     T=[ T1_o T2_o T_addition_enhance];
%     T=[train_x T_random T2_o T_addition_enhance];
     T=[train_x  T2_o T_addition_enhance];
    %     T=[T2_o];       %只使用非线性单元作为输出
    clear T1_o; clear T2_o; clear T_addition_enhance;
    %     beta  =  0;
    %     beta2  =  sparse_bls_LF_L21_0922_night(train_y,T,C,1,50);
    %     beta2 = sparse_bls(T,train_y,C, 50);
    beta = (T'  *  T+eye(size(T',1)) * (C)) \ ( T');
    beta2=beta*train_y;
    %     beta2 = sparse_bls_LF_L21_0922_night(train_y,T,C,1,50);     %用Lf L21范数求
    
    
    Training_time = toc;
    disp('Training has been finished!');
    disp(['The Total Training Time is : ', num2str(Training_time), ' seconds' ]);
    %%%%%%%%%%%%%%%%%Training Accuracy%%%%%%%%%%%%%%%%%%%%%%%%%%
    xx = T * beta2;
    [xx]=unnormalize(xx,Max,Min);
    train_y=unnormalize(train_y,Max,Min);
    err_train=xx-train_y;
    [NRMSE_train,RMSE_train]=nrmse(err_train,Max,Min);
    disp(['Training Accuracy is : ','NRMSE',num2str(NRMSE_train) ,'RMSE',num2str(RMSE_train) ]);
    %clear T;
    tic;
    %%%%%%%%%%%%%%%%%%%%%%Testing Process%%%%%%%%%%%%%%%%%%%
    
    HH1 = [test_x ];
    %clear test_x;
    yy1=zeros(size(test_x,1),N2*N1);
    %------------ mapped feature for testing---------
    for i=1:N2
        beta1=beta11{i};
        TT1 = HH1 * beta1;
        clear beta1;
        yy1(:,N1*(i-1)+1:N1*i)=TT1;
    end
    TT_random=tansig(HH1*wh_random);
    clear TT1;clear HH1;
    %------------enhancement nodes layer 1 for testing-------------------
    HH2 = [yy1  1 * ones(size(yy1,1),1)];
    TT2 = HH2 * wh2 ;
    TT2_o=tansig(TT2 * l2);
    clear TT2; clear HH2; clear wh2;
    
    %-------------------------------------
    %     TT=[ test_x yy1 TT2_o ]; %隐含层串接矩阵
%     TT=[ test_x TT_random TT2_o ]; %隐含层串接矩阵
    TT=[ test_x  TT2_o ]; %隐含层串接矩阵
    %     TT=[  yy1 TT2_o ]; %隐含层串接矩阵
    %     TT=[TT2_o];   %只使用非线性单元作为输出
    clear T_addition_enhance; clear HH; clear l; clear wh; clear yy1;clear TT2_o;
    %%%%%%%%%%%%%%%%% testing accuracy%%%%%%%%%%%%%%%%%%%%%%%%%%%
    x = TT * beta2;
    [x]=unnormalize(x,Max,Min);
    test_y=unnormalize(test_y,Max,Min);
    err_test=x-test_y;
    [NRMSE_test,RMSE_test]=nrmse(err_test,Max,Min);
    
    Testing_time = toc;
    disp('Testing has been finished!');
    disp(['The Total Testing Time is : ', num2str(Testing_time), ' seconds' ]);
    disp(['Testing Accuracy is : ', 'NRMSE',num2str(NRMSE_test) ,'RMSE',num2str(RMSE_test) ]);
    %% plot the result
    for i_plot=1:size(x,2)
        figure;
        subplot(2,1,1);
        h(i_plot,1)=plot(x(:,i_plot),'.b');

        hold on;
        h(i_plot,2)=plot(test_y(:,i_plot),'k');
        legend(h(i_plot,:),'real value','predicted');
        title(i_plot);
        subplot(2,1,2);
        plot(err_test(:,i_plot));
        title('test error');
        
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end