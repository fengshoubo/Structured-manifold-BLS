function [TrainingAccuracy,TestingAccuracy,Training_time,Testing_time] = bls_train_feng(train_x,train_y,test_x,test_y,s,C,N1,N2,N3,N4,N5,m1,m2,test_epochs,G_W,G_Y)
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
    %                            将拉普拉斯特征映射 的稀疏矩阵 作为mapped feature 的映射权值
    %%%%%%%%%%%%%%%%%%%%%%%%%%%% mapped feature %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     train_x = zscore(train_x')';                        %train_x 24300*2048
    T=zeros(size(train_x,1), N1*N2+ N3+N4*N5);      
    tic;
%     H1 = [train_x 1 * ones(size(train_x,1),1)];        % 输入 24300*2049 往外扩0.1
      H1=[train_x];
%     H1=mapminmax(H1);
    T1_o=zeros(size(train_x,1),N2*N1);                 %y 24300*1000，表示feature层节点数 N1 每个窗口特征节点 100  N2 窗口数 10
    for i=1:N2
%         we=2*rand(size(train_x,2)+1,N1)-1;               %2049*100
%         we=G_W;
%         A1 = H1 * we;
%         A1 = mapminmax(A1);
        clear we;
%        beta1  =  sparse_bls(A1,H1,1e-3,50)';               %beta1 2049*100
       beta1=G_W;
%       beta1  =  sparse_bls_L21(A1,H1,1e-2,1,20);
%         beta1  =  sparse_bls_LF_L21_0922_night(A1,H1,C,1,50);
        beta11{i}=beta1;                                    %结构体
        T1 = H1 * beta1;                                    % 24300*100
        fprintf(1,'Feature nodes in window %f: Max Val of Output %f Min Val %f\n',i,max(T1(:)),min(T1(:)));
        [T1,ps1]  =  mapminmax(T1',0,1);
        T1 = T1';
        ps(i)=ps1;
        T1_o(:,N1*(i-1)+1:N1*i)=T1;                            %T1连接上，行表示样本数，列表示节点数  feature层  总体：243000*（N2*N1）（1000）
        clear A1; clear T1;
    end
    clear H1;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%% mapped feature end %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % output:
    % y: mapped feature 的表征
    % beta11：mapped feature 的随机映射参数，测试时使用
    % ps: 输出表征T1_o 的缩放参数
    
    %%%%%%%%%%%%% enhancement nodes layer 1 H2 T2  start %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    H2 = [ T1_o 1 * ones(size(T1_o,1),1)];                    %24300*1001
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
        T_o = tansig(T_i * l);
        T_addition_enhance(:,((i_e_l-1)*N4)+1:(i_e_l*N4)) = T_o;
        fprintf(1,'%fst Enhancement nodes: Max Val of Output %f Min Val %f\n',i_e_l, l,min(T_o(:)));
        H = [T_o .1 * ones(size(T_o,1),1)]; 
        L(i_e_l)=l;
        Wh{i_e_l}=wh;
        clear l;clear T_o; clear wh;
    end
    clear H;
    %%%%%%%%%%%%%%%%%%%%Compute the output weights%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    T=[train_x T1_o T2_o T_addition_enhance];
    clear T1_o; clear T2_o; clear T_addition_enhance;
    beta  =  0;
%     beta2  =  sparse_bls_LF_L21_0922_night(train_y,T,C,1,50);
%     beta2 = sparse_bls(T,train_y, 0.03, 50);
    beta = (T'  *  T+eye(size(T',1)) * (C)) \ ( T');
    beta2=beta*train_y;
    Training_time = toc;
    disp('Training has been finished!');
    disp(['The Total Training Time is : ', num2str(Training_time), ' seconds' ]);
    %%%%%%%%%%%%%%%%%Training Accuracy%%%%%%%%%%%%%%%%%%%%%%%%%%
    xx = T * beta2;
    yy = result(xx);
    train_yy = result(train_y);
    TrainingAccuracy = length(find(yy == train_yy))/size(train_yy,1);
    disp(['Training Accuracy is : ', num2str(TrainingAccuracy * 100), ' %' ]);
    %clear T;
    tic;
    %%%%%%%%%%%%%%%%%%%%%%Testing Process%%%%%%%%%%%%%%%%%%%
%     test_x = zscore(test_x')';
%     HH1 = [test_x 1 * ones(size(test_x,1),1)];
      HH1=[test_x];
%     HH1 = [test_x*W_compute  .00001*ones(size(test_x,1),1) ];
%     HH1=mapminmax(HH1);
    %clear test_x;
    yy1=zeros(size(test_x,1),N2*N1);
    %------------ mapped feature for testing---------
    for i=1:N2
        beta1=beta11{i};ps1=ps(i);
        TT1 = HH1 * beta1;
        TT1  =  mapminmax('apply',TT1',ps1)';
        clear beta1; clear ps1;
        yy1(:,N1*(i-1)+1:N1*i)=TT1;
    end
    clear TT1;clear HH1;
    %------------enhancement nodes layer 1 for testing-------------------
    HH2 = [yy1 1 * ones(size(yy1,1),1)]; 
    TT2 = HH2 * wh2 ;
    TT2_o=tansig(TT2 * l2);
    clear TT2; clear HH2; clear wh2;
    %-------------Additional enhancement nodes  for testing----------------
    T_addition_enhance=zeros(size(test_x,1),N4*N5);
    HH = [TT2_o 1 * ones(size(TT2_o,1),1)];
    for i_e_l=1:N5
        l=L(i_e_l);
        wh=Wh{i_e_l};
        TT = HH * wh;
        TT_o = tansig(TT * l);
        HH=[TT_o 1 * ones(size(TT_o,1),1)];
        T_addition_enhance(:,(i_e_l-1)*N4+1:(i_e_l)*N4) = TT_o;
        clear TT_o;
    end
    %-------------------------------------
    TT=[test_x yy1 TT2_o T_addition_enhance]; %隐含层串接矩阵
    clear T_addition_enhance; clear HH; clear l; clear wh; clear yy1;clear TT2_o;
    %%%%%%%%%%%%%%%%% testing accuracy%%%%%%%%%%%%%%%%%%%%%%%%%%%
    x = TT * beta2;
    test_xx = result(x);
    test_yy = result(test_y);
    TestingAccuracy = length(find(test_xx == test_yy))/size(test_yy,1);
    Testing_time = toc;
    disp('Testing has been finished!');
    disp(['The Total Testing Time is : ', num2str(Testing_time), ' seconds' ]);
    disp(['Testing Accuracy is : ', num2str(TestingAccuracy * 100), ' %' ]);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
%     %%%%%%%%%%%%%%%% Increment of mapped feature nodes %%%%%%%%%%%%%%%%%%%%%%%%%%
%     train_x = zscore(train_x')';                        %train_x 24300*2048
%     H1_inc = [train_x .1 * ones(size(train_x,1),1)];
%     T1_inc=[];
%     for i=N2+1:N2+m1
%         we=2*rand(size(train_x,2)+1,N1)-1;
%         A1 = H1_inc * we; A1 = mapminmax(A1);
%         clear we;
%         beta1  =  sparse_bls(A1,H1_inc,1e-3,50)';               
%         beta11{i}=beta1;                                    
%         T1 = H1_inc * beta1;                                   
%         fprintf(1,'Feature nodes in window %f: Max Val of Output %f Min Val %f\n',i,max(T1(:)),min(T1(:)));
%         [T1,ps1]  =  mapminmax(T1',0,1);
%         T1 = T1';
%         ps(i)=ps1;
%         T1_inc=[T1_inc T1];                            
%         clear A1; clear T1; 
%     end
%     clear H1;
%     %%%%%%%%%%%%%%%%%%%%%% Corresponding enhancement nodes %%%%%%%%%%%%%%%
%     %%%%%%%%%%%%%%%%%%%%%% 第一层增强节点 %%%%%%%%%%%%%%%%%
%     H2 = [T1_inc .1 * ones(size(T1_inc,1),1)];               %24300*1001
%     if N1*m1>=m2
%          wh2=orth(2*rand(N1*m1+1,m2)-1);                 %1001*1000
%     else
%          wh2=orth(2*rand(N1*m1+1,m2)'-1)'; 
%     end
%     T2 = H2 *wh2;
%     l2 = max(max(T2));
%     l2 = s/l2;
%     fprintf(1,'Enhancement nodes: Max Val of Output %f Min Val %f\n',l2,min(T2(:)));
%     T2_inc = tansig(T2 * l2);
%     clear T2；clear H2;
%     
%     %%%%%%%%%% corrrsponding additional enhancement nodes %%%%%%%%%%%
%     %%%%%%%%%% 其它层增强节点 %%%%%%%%%%%%%%%%%%%
%     T_add_enhance_inc=zeros(size(train_x,1),m2*N5);
%     H = [T2_inc .1 * ones(size(T2_inc,1),1)];        
%     for i_e_l=1:N5
%         wh=orth(2*rand(m2+1,m2)-1);
%         T_i = H * wh;
%         l = max(max(T_i));
%         l = s/l;
%         T_o = tansig(T_i * l);
%         T_add_enhance_inc(:,((i_e_l-1)*m2)+1:(i_e_l*m2)) = T_o;
%         fprintf(1,'%fst Enhancement nodes: Max Val of Output %f Min Val %f\n',i_e_l, l,min(T_o(:)));
%         H = [T_o .1 * ones(size(T_o,1),1)]; 
%         L_inc(i_e_l)=l;
%         Wh_inc{i_e_l}=wh;
%         clear l;clear T_o; clear wh;
%     end
%     clear H;
%     T_inc=[T1_inc T2_inc T_add_enhance_inc];     %增量输出矩阵
%     %%%%%%%%% Compute the output weights of additional mapped
%     %%%%%%%%% feature  %%%%%%%%%%
%     d=beta*T_inc;                         %以下对应18-20
%     c=T_inc-T*d;
%     if all(c(:)==0)
%         [q,w]=size(d);
%         b=(eye(w)+d'*d)\(d'*beta);
%     else
%         b = (c'  *  c+eye(size(c',1)) * (C)) \ ( c' );
%     end
%     beta=[beta-d*b;b];  %以上对应18-20
%     beta2=beta*train_y;
%     
%     T=[T T_inc];                          %扩展输出矩阵
%     %%%%%%%%%%%% Training accutacy fo additional mapped feature %%%%%%%%%%%%
%     xx = T * beta2;    
%     yy = result(xx);
%     train_yy = result(train_y);
%     TrainingAccuracy = length(find(yy == train_yy))/size(train_yy,1);
%     train_err(1,1)=TrainingAccuracy;
%     disp(['Training Accuracy is : ', num2str(TrainingAccuracy * 100), ' %' ]);
%     %%%%%%%%%%%%%%%%%%Additional mapped feature Testing Process %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   
%     test_x = zscore(test_x')';
%     HH1_inc = [test_x .1 * ones(size(test_x,1),1)];
%     TT1_o_inc=[];
%     %------------ mapped feature for testing---------
%     for i=N2+1:N2+m1
%         beta1=beta11{i};ps1=ps(i);
%         TT1 = HH1_inc * beta1;
%         TT1  =  mapminmax('apply',TT1',ps1)';
%         clear beta1; clear ps1;
%         TT1_o_inc = [TT1_o_inc TT1];
%     end
%     clear TT1;clear HH1_inc;
%     %------------enhancement nodes layer 1 for testing-------------------
%     HH2_inc = [TT1_o_inc .1 * ones(size(TT1_o_inc,1),1)]; 
%     TT2_inc = HH2_inc * wh2 ;
%     TT2_o_inc=tansig(TT2_inc * l2);
%     clear TT2_inc; clear HH2; clear wh2;
%     %-------------Additional enhancement nodes  for testing----------------
%     TT_addition_enhance_inc=zeros(size(test_x,1),N4*N5);
%     HH_inc = [TT2_o_inc .1 * ones(size(TT2_o_inc,1),1)];
%     for i_e_l=1:N5
%         l=L_inc(i_e_l);
%         wh=Wh_inc{i_e_l};
%         TT_inc = HH_inc * wh;
%         TT_o_inc = tansig(TT_inc * l);
%         HH_inc=[TT_o_inc .1 * ones(size(TT_o_inc,1),1)];
%         TT_addition_enhance_inc(:,(i_e_l-1)*N4+1:(i_e_l)*N4) = TT_o_inc;
%         clear TT_o_inc;
%     end
%     %-------------------------------------
%     TT=[TT TT1_o_inc TT2_o_inc TT_addition_enhance_inc]; %隐含层串接矩阵
%     clear T_addition_enhance; clear HH; clear l; clear wh; clear yy1;clear TT2_o;
%     %%%%%%%%%%%%%%%%%% Testing Accuracy  %%%%%%%%%%%%%%%%%%%%%%%%%%%
%     x = TT * beta2;
%     test_xx = result(x);
%     test_yy = result(test_y);
%     TestingAccuracy = length(find(test_xx == test_yy))/size(test_yy,1);
%     Testing_time = toc;
%     disp('Testing has been finished!');
%     disp(['The Total Testing Time is : ', num2str(Testing_time), ' seconds' ]);
%     disp(['Testing Accuracy is : ', num2str(TestingAccuracy * 100), ' %' ]);
%     clear TT;
end