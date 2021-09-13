function [W,G_Y]=LapEigenmaps_L21(Data,n_d,alpha, beta)
Data=single(Data);
num_GraphLaplacian=length(Data);
%% Compute graph laplacian matrix
[mappedX, mapping] = laplacian_eigen(Data(1:num_GraphLaplacian,:), n_d);

X=Data(mapping.conn_comp,:)';
L=mapping.L;
[n_D,n_sam]=size(X);
W=ones(n_D,n_d);
Y=eye(n_d,n_sam);
G_Y=gpuArray(single(Y));
G_W=gpuArray(single(W));
G_X=gpuArray(single(X));
temp_XXT=beta*(G_X*G_X');   %数据首先要归一化，然后才能进行计算
CPU_XXT=gather(temp_XXT);


%% Start the iteration

for i_iter=1:50
    G_U=compute_L2(G_W);
    U=gather(G_U);
    %% Compute Y
  
    for i_Y_iter=1:10
        temp_Y=(G_Y'*G_W'*G_X);
        A=L+beta*(eye(n_sam)-2*temp_Y);       %矩阵不对阵，造成特征值存在虚根
        
        [G_V_eig,G_D_eig]=eig((A+A')/2);        %防止出现不对称矩阵的情况
        
        V_eig=gather(G_V_eig);
        D_eig=gather(G_D_eig);
        [D_sort,index]=sort(diag(D_eig),'ascend');
        V_sort = V_eig(:,index);
        index_0=find(abs(D_sort)<1e-8);
        
        if index_0<=n_d
            Y=V_sort(:,[1:index_0-1 index_0+1])';
        else
            Y=V_sort(:,1:n_d)';
        end
        G_Y=gpuArray(Y);
        Y_converge_inner(i_Y_iter)=norm(Y,'fro');
    end
    plot(Y_converge_inner);
    
    %% Compute W
    
    for i_W_iter=1:50
        temp_W_up=beta*G_X*G_Y';
        temp_W_down=(temp_XXT+alpha*G_U)*G_W;
        CPU_W_up=gather(temp_W_up);
        CPU_W_down=gather(temp_W_down);
        
%         G_W=G_W.*(temp_W_up./temp_W_down);
        %这个和下面这个有什么区别？
            temp_W=(temp_W_up./temp_W_down);
            G_W=(2/3)*G_W+(1/3)*G_W.*temp_W;
        W=gather(G_W);
        W_converge_inner(i_W_iter)=norm(W,'fro');
    end
    plot(W_converge_inner);
    
    W_converge_outter(i_iter)=norm(W,'fro');
    Y_converge_outter(i_iter)=norm(Y,'fro');
end

subplot(2,1,1);
plot(W_converge_outter);
subplot(2,1,2);
plot(Y_converge_outter);
% parfor i_W= 1:size(W,1)
%     if norm(W(i_W,:),2)<0.001
%         W(i_W,:)=zeros(size( W(i_W,:)));
%     end
% end

end





%%  Others
function [G]=compute_L2(W1)
[m,n]=size(W1);
delta=gpuArray(0.0001);
parfor i=1:m
    if norm(W1(i,:),2)<delta
        g(i)=1/delta;
    else
        g(i)=1/(norm(W1(i,:),2));
    end
end
G=diag(g);
end