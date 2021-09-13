function [W,Y]=LapEigenmaps_L21(Data,n_d,alpha, beta)
Data=Data;
num_GraphLaplacian=5000;
%% Compute graph laplacian matrix
[mappedX, mapping] = laplacian_eigen(Data(1:num_GraphLaplacian,:), n_d);

% %%
X=Data(mapping.conn_comp,:)';
L=mapping.L;
[n_D,n_sam]=size(X);
W=rand(n_D,n_d);
Y=eye(n_d,n_sam);
temp_XXT=beta*(X*X');   %数据首先要归一化，然后才能进行计算
%  temp_XXT=X*X';
for i_iter=1:50
    U=compute_L2(W);
    %compute Y
    %   A=L+beta*(eye(n_sam)-2*((G_Y'*G_W'*G_X)+(G_Y'*G_W'*G_X)')/2);       %矩阵不对阵，造成特征值存在虚根
    K=(X*X'+alpha*U);
%     A=L+beta*eye(n_sam)-beta*X'/(K)*X;
     A=L-beta*X'/(K)*X;
    [V_eig,D_eig]=eigs((A+A')/2,n_d+1,'sm');        %防止出现不对称矩阵的情况
    [D_sort,index]=sort(diag(D_eig),'ascend');
    V_sort = V_eig(:,index);
    Y=V_sort(:,1:n_d)';
    %compute W
    W=(temp_XXT+alpha*U)\(X*Y'); 
end

for j_iter=1:50
    U=compute_L2(W);

    
    [U_0,Sigma_0,V_0]=svd(2*beta*X*W)
    [U_a,Sigma_a,V_a]=svd()

    Y=[U_0*V_0; U_a*V_a];
    
    
end
for i_W= 1:size(W,1)
    if norm(W(i_W,:),2)<0.01
        W(i_W,:)=zeros(size( W(i_W,:)));
    end
end

end

function [G]=compute_L2(W1)
[m,n]=size(W1);
delta=0.0001;
for i=1:m
    if norm(W1(i,:),2)<delta
        g(i)=1/delta;
    else
    g(i)=1/(norm(W1(i,:),2));
    end
end
G=diag(g);
end