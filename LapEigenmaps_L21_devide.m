function [W,G_Y]=LapEigenmaps_L21_devide(Data,n_d,alpha, beta)
%% Data the 
num_GraphLaplacian=5000;


%% Compute graph laplacian matrix
[mappedX, mapping] = laplacian_eigen(Data(1:num_GraphLaplacian,:), n_d);
% Y=mappedX';
% G_Y=mappedX;
% %% 
X=Data(mapping.conn_comp,:)';
L=mapping.L;
rou=1;
[n_D,n_sam]=size(X);
W=rand(n_D,n_d);
Y=eye(n_d,n_sam);
G_Y=gpuArray(single(Y));
G_W=gpuArray(single(W));
G_X=gpuArray(single(X));
temp_XXT=(G_X*G_X');   %��������Ҫ��һ����Ȼ����ܽ��м���
CPU_XXT=gather(temp_XXT);

%% Compute Y by Laplacian Eigenmaps
A=L;
[G_V_eig,G_D_eig]=eigs((A+A')/2, n_d+1, 'sm');        %��ֹ���ֲ��Գƾ�������
V_eig=gather(G_V_eig);
D_eig=gather(G_D_eig);
[D_sort,index]=sort(diag(D_eig),'ascend');
V_sort = V_eig(:,index);
index_0=find(abs(D_sort)<1e-8);

if index_0<=n_d
    G_Y=V_sort(:,[1:(index_0-1) (index_0+1):n_d+1])';
else
    G_Y=V_sort(:,1:n_d)';
end
X=gather(G_X);
Y=gather(G_Y);
%% Compute W by ADMM LF+L21

%     W = sparse_bls_LF_L21_1012(Y',X',alpha,1,50);    %û��ʹ��ADMM�㷨����Ȩֵû������W=O������Wÿһ�ε������ڱ任
%     W = sparse_bls_LF_L21_0922_night(Y',X',alpha,1,50);
    W = sparse_bls_L21(Y',X',2,1,50);
    %     W = sparse_bls(X',Y',0.1,100);                               % alpha
%     ��ֵӦ����0.6����
%     W=gather(G_W)  X ��Y �����ݳ߶Ȳ�һ��
end


function [G]=compute_L2(W1)
[m,n]=size(W1);
parfor i=1:m
    g(i)=1/(norm(W1(i,:),1)+0.001);
end
G=diag(g);
end
