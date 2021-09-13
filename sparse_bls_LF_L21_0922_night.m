function W = sparse_bls_LF_L21_0922_night(Y,A,lam,rou,itrs)
%优化目标函数变为 ZW-X 的F 范数的平方 + W 的 L21 范数
%样本数目为 s， 样本维度为 m ， 映射维度为 n 
%lam: W 权值稀疏惩罚系数
%rou： 增广拉格朗日项 系数
s=size(A,1);
m = size(A,2);
n = size(Y,2);
Wk =zeros(m,n);
V1=zeros(s,n);
V2=zeros(m,n);
D1=zeros(s,n);
D2=zeros(m,n);
tau=lam/rou;
temp1=inv(A'*A+eye(size(A'*A)));
% ATA=A'*A;
temp=inv(A'*A+eye(size(A'*A)));
temp1=temp*A';
%%%%%% The ADMM algorithm %%%%%%%%%%%%%

for i_itrs = 1:itrs
    epsilon1=V1+D1;
    epsilon2=V2+D2;
    Wk=temp1*epsilon1+temp*epsilon2;          
    V1=(1/(1+rou))*(Y+rou*(A*Wk-D1));
    imp=Wk-D2;
    [g]=compute_W2(imp, m); 
    temp2=max(g-tau,0);
    temp3=temp2./(temp2+tau);
    
%     n_V2=size(V2,1);
    V2=temp3.*imp;
%     for i_V2=1:n_V2
%         V2(i_V2,:)=temp3(i_V2)*imp(i_V2,:);
%     end
    
    D1=D1-A*Wk+V1;
    D2=D2-Wk+V2;
end
    W=V2;
end

function [g]=compute_W2(X,gm)
    g=zeros(gm,1);
    for i= 1:gm
        g(i)=norm(X(i,:),2);
    end
end

