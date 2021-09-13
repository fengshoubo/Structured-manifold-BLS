function W = sparse_bls_LF_L21_1012(cpu_Y,cpu_A,lam,rou,itrs)
%优化目标函数变为 ZW-X 的F 范数的平方 + W 的 L21 范数
%样本数目为 s， 样本维度为 m ， 映射维度为 n
%A: s*m  输入
%Y: s*n  输出
%W: m*n  线性变换矩阵
%lam: W 权值稀疏惩罚系数
%rou： 增广拉格朗日项 系数
A=gpuArray(cpu_A);
Y=gpuArray(cpu_Y);
s=size(A,1);
m = size(A,2);
n = size(Y,2);
Ok=gpuArray.rand(m,n);
Uk=gpuArray.zeros(m,n);      %对偶乘子
tau=lam/rou;       %used  in the vect-soft threshold
temp=inv(2*(A'*A)+rou*eye(size(A'*A)));
temp1=2*temp*A'*Y;
%%%%%% The ADMM algorithm %%%%%%%%%%%%%

for i_itrs = 1:itrs
    % ADMM Wk update
    i_itrs;
    Wk=temp1+rou*temp*(Ok-Uk);
    %ADMM Ok update
    imp=Wk+Uk;                  %为什么是 Wk-D2？ 更新之后的 Ok+1 所
    [Row_L2]=compute_W2(imp);   %在soft-vect中会用到的行二范数
    temp2=max(Row_L2-tau,0);    %soft-vect用到的使行为零的项
    temp3=temp2./(temp2+tau);   %
    Ok=temp3.*imp;              % soft-vect update finish
    % ADMM Uk update
    Uk=Uk+Wk-Ok;
    W=gather(Ok);
end
err_W=gather(Wk-Ok);
end

function [g]=compute_W2(W1)
[m,n]=size(W1);
g=gpuArray.zeros(m,1);
parfor i=1:m
    g(i)=norm(W1(i,:),2); %计算矩阵的行二范数
end
% G=diag(g);
end

