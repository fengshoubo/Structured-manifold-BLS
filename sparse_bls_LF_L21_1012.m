function W = sparse_bls_LF_L21_1012(cpu_Y,cpu_A,lam,rou,itrs)
%�Ż�Ŀ�꺯����Ϊ ZW-X ��F ������ƽ�� + W �� L21 ����
%������ĿΪ s�� ����ά��Ϊ m �� ӳ��ά��Ϊ n
%A: s*m  ����
%Y: s*n  ���
%W: m*n  ���Ա任����
%lam: W Ȩֵϡ��ͷ�ϵ��
%rou�� �������������� ϵ��
A=gpuArray(cpu_A);
Y=gpuArray(cpu_Y);
s=size(A,1);
m = size(A,2);
n = size(Y,2);
Ok=gpuArray.rand(m,n);
Uk=gpuArray.zeros(m,n);      %��ż����
tau=lam/rou;       %used  in the vect-soft threshold
temp=inv(2*(A'*A)+rou*eye(size(A'*A)));
temp1=2*temp*A'*Y;
%%%%%% The ADMM algorithm %%%%%%%%%%%%%

for i_itrs = 1:itrs
    % ADMM Wk update
    i_itrs;
    Wk=temp1+rou*temp*(Ok-Uk);
    %ADMM Ok update
    imp=Wk+Uk;                  %Ϊʲô�� Wk-D2�� ����֮��� Ok+1 ��
    [Row_L2]=compute_W2(imp);   %��soft-vect�л��õ����ж�����
    temp2=max(Row_L2-tau,0);    %soft-vect�õ���ʹ��Ϊ�����
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
    g(i)=norm(W1(i,:),2); %���������ж�����
end
% G=diag(g);
end

