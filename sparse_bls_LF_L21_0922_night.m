function W = sparse_bls_LF_L21_0922_night(Y,A,lam,rou,itrs)
%�Ż�Ŀ�꺯����Ϊ ZW-X ��F ������ƽ�� + W �� L21 ����
%������ĿΪ s�� ����ά��Ϊ m �� ӳ��ά��Ϊ n 
%lam: W Ȩֵϡ��ͷ�ϵ��
%rou�� �������������� ϵ��
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

