function X=reconstruct_mul(data,m,tau)
%�ú��������ع���ռ�
% mΪǶ��ռ�ά������������������Ƕ��ռ�ά�����һ�£�
% tauΪʱ���ӳ�����
% dataΪ��Ԫ����ʱ������
% NΪʱ�����г���
% XΪ���,��m*nά����
%LΪ��Ԫʱ�����еı�������
data=data';
m_max=max(m);
N=size(data,2);
L=size(data,1);
X=zeros(L,m_max);     %X��ʼ��

for i_L=1:L
   for i_m=1:m(i_L)
        X(i_L,i_m)=data(i_L,N-(i_m-1)*tau(i_L));   %��N��������������֮����ռ��ع�֮�������
   end
end
    


