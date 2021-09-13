function X=reconstruct_mul(data,m,tau)
%该函数用来重构相空间
% m为嵌入空间维数向量，向量中两个嵌入空间维数务必一致！
% tau为时间延迟向量
% data为多元输入时间序列
% N为时间序列长度
% X为输出,是m*n维矩阵
%L为多元时间序列的变量个数
data=data';
m_max=max(m);
N=size(data,2);
L=size(data,1);
X=zeros(L,m_max);     %X初始化

for i_L=1:L
   for i_m=1:m(i_L)
        X(i_L,i_m)=data(i_L,N-(i_m-1)*tau(i_L));   %第N个混沌数据来临之后，相空间重构之后的数据
   end
end
    


