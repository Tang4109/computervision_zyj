A=imread('Wirebond.tif');
A=A*255;%����255
figure,imshow(A);%��ʾԭͼ��
F1=[-1 -1 -1;2 2 2;-1 -1 -1];%0�ȣ�ˮƽ�߼��ģ��
g1=sob(A,F1);%��ˮƽ��Ե

F2=[2 -1 -1;-1 2 -1;-1 -1 2];%90�ȣ��߼��ģ��
g2=sob(A,F2);%����ֱ��Ե

F3=[-1 2 -1;-1 2 -1;-1 2 -1];%45�ȣ��߼��ģ��
g3=sob(A,F3);%��45�ȱ�Ե

F4=[-1 -1 2;-1 2 -1;2 -1 -1];%135�ȣ�135���߼��ģ��
g4=sob(A,F4);%��135�ȱ�Ե

g=g1+g2+g3+g4;%��4������ı�Ե��ӵ������ı�Եͼ��
[m,n]=size(g);
%��g��Ϊ��ֵͼ��
for x=1:m
    for y=1:n
        if g(x,y)>=255
            g(x,y)=255;
        else
            g(x,y)=0;
        end
    end
end

%��ʾͼ��            
figure,imshow(g,[]);

%��Ե��⺯��
function [g]=sob(A,F)
w=1;%3x3��ģ��
A=double(A);%�������ͼ��
dim = size(A);%ͼƬ��С486x486
B = zeros(dim);%����һ����ԭͼ��һ����С��0����

for i = 1:dim(1)
   for j = 1:dim(2)
         % ��ȡ�ֲ�����   
         iMin = max(i-w,1);
         iMax = min(i+w,dim(1));
         jMin = max(j-w,1);
         jMax = min(j+w,dim(2));
         %���嵱ǰ�����õ�����Ϊ(iMin:iMax,jMin:jMax)
         I = A(iMin:iMax,jMin:jMax);%��ȡ�������Դͼ��ֵ����I
         G=F((iMin:iMax)-i+w+1,(jMin:jMax)-j+w+1);%��ȡĳһ���ص��ģ��
         B(i,j) = sum(G(:).*I(:));%���    
   end
end
g=B;
end

