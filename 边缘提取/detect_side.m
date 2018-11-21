A=imread('Wirebond.tif');
A=A*255;%乘以255
figure,imshow(A);%显示原图像
F1=[-1 -1 -1;2 2 2;-1 -1 -1];%0度，水平线检测模板
g1=sob(A,F1);%得水平边缘

F2=[2 -1 -1;-1 2 -1;-1 -1 2];%90度，线检测模板
g2=sob(A,F2);%得竖直边缘

F3=[-1 2 -1;-1 2 -1;-1 2 -1];%45度，线检测模板
g3=sob(A,F3);%得45度边缘

F4=[-1 -1 2;-1 2 -1;2 -1 -1];%135度，135度线检测模板
g4=sob(A,F4);%得135度边缘

g=g1+g2+g3+g4;%将4个方向的边缘相加得完整的边缘图像
[m,n]=size(g);
%将g化为二值图像
for x=1:m
    for y=1:n
        if g(x,y)>=255
            g(x,y)=255;
        else
            g(x,y)=0;
        end
    end
end

%显示图像            
figure,imshow(g,[]);

%边缘检测函数
function [g]=sob(A,F)
w=1;%3x3的模板
A=double(A);%输出浮点图像
dim = size(A);%图片大小486x486
B = zeros(dim);%创建一个和原图像一样大小的0矩阵

for i = 1:dim(1)
   for j = 1:dim(2)
         % 提取局部区域   
         iMin = max(i-w,1);
         iMax = min(i+w,dim(1));
         jMin = max(j-w,1);
         jMax = min(j+w,dim(2));
         %定义当前所作用的区域为(iMin:iMax,jMin:jMax)
         I = A(iMin:iMax,jMin:jMax);%提取该区域的源图像值赋给I
         G=F((iMin:iMax)-i+w+1,(jMin:jMax)-j+w+1);%提取某一像素点的模板
         B(i,j) = sum(G(:).*I(:));%卷积    
   end
end
g=B;
end

