function [sci,sct]=SPWCF(R1,iter,w)

[m,n,b]=size(R1);
bj=(w+1)/2;
aj=(w-1)/2;
tp=zeros(1,8,b);
tep1=zeros(b,8);
tep2=zeros(b,8);
val=zeros(8,1);
tnum=((w+1)/2)*w-1;
re=zeros(m,n);
for k=1:iter
    Img=zeros(m+2*aj,n+2*aj,b);
    Img(aj+1:m+aj,aj+1:n+aj,:)=R1;
    Img(:,1:aj,:)=Img(:,2*aj:-1:aj+1,:);
    Img(:,n+aj+1:n+2*aj,:)=Img(:,n+aj:-1:n+1,:);
    Img(1:aj,:,:)=Img(2*aj:-1:aj+1,:,:);
    Img(m+aj+1:m+2*aj,:,:)=Img(m+aj:-1:m+1,:,:);
    itimg=Img;
    for i=1+aj:m+aj
        for j=1+aj:n+aj
            T=itimg(i,j,:);
            T=T(:);
            pat = itimg(i-aj: i+aj, j-aj: j+aj, :);
            block=pat;
            for x=1:w         % 1 subwindow
                for y=1:bj
                    if x==bj && y==bj
                        tp(1,1,:)=tp(1,1,:)+0;
                    else
                        tp(1,1,:)=tp(1,1,:)+block(x,y,:);
                    end
                end
            end
            tep1(:,1)=reshape(tp(1,1,:),1,b)';
            tep2(:,1)=tep1(:,1)./tnum-T;
            for x=1:w         % 2 subwindow
                for y=bj:w
                    if x==bj && y==bj
                        tp(1,2,:)=tp(1,2,:)+0;
                    else
                        tp(1,2,:)=tp(1,2,:)+block(x,y,:);
                    end
                end
            end
            tep1(:,2)=reshape(tp(1,2,:),1,b)';
            tep2(:,2)=tep1(:,2)./tnum-T;
            for x=1:bj         % 3 subwindow
                for y=1:w
                    if x==bj && y==bj
                        tp(1,3,:)=tp(1,3,:)+0;
                    else
                        tp(1,3,:)=tp(1,3,:)+block(x,y,:);
                    end
                end
            end
            tep1(:,3)=reshape(tp(1,3,:),1,b)';
            tep2(:,3)=tep1(:,3)./tnum-T;
            for x=bj:w         % 4 subwindow
                for y=1:w
                    if x==bj && y==bj
                        tp(1,4,:)=tp(1,4,:)+0;
                    else
                        tp(1,4,:)=tp(1,4,:)+block(x,y,:);
                    end
                end
            end
            tep1(:,4)=reshape(tp(1,4,:),1,b)';
            tep2(:,4)=tep1(:,4)./tnum-T;
            num=0;
            for x=1:w          % 5 subwindow
                for y=1:(w-num)
                    if x==bj && y==bj
                        tp(1,5,:)=tp(1,5,:)+0;
                    else
                        tp(1,5,:)=tp(1,5,:)+block(x,y,:);
                    end
                end
                num=num+1;
            end
            tep1(:,5)=reshape(tp(1,5,:),1,b)';
            tep2(:,5)=tep1(:,5)./tnum-T;
            num=0;
            for x=1:w          % 6 subwindow
                for y=(1+num):w
                    if x==bj && y==bj
                        tp(1,6,:)=tp(1,6,:)+0;
                    else
                        tp(1,6,:)=tp(1,6,:)+block(x,y,:);
                    end
                end
                num=num+1;
            end
            tep1(:,6)=reshape(tp(1,6,:),1,b)';
            tep2(:,6)=tep1(:,6)./tnum-T;
            num=0;
            for x=1:w          % 7 subwindow
                for y=(w-num):w
                    if x==bj && y==bj
                        tp(1,7,:)=tp(1,7,:)+0;
                    else
                        tp(1,7,:)=tp(1,7,:)+block(x,y,:);
                    end
                end
                num=num+1;
            end
            tep1(:,7)=reshape(tp(1,7,:),1,b)';
            tep2(:,7)=tep1(:,7)./tnum-T;
            num=0;
            for x=1:w          % 8 subwindow
                for y=1:(1+num)
                    if x==bj && y==bj
                        tp(1,8,:)=tp(1,8,:)+0;
                    else
                        tp(1,8,:)=tp(1,8,:)+block(x,y,:);
                    end
                end
                num=num+1;
            end
            tep1(:,8)=reshape(tp(1,8,:),1,b)';
            tep2(:,8)=tep1(:,8)./tnum-T;
            for t=1:8
                val(t)=(norm(tep2(:,t)))^2;
            end
            [value,index]=min(val);     % min value
            re(i-aj,j-aj)=re(i-aj,j-aj)+exp(-val(index)^2);   %result
            Dis=T+tep2(:,index);
            R1(i-aj,j-aj,:)=reshape(Dis,1,1,b);
            tp(:,:,:)=0;
            tep1(:,:)=0;
            tep2(:,:)=0;
        end
    end
end
sci=re;
sct=R1;
end