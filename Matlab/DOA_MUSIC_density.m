
clear; 
clc;
close all;

%MUSIC_Function = 'Spatial_Smoothing';
MUSIC_Function = 'Toeplitz_Construction';

%% 输入参数
% % high density
% G = 10;              % 目标数目	
% B=eye(G); % 衰减系数矩阵
% dist = [10:1:10+G-1]; 	% 待估计距离
% theta = [-60:5:-60+5*(G-1)]; % 待估计角度
% SNR = 10;           % 信噪比

% % 距离、角度均稀疏
% G = 5;              % 目标数目	
% B=eye(G); % 衰减系数矩阵
% dist = [10:4:10+4*(G-1)]; 	% 待估计距离
% theta = [-60:20:-60+20*(G-1)]; % 待估计角度
% SNR = 10;           % 信噪比

% 目标数目少，但距离角度仍接近
G = 5;              % 目标数目	
B=eye(G); % 衰减系数矩阵
dist = [10:1:10+1*(G-1)]; 	% 待估计距离
theta = [-60:5:-60+5*(G-1)]; % 待估计角度
SNR = 10;           % 信噪比

%% 参数设置
c=3e8;
% 阵列参数
derad = pi/180;      % 角度->弧度
Nr = 16;              % 接收阵元个数 
Nt = 4;              % 发射阵元个数        
ddt = 0.5*Nr;            % 阵元间距 
dt=0:ddt:(Nt-1)*ddt;
At=exp(-1i*2*pi*dt.'*sin(derad*theta));  %方向矢量（表示的是经K个目标反射后的方向）
      
ddr = 0.5;            % 接收阵元间距 
dr=0:ddr:(Nr-1)*ddr;
Ar=exp(-1i*2*pi*dr.'*sin(derad*theta));  %方向矢量（表示的是经K个目标反射后的方向）

% 组成的虚拟阵列
A = zeros(Nt*Nr,G); % at(k)和ar(k)的Kronecker积
for g=1:G
    A(:,g)=kron(At(:,g),Ar(:,g));
end
d = 0:ddr:(Nt*Nr-1)*ddr;
%d = 0:ddr:(Nt*Nr/2-1)*ddr;

% OFDM参数
% 这篇文章没有假设每个天线上传输的子载波不同，而是所有天线共享这些子载波
K = 32;               % 总子载波数目
delta_f=6e4;		 % 每个子载波的带宽
% fc=1e5;              % 载波的起始频率
% fp=fc:delta_f:fc+delta_f*(K-1);  % 各个OFDM子载波的频率
T=1/delta_f;         % 每符号间隔

%% 信号产生
M=100;                  % 总传输符号数
n=Nt*K*M;               % 总传送比特数=总传输符号数*子载波数目*天线个数
% 注：这里各个天线之间的子载波分配是重合的，也就是各个天线发射的信号在接收端可能需要特殊方法区分

% 产生传输符号
data = randi([0,3],1,n);        % 产生信源数据
s = reshape(data,Nt,M*K);       % 串/并变换
s = pskmod(s, 4, pi/4);         % 4PSK调制

% qpsk如下
% data=2*round(rand(1,Nt*K*M))-1; %产生信源数据
% s=reshape(data,Nt,M*K); %串/并变换

% 4个发射天线的信号 每行代表一个子载波的所有信号，每列代表每个时刻所有子载波的信号
s1 = reshape(s(1,:),K,M);
s2 = reshape(s(2,:),K,M);
s3 = reshape(s(3,:),K,M);
s4 = reshape(s(4,:),K,M);

%% 构建接收信号模型 
G_all=[];
nt=0:Nt-1;
G_all=[];
nt=0:Nt-1;
%  https://zhuanlan.zhihu.com/p/358985352
% 导频矩阵列满秩
% 对于每个子载波单独处理
for k=1:K
    s_k = [s1(k,:);s2(k,:);s3(k,:);s4(k,:)];
    E=exp(1i*2*pi*k*dist'./T/c); 
    y = Ar*diag(E)*B*At.'*s_k;
    y=awgn(y,SNR,'measured');
    G_k=y*s_k'*inv(s_k*s_k'+10^(-SNR/10)*Nt*eye(Nt));
    G_k=reshape(G_k,1,Nt*Nr);
    G_all=[G_all,G_k.'];            
end

% 计算协方差矩阵
switch(MUSIC_Function) 
    case 'Spatial_Smoothing'
        issp = 1;           %选用的平滑算法模式
        kelm = Nt*Nr/2;
        Rxx = 0;
        J = fliplr(eye(kelm));         %翻转对角阵，中心对称
        for n=1:Nt*Nr-kelm+1
            Rxx = Rxx + G_all(n:n+kelm-1,:)*G_all(n:n+kelm-1,:)'; % 前向平滑
            Rxx = Rxx + J*conj(G_all(n:n+kelm-1,:))*G_all(n:n+kelm-1,:).'*J; % 后向平滑
        end
        Rxx = Rxx/2;
    case 'Toeplitz_Construction'
        Rxx = 0;
        % 对于偶数阵元，加入了左右移动，貌似可能没什么用
        for k=1:K
            x_equal0 = G_all(1:end-1,k);
            x_equal1 = G_all(2:end,k);
            kelm = Nt*Nr/2;
            R_tilde0 = toeplitz(x_equal0(kelm:end), x_equal0(kelm:-1:1));
            R_tilde1 = toeplitz(x_equal1(kelm:end), x_equal1(kelm:-1:1));
            Rxx = Rxx + R_tilde1*R_tilde1'+R_tilde0*R_tilde0;
        end
        Rxx = Rxx / K;
end

% % 计算协方差矩阵
Rxx=G_all*G_all'/K;
% 特征值分解
[EV,D]=eig(Rxx);                   %特征值分解
EVA=diag(D)';                      %将特征值矩阵对角线提取并转为一行
[EVA,I]=sort(EVA);                 %将特征值排序 从小到大
EV=fliplr(EV(:,I));                % 对应特征矢量排序
                 
% 遍历每个角度，计算空间谱
for iang = 1:18001
    angle(iang)=(iang-9001)/100;
    phim=derad*angle(iang);
    a=exp(-1i*2*pi*d*sin(phim)).'; 
    En=EV(:,G+1:Nt*Nr);                   % 取矩阵的第M+1到N列组成噪声子空间
    Pmusic(iang)=1/(a'*En*En'*a);
end
Pmusic=abs(Pmusic);
Pmmax=max(Pmusic);
Pmusic=10*log10(Pmusic/Pmmax);            % 归一化处理
h=plot(angle,Pmusic);
set(h,'Linewidth',2);
xlabel('入射角/(degree)');
ylabel('空间谱/(dB)');
set(gca, 'XTick',[-90:30:90]);
grid on;

peak = my_peak_seek(Pmusic,angle,G);              %寻找峰值
peak = sort(peak)
theta
theta_error_music=abs(peak-sort(theta))   %估计偏差

