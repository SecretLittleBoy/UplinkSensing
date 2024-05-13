clear; 
clc;
close all;

%% Parameters
c=3e8;
derad = pi/180;    
MUSIC_Function = 'Spatial_Smoothing';
%MUSIC_Function = 'Toeplitz_Construction';

% target
G = 3;                          % number
B=eye(G);                       % attenuation
dist = 100+10*randn(G,1).';     % dist
theta = 30+5*randn(G,1).';      % DOA
% check if too close -> new DOAs
    theta_sort = sort(theta);
    theta_check = [theta_sort,0]-[0,theta_sort];
    theta_check = sort(theta_check);
    while(theta_check(2)<3)
        theta = 30+5*randn(G,1).';      % DOA
        theta_sort = sort(theta);
        theta_check = [theta_sort,0]-[0,theta_sort];
        theta_check = sort(theta_check);
    end
    
SNR = -10;                       % SNR

% mobile user
Nt = 16;
ddt = 0.5;
dt=0:0.5:(Nt-1)*0.5;
At=exp(-1i*2*pi*dt.'*sin(derad*theta)); 

% base station
Nr = 16;
ddr = 0.5;
dr=0:0.5:(Nr-1)*0.5;
Ar=exp(-1i*2*pi*dr.'*sin(derad*theta)); 

kelm = (Nt+Nr)/2;
d = 0:0.5:(kelm-1)*0.5;
A=exp(-1i*2*pi*d.'*sin(derad*theta));

% OFDM signal
K = 100;                % number of subcarriers  
delta_f=6e4;		 
T=1/delta_f;           
M=10;                  % number of symbols
n=Nt*K*M;               
data = randi([0,3],1,n);        
s = reshape(data,K,M*Nt);      
s = pskmod(s, 4, pi/4);        

%% Uplink Channel
G_all=[];
for k=1:K
    s_k = reshape(s(k,:),Nt,M);
    E=exp(1i*2*pi*k*dist'./T/c); 
    H = Ar*diag(E)*B*At.';
    y = H * s_k;
    y = awgn(y,SNR,'measured');
    G_k=y*s_k'*inv(s_k*s_k'+10^(-SNR/10)*eye(Nt));
    G_k_reshape = zeros(Nt+Nr-1,1);
    for i_nt = 0:Nt-1
        gk = [zeros(i_nt,1);G_k(:,i_nt+1);zeros(Nt-1-i_nt,1)];
        G_k_reshape = G_k_reshape + gk;
    end    
    G_all = [G_all,G_k_reshape];   
end

%% DOA test

switch(MUSIC_Function) 
    case 'Spatial_Smoothing'
        issp = 1;           
        Rxx = 0;
        J = fliplr(eye(kelm));        
        for n=1:Nt+Nr-1-kelm+1
            Rxx = Rxx + G_all(n:n+kelm-1,:)*G_all(n:n+kelm-1,:)'; 
            Rxx = Rxx + J*conj(G_all(n:n+kelm-1,:))*G_all(n:n+kelm-1,:).'*J; 
        end
        Rxx = Rxx/2;
    case 'Toeplitz_Construction'
        Rxx = 0;
        for k=1:K
            x_equal0 = G_all(1:end-1,k);
            x_equal1 = G_all(2:end,k);
            R_tilde0 = toeplitz(x_equal0(kelm:end), x_equal0(kelm:-1:1));
            R_tilde1 = toeplitz(x_equal1(kelm:end), x_equal1(kelm:-1:1));
            Rxx = Rxx + R_tilde1*R_tilde1'+R_tilde0*R_tilde0;
        end
        Rxx = Rxx / K;
end

[EV,D]=eig(Rxx);                 
EVA=diag(D)';                    
[EVA,I]=sort(EVA);               
EV=fliplr(EV(:,I));               

%% MUSIC
% for iang = 1:18001
%     angle(iang)=(iang-9001)/100;
%     phim=derad*angle(iang);
%     a=exp(-1i*2*pi*d*sin(phim)).';
%     % a=exp(-1i*2*pi*d*sin(phim)).'; 
%     En=EV(:,G+1:kelm);                   
%     Pmusic(iang)=1/(a'*En*En'*a);
% end
% Pmusic=abs(Pmusic);
% Pmmax=max(Pmusic);
% Pmusic=10*log10(Pmusic/Pmmax);            % 归一化处理
% h=plot(angle,Pmusic);
% set(h,'Linewidth',2);
% xlabel('入射角/(degree)');
% ylabel('空间谱/(dB)');
% set(gca, 'XTick',[-90:30:90]);
% grid on;
% 
% theta_est = my_peak_seek(Pmusic,angle,G);
% theta_est = sort(theta_est)
% theta
% theta_error = abs(sort(theta_est)-sort(theta))   

%% Esprit               
estimates=(tls_esprit(ddr,Rxx,G));
theta_sort = sort(theta)
theta_est=sort(estimates(1,:))
theta_error = abs(sort(theta_est)-sort(theta))   %估计偏差  

%% Root-MUSIC
% En=EV(:,G+1:kelm);              % 取矩阵的第M+1到N列组成噪声子空间
% syms z
% pz = z.^([0:kelm-1]');
% pz1 = (z^(-1)).^([0:kelm-1]);
% fz = z.^(kelm-1)*pz1*En*En'*pz;
% a = sym2poly(fz);
% a1 = roots(a);                                             %使用ROOTS函数求出多项式的根 
% a2=a1(abs(a1)<1);                                          %找出在单位圆里且最接近单位圆的N个根
% [lamda,I]=sort(abs(abs(a2)-1));                            %挑选出最接近单位圆的N个根
% f=a2(I(1:G));                                              %计算信号到达方向角
% theta_est=-asin(angle(f(1:G))/pi)*180/pi;
% theta_est=sort(theta_est.')
% theta_error = abs(sort(theta_est)-sort(theta))   %估计偏差
