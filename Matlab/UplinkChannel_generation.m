clear; 
clc;
close all;

%% Parameters
c=3e8;
derad = pi/180;    
SNR = 0;

% array settings
Nt = 16;
Nr = 16;
ddt = 0.5;
dt=0:0.5:(Nt-1)*0.5;
ddr = 0.5;
dr=0:0.5:(Nr-1)*0.5;
    
% OFDM signal
delta_f=6e4;		 
T=1/delta_f; 
K = 5;
M = 100;
n=Nt*K*M;               
data = randi([0,3],1,n);       
s = reshape(data,Nt,M*K);      
s = pskmod(s, 4, pi/4);         

s1 = reshape(s(1,:),K,M);
s2 = reshape(s(2,:),K,M);
s3 = reshape(s(3,:),K,M);
s4 = reshape(s(4,:),K,M);
s5 = reshape(s(5,:),K,M);
s6 = reshape(s(6,:),K,M);
s7 = reshape(s(7,:),K,M);
s8 = reshape(s(8,:),K,M);
s9 = reshape(s(9,:),K,M);
s10 = reshape(s(10,:),K,M);
s11 = reshape(s(11,:),K,M);
s12 = reshape(s(12,:),K,M);
s13 = reshape(s(13,:),K,M);
s14 = reshape(s(14,:),K,M);
s15 = reshape(s(15,:),K,M);
s16 = reshape(s(16,:),K,M);

for variname=1:1e4
    % target
    G = 6;                          % number
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
       
    % mobile user
    At=exp(-1i*2*pi*dt.'*sin(derad*theta)); 

    % base station
    Ar=exp(-1i*2*pi*dr.'*sin(derad*theta)); 

    kelm = (Nt+Nr)/2;
    d = 0:0.5:(kelm-1)*0.5;
    A=exp(-1i*2*pi*d.'*sin(derad*theta));

    %% Uplink Channel
    for k=1:K
        s_k = [s1(k,:);s2(k,:);s3(k,:);s4(k,:);s5(k,:);s6(k,:);s7(k,:);s8(k,:);s9(k,:);s10(k,:);s11(k,:);s12(k,:);s13(k,:);s14(k,:);s15(k,:);s16(k,:)];
        E=exp(1i*2*pi*k*dist'./T/c); 
        %H_full = Ar*diag(E)*B*At.'; 
        E=exp(1i*2*pi*k*dist'./T/c); 
        y = Ar*diag(E)*B*At.'*s_k;
        y=awgn(y,SNR,'measured');
        H_full=y*s_k'*inv(s_k*s_k'+10^(-SNR/10)*eye(Nt));
        H_split1 = Ar(:,[1,3,5])*diag(E([1,3,5]))*At(:,[1,3,5]).'; 
        H_split2 = Ar(:,[2,4,6])*diag(E([2,4,6]))*At(:,[2,4,6]).'; 
        H_full_spplit12 = [H_full;H_split1;H_split2];
        FILENAME = ['E:\科研\coding\Uplink Sensing\ChannelMatrix\',num2str(variname),'k',num2str(k),'.mat'];
        save(FILENAME, 'H_full_spplit12'); 
    end
end




