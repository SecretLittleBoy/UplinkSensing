%==============角度搜索程序=============
%  输出：
%    out                  <- N个峰对应的角度（按伪谱峰值的从大到小排列）
%  输入：
%    pmusic   <- 频谱向量
%    p_index  <- 对应频谱向量的角度
%    N        <- 待测目标数（前N个峰）
%
%copyright by luning 2021
%======================================

function peak = my_peak_seek(P,p_index,N)

[est_theta,locs] = findpeaks(P,'minpeakheight',-300,'minpeakdistance',8);  % 找到所有大于1dB、间隔超过8的峰值
[~,i] = sort(est_theta);                                                   % 得到真实角度的索引
i = fliplr(i);
locs = locs(i);
peak = p_index(locs(1:N));                                                 % 得到前G个真实角度的峰值

end