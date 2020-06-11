%% 读取ET0格式数据
[dfile_name,dfile_path] = uigetfile('*.et0');
full_dfile_name = strcat(dfile_path,dfile_name);
fid2=fopen(full_dfile_name,'rb');
fseek(fid2,0,'eof');
fsize = ftell(fid2); %fsize就是文件大小！
frames=fsize/(1024+4096);%.et4文件：1k帧头，256个double实部，256个double虚部
Voltage_re=zeros(256,frames);
Voltage_Im=zeros(256,frames);
Voltage_Am=zeros(256,frames);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
frewind(fid2);                                          % 回文件头
for i=1:frames
   ver(i)=fread(fid2,1,'int32');                        % 版本号
   frame(i)=fread(fid2,1,'uint32');                     % 帧号
   time(i)=fread(fid2,1,'double');                      % 时间，Vc中自1899年12月30日到现在的天数，精确到us
   time_(:,i)=datevec(time(i)-1);                       % matlab中从公元元年算起，故所得结果年要加1900，日要减1
   time_(1,i)=time_(1,i)+1990;  
   tt=fread(fid2,3,'double');                           % 保留

   mean_TransImp(i)=fread(fid2,1,'double');             % 原始数据平均传输阻抗
   mean_TransImp_Processed(i)=fread(fid2,1,'double');   % 预处理后的平均传输阻抗
   mean_TransImp_Left(i)=fread(fid2,1,'double');        % 左脑平均传输阻抗
   mean_TransImp_Right(i)=fread(fid2,1,'double');       % 右脑平均传输阻抗
   mean_TransImp_Left_N(i)=fread(fid2,1,'double');      % 新法左脑平均阻抗
   mean_TransImp_Right_N(i)=fread(fid2,1,'double');     % 新法右脑平均阻抗
   tt=fread(fid2,34,'double');                          % 保留            
   
   %以下是配置信息
   dwDriveDelay(i)=fread(fid2,1,'uint32');              % 驱动切换间隔（微妙）
   dwMeasureDelay(i)=fread(fid2,1,'uint32');            % 测量切换间隔（微妙）
   nDrvNode(i)=fread(fid2,1,'int');                     % 驱动电极（一对驱动电极间隔电极数目）
   nMeaNode(i)=fread(fid2,1,'int');                     % 测量电极（一对测量电极间隔电极数目）
   nFrequency(i)=fread(fid2,1,'int');                   % 激励电流的频率（赫兹）
   nCurrent(i)=fread(fid2,1,'int');                     % 激励电流的幅度（微安）
   nGain(i)=fread(fid2,1,'int');                        % 程控增益（序号）
   nElecNum(i)=fread(fid2,1,'int');                     % 电极数目（个）
   fElecSize(i)=fread(fid2,1,'float');                  % 电极尺寸（厘米）
   fPeriod(i)=fread(fid2,1,'float');                    % 帧数据间采样间隔（秒）
   MeasureInfo(:,:,i)=fread(fid2,[5,16],'int');         % 每一个通道的采集参数，5*16。
                                                        % int ReferenceAmplitudeINmV; /// 参考信号幅值（mV）     700
                                                        % int ReferenceOffset;        /// 参考信号偏移           0x83FF
                                                        % int ReferencePhase;         /// 参考信号相移           0
                                                        % int AnalogPipeGain;         /// 测量模拟通道的放大倍数 3
                                                        % int DriverCurrentINuA;      /// 驱动电流峰值（uA）     1000

   %配置信息结束，共360字节  
   pdElectrodeSQ(:,i)=fread(fid2,16,'double');          % 16个电极信号质量
   tt=fread(fid2,14,'double');                          % 保留
   
   % 以下用以读取温度信息
% 自968起被马航定义为 温度
% 6路，左耳，右耳，鼻咽温，膀胱温
% TempL = orignal_data(121,:);
% TempR = orignal_data(122,:);
% TempReserved1 = orignal_data(123,:);
% TempReserved2 = orignal_data(124,:);
% TempReserved3 = orignal_data(125,:);
% TempReserved4 = orignal_data(126,:);
   Temp(:,i)=fread(fid2,6,'double');  
   yy=fread(fid2,16,'int8'); % 保留
   
   Voltage_re(:,i)=fread(fid2,256,'double');
   Voltage_Im(:,i)=fread(fid2,256,'double');
     if (feof(fid2))
       break;
   end
end
