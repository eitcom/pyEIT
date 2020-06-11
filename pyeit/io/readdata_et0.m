%% ��ȡET0��ʽ����
[dfile_name,dfile_path] = uigetfile('*.et0');
full_dfile_name = strcat(dfile_path,dfile_name);
fid2=fopen(full_dfile_name,'rb');
fseek(fid2,0,'eof');
fsize = ftell(fid2); %fsize�����ļ���С��
frames=fsize/(1024+4096);%.et4�ļ���1k֡ͷ��256��doubleʵ����256��double�鲿
Voltage_re=zeros(256,frames);
Voltage_Im=zeros(256,frames);
Voltage_Am=zeros(256,frames);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
frewind(fid2);                                          % ���ļ�ͷ
for i=1:frames
   ver(i)=fread(fid2,1,'int32');                        % �汾��
   frame(i)=fread(fid2,1,'uint32');                     % ֡��
   time(i)=fread(fid2,1,'double');                      % ʱ�䣬Vc����1899��12��30�յ����ڵ���������ȷ��us
   time_(:,i)=datevec(time(i)-1);                       % matlab�дӹ�ԪԪ�����𣬹����ý����Ҫ��1900����Ҫ��1
   time_(1,i)=time_(1,i)+1990;  
   tt=fread(fid2,3,'double');                           % ����

   mean_TransImp(i)=fread(fid2,1,'double');             % ԭʼ����ƽ�������迹
   mean_TransImp_Processed(i)=fread(fid2,1,'double');   % Ԥ������ƽ�������迹
   mean_TransImp_Left(i)=fread(fid2,1,'double');        % ����ƽ�������迹
   mean_TransImp_Right(i)=fread(fid2,1,'double');       % ����ƽ�������迹
   mean_TransImp_Left_N(i)=fread(fid2,1,'double');      % �·�����ƽ���迹
   mean_TransImp_Right_N(i)=fread(fid2,1,'double');     % �·�����ƽ���迹
   tt=fread(fid2,34,'double');                          % ����            
   
   %������������Ϣ
   dwDriveDelay(i)=fread(fid2,1,'uint32');              % �����л������΢�
   dwMeasureDelay(i)=fread(fid2,1,'uint32');            % �����л������΢�
   nDrvNode(i)=fread(fid2,1,'int');                     % �����缫��һ�������缫����缫��Ŀ��
   nMeaNode(i)=fread(fid2,1,'int');                     % �����缫��һ�Բ����缫����缫��Ŀ��
   nFrequency(i)=fread(fid2,1,'int');                   % ����������Ƶ�ʣ����ȣ�
   nCurrent(i)=fread(fid2,1,'int');                     % ���������ķ��ȣ�΢����
   nGain(i)=fread(fid2,1,'int');                        % �̿����棨��ţ�
   nElecNum(i)=fread(fid2,1,'int');                     % �缫��Ŀ������
   fElecSize(i)=fread(fid2,1,'float');                  % �缫�ߴ磨���ף�
   fPeriod(i)=fread(fid2,1,'float');                    % ֡���ݼ����������룩
   MeasureInfo(:,:,i)=fread(fid2,[5,16],'int');         % ÿһ��ͨ���Ĳɼ�������5*16��
                                                        % int ReferenceAmplitudeINmV; /// �ο��źŷ�ֵ��mV��     700
                                                        % int ReferenceOffset;        /// �ο��ź�ƫ��           0x83FF
                                                        % int ReferencePhase;         /// �ο��ź�����           0
                                                        % int AnalogPipeGain;         /// ����ģ��ͨ���ķŴ��� 3
                                                        % int DriverCurrentINuA;      /// ����������ֵ��uA��     1000

   %������Ϣ��������360�ֽ�  
   pdElectrodeSQ(:,i)=fread(fid2,16,'double');          % 16���缫�ź�����
   tt=fread(fid2,14,'double');                          % ����
   
   % �������Զ�ȡ�¶���Ϣ
% ��968��������Ϊ �¶�
% 6·��������Ҷ��������£�������
% TempL = orignal_data(121,:);
% TempR = orignal_data(122,:);
% TempReserved1 = orignal_data(123,:);
% TempReserved2 = orignal_data(124,:);
% TempReserved3 = orignal_data(125,:);
% TempReserved4 = orignal_data(126,:);
   Temp(:,i)=fread(fid2,6,'double');  
   yy=fread(fid2,16,'int8'); % ����
   
   Voltage_re(:,i)=fread(fid2,256,'double');
   Voltage_Im(:,i)=fread(fid2,256,'double');
     if (feof(fid2))
       break;
   end
end
