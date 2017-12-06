clear all
close all 
clc
% Load Audio files and Annotations:
% ---------------------------------
% DV = Valence
% DA = Activation/Arousal
% DP = Power
% DE = Anticipation/Expectation
% DI = Intensity


% DFr = Fear
% DAn = Anger
% DHp = Happiness
% DSd = Sadness
% DDg = Disgust
% DCt = Contempt
% DAm = Amusement
% ---------------------------------
y = [];
L_A = [];
L_E = [];
L_P = [];
L_V = [];
L_I = [];

measure = 'User HeadMounted_Poppy';
datapth = '../semaine-database_download_2017-11-22_20_07_36/Sessions/2';
[yy,Fs] = audioread(fullfile(datapth,['2008.12.05.16.03.15_' measure '.wav']));
measure = 'A';
l_A = load(fullfile(datapth,['R5S1TUCPoD' measure '.txt']));
measure = 'E';
l_E = load(fullfile(datapth,['R5S1TUCPoD' measure '.txt']));
measure = 'P';
l_P = load(fullfile(datapth,['R5S1TUCPoD' measure '.txt']));
measure = 'V';
l_V = load(fullfile(datapth,['R5S1TUCPoD' measure '.txt']));
measure = 'I';
l_I = load(fullfile(datapth,['R6S1TUCPoD' measure '.txt']));

y = cat(1,y,yy);
L_A = cat(1,L_A,l_A);
L_E = cat(1,L_E,l_E);
L_P = cat(1,L_P,l_P);
L_V = cat(1,L_V,l_V);
L_I = cat(1,L_I,l_I);

measure = 'User HeadMounted_Poppy';
datapth = '../semaine-database_download_2017-11-22_20_07_36/Sessions/11';
[yy,Fs] = audioread(fullfile(datapth,['2008.12.14.14.47.07_' measure '.wav']));
measure = 'A';
l_A = load(fullfile(datapth,['R5S2TUCPoD' measure '.txt']));
measure = 'E';
l_E = load(fullfile(datapth,['R5S2TUCPoD' measure '.txt']));
measure = 'P';
l_P = load(fullfile(datapth,['R5S2TUCPoD' measure '.txt']));
measure = 'V';
l_V = load(fullfile(datapth,['R5S2TUCPoD' measure '.txt']));
measure = 'I';
l_I = load(fullfile(datapth,['R5S2TUCPoD' measure '.txt']));

y = cat(1,y,yy);
L_A = cat(1,L_A,l_A);
L_E = cat(1,L_E,l_E);
L_P = cat(1,L_P,l_P);
L_V = cat(1,L_V,l_V);
L_I = cat(1,L_I,l_I);

measure = 'User HeadMounted_Spike';
datapth = '../semaine-database_download_2017-11-22_20_07_36/Sessions/3';
[yy,Fs] = audioread(fullfile(datapth,['2008.12.05.16.03.15_' measure '.wav']));
measure = 'A';
l_A = load(fullfile(datapth,['R5S1TUCSpD' measure '.txt']));
measure = 'E';
l_E = load(fullfile(datapth,['R5S1TUCSpD' measure '.txt']));
measure = 'P';
l_P = load(fullfile(datapth,['R5S1TUCSpD' measure '.txt']));
measure = 'V';
l_V = load(fullfile(datapth,['R5S1TUCSpD' measure '.txt']));
measure = 'I';
l_I = load(fullfile(datapth,['R5S1TUCSpD' measure '.txt']));

y = cat(1,y,yy);
L_A = cat(1,L_A,l_A);
L_E = cat(1,L_E,l_E);
L_P = cat(1,L_P,l_P);
L_V = cat(1,L_V,l_V);
L_I = cat(1,L_I,l_I);

measure = 'User HeadMounted_Spike';
datapth = '../semaine-database_download_2017-11-22_20_07_36/Sessions/13';
[yy,Fs] = audioread(fullfile(datapth,['2008.12.19.11.03.11_' measure '.wav']));
measure = 'A';
l_A = load(fullfile(datapth,['R5S3TUCSpD' measure '.txt']));
measure = 'E';
l_E = load(fullfile(datapth,['R5S3TUCSpD' measure '.txt']));
measure = 'P';
l_P = load(fullfile(datapth,['R5S3TUCSpD' measure '.txt']));
measure = 'V';
l_V = load(fullfile(datapth,['R5S3TUCSpD' measure '.txt']));
measure = 'I';
l_I = load(fullfile(datapth,['R5S3TUCSpD' measure '.txt']));

y = cat(1,y,yy);
L_A = cat(1,L_A,l_A);
L_E = cat(1,L_E,l_E);
L_P = cat(1,L_P,l_P);
L_V = cat(1,L_V,l_V);
L_I = cat(1,L_I,l_I);

measure = 'User HeadMounted_Spike';
datapth = '../semaine-database_download_2017-11-22_20_07_36/Sessions/9';
[yy,Fs] = audioread(fullfile(datapth,['2008.12.14.14.47.07_' measure '.wav']));
measure = 'A';
l_A = load(fullfile(datapth,['R5S2TUCSpD' measure '.txt']));
measure = 'E';
l_E = load(fullfile(datapth,['R5S2TUCSpD' measure '.txt']));
measure = 'P';
l_P = load(fullfile(datapth,['R5S2TUCSpD' measure '.txt']));
measure = 'V';
l_V = load(fullfile(datapth,['R5S2TUCSpD' measure '.txt']));
measure = 'I';
l_I = load(fullfile(datapth,['R5S2TUCSpD' measure '.txt']));

y = cat(1,y,yy);
L_A = cat(1,L_A,l_A);
L_E = cat(1,L_E,l_E);
L_P = cat(1,L_P,l_P);
L_V = cat(1,L_V,l_V);
L_I = cat(1,L_I,l_I);

measure = 'User HeadMounted_Prudence';
datapth = '../semaine-database_download_2017-11-22_20_07_36/Sessions/4';
[yy,Fs] = audioread(fullfile(datapth,['2008.12.05.16.03.15_' measure '.wav']));
measure = 'A';
l_A = load(fullfile(datapth,['R5S1TUCPrD' measure '.txt']));
measure = 'E';
l_E = load(fullfile(datapth,['R5S1TUCPrD' measure '.txt']));
measure = 'P';
l_P = load(fullfile(datapth,['R5S1TUCPrD' measure '.txt']));
measure = 'V';
l_V = load(fullfile(datapth,['R5S1TUCPrD' measure '.txt']));
measure = 'I';
l_I = load(fullfile(datapth,['R6S1TUCPrD' measure '.txt']));

y = cat(1,y,yy);
L_A = cat(1,L_A,l_A);
L_E = cat(1,L_E,l_E);
L_P = cat(1,L_P,l_P);
L_V = cat(1,L_V,l_V);
L_I = cat(1,L_I,l_I);

measure = 'User HeadMounted_Prudence';
datapth = '../semaine-database_download_2017-11-22_20_07_36/Sessions/10';
[yy,Fs] = audioread(fullfile(datapth,['2008.12.14.14.47.07_' measure '.wav']));
measure = 'A';
l_A = load(fullfile(datapth,['R5S2TUCPrD' measure '.txt']));
measure = 'E';
l_E = load(fullfile(datapth,['R5S2TUCPrD' measure '.txt']));
measure = 'P';
l_P = load(fullfile(datapth,['R5S2TUCPrD' measure '.txt']));
measure = 'V';
l_V = load(fullfile(datapth,['R5S2TUCPrD' measure '.txt']));
measure = 'I';
l_I = load(fullfile(datapth,['R5S2TUCPrD' measure '.txt']));

y = cat(1,y,yy);
L_A = cat(1,L_A,l_A);
L_E = cat(1,L_E,l_E);
L_P = cat(1,L_P,l_P);
L_V = cat(1,L_V,l_V);
L_I = cat(1,L_I,l_I);

measure = 'User HeadMounted_Prudence';
datapth = '../semaine-database_download_2017-11-22_20_07_36/Sessions/14';
[yy,Fs] = audioread(fullfile(datapth,['2008.12.19.11.03.11_' measure '.wav']));
measure = 'A';
l_A = load(fullfile(datapth,['R5S3TUCPrD' measure '.txt']));
measure = 'E';
l_E = load(fullfile(datapth,['R5S3TUCPrD' measure '.txt']));
measure = 'P';
l_P = load(fullfile(datapth,['R5S3TUCPrD' measure '.txt']));
measure = 'V';
l_V = load(fullfile(datapth,['R5S3TUCPrD' measure '.txt']));
measure = 'I';
l_I = load(fullfile(datapth,['R5S3TUCPrD' measure '.txt']));

y = cat(1,y,yy);
L_A = cat(1,L_A,l_A);
L_E = cat(1,L_E,l_E);
L_P = cat(1,L_P,l_P);
L_V = cat(1,L_V,l_V);
L_I = cat(1,L_I,l_I);



measure = 'User HeadMounted_Obadiah';
datapth = '../semaine-database_download_2017-11-22_20_07_36/Sessions/5';
[yy,Fs] = audioread(fullfile(datapth,['2008.12.05.16.03.15_' measure '.wav']));
measure = 'A';
l_A = load(fullfile(datapth,['R5S1TUCObD' measure '.txt']));
measure = 'E';
l_E = load(fullfile(datapth,['R5S1TUCObD' measure '.txt']));
measure = 'P';
l_P = load(fullfile(datapth,['R5S1TUCObD' measure '.txt']));
measure = 'V';
l_V = load(fullfile(datapth,['R5S1TUCObD' measure '.txt']));
measure = 'I';
l_I = load(fullfile(datapth,['R5S1TUCObD' measure '.txt']));

y = cat(1,y,yy);
L_A = cat(1,L_A,l_A);
L_E = cat(1,L_E,l_E);
L_P = cat(1,L_P,l_P);
L_V = cat(1,L_V,l_V);
L_I = cat(1,L_I,l_I);

measure = 'User HeadMounted_Obadiah';
datapth = '../semaine-database_download_2017-11-22_20_07_36/Sessions/8';
[yy,Fs] = audioread(fullfile(datapth,['2008.12.14.14.47.07_' measure '.wav']));
measure = 'A';
l_A = load(fullfile(datapth,['R5S2TUCObD' measure '.txt']));
measure = 'E';
l_E = load(fullfile(datapth,['R5S2TUCObD' measure '.txt']));
measure = 'P';
l_P = load(fullfile(datapth,['R5S2TUCObD' measure '.txt']));
measure = 'V';
l_V = load(fullfile(datapth,['R5S2TUCObD' measure '.txt']));
measure = 'I';
l_I = load(fullfile(datapth,['R6S2TUCObD' measure '.txt']));

y = cat(1,y,yy);
L_A = cat(1,L_A,l_A);
L_E = cat(1,L_E,l_E);
L_P = cat(1,L_P,l_P);
L_V = cat(1,L_V,l_V);
L_I = cat(1,L_I,l_I);

measure = 'User HeadMounted_Obadiah';
datapth = '../semaine-database_download_2017-11-22_20_07_36/Sessions/15';
[yy,Fs] = audioread(fullfile(datapth,['2008.12.19.11.03.11_' measure '.wav']));
measure = 'A';
l_A = load(fullfile(datapth,['R5S3TUCObD' measure '.txt']));
measure = 'E';
l_E = load(fullfile(datapth,['R5S3TUCObD' measure '.txt']));
measure = 'P';
l_P = load(fullfile(datapth,['R5S3TUCObD' measure '.txt']));
measure = 'V';
l_V = load(fullfile(datapth,['R5S3TUCObD' measure '.txt']));
measure = 'I';
l_I = load(fullfile(datapth,['R5S3TUCObD' measure '.txt']));

y = cat(1,y,yy);
L_A = cat(1,L_A,l_A);
L_E = cat(1,L_E,l_E);
L_P = cat(1,L_P,l_P);
L_V = cat(1,L_V,l_V);
L_I = cat(1,L_I,l_I);





%% Pre processing:

% For speech Fs~16KHz is sufficient:
y = resample(y,1,3); 
Fs = Fs/3;

% Pre-emphasis high-pass filtering:
alpha = 0.97;
yp = y-alpha*cat(1,0,y(1:end-1));


%% feature extraction:
WL = 35;%(ms)
Ov = 10;%(ms)
LF = 0;
HF = 8000;
R = [LF HF];
M = 26;
N = 12;
L = 22;

%--------------------------------------------------------------------------%
%MFCC 1-12:

[ MFCCs, FBE, frames ] = mfcc( y, Fs, WL, WL-Ov, alpha, @hamming, R, M, N, L );

[ Nw, NF ] = size( frames );                
time_frames = [0:NF-1]*(WL-Ov)*0.001+0.5*Nw/Fs;  
time = [ 0:length(y)-1 ]/Fs; 
logFBEs = 20*log10( FBE );                
logFBEs_floor = max(logFBEs(:))-50;         % get logFBE floor 50 dB below max
logFBEs( logFBEs<logFBEs_floor ) = logFBEs_floor; % limit logFBE dynamic range

figure;
imagesc( time_frames, [1:N], MFCCs(2:end,:) ); 
axis( 'xy' );
xlim( [ min(time_frames) max(time_frames) ] );
xlabel( 'Time (s)' ); 
ylabel( 'Cepstrum index' );
title( 'Mel frequency cepstrum' );

% Set color map to grayscale
colormap( 1-colormap('gray') );

figure;
imagesc( time_frames, [1:M], logFBEs ); 
axis( 'xy' );
xlim( [ min(time_frames) max(time_frames) ] );
xlabel( 'Time (s)' ); 
ylabel( 'Channel index' ); 
title( 'Log (mel) filterbank energies'); 

% Set color map to grayscale
colormap( 1-colormap('gray') );

%--------------------------------------------------------------------------%
%RMS:
% FRMS = rms(frames);
%--------------------------------------------------------------------------%
%F0 and Subharmonic-to-harmonic ratio:
F0 = zeros(size(frames,2),1);
[b0,a0]=butter(2,325/(Fs/2));
for i=1:size(frames,2)
    x=frames(:,i);
    xin = abs(x);
    xin=filter(b0,a0,xin);
    xin = xin-mean(xin);
    x2=zeros(length(xin),1);
    x2(1:length(x)-1)=xin(2:length(x));
    zc=length(find((xin>0 & x2<0) | (xin<0 & x2>0)));
    F0(i)=0.5*Fs*zc/length(x);
end

[f0_time,f0_value,SHR,f0_candidates]=shrp(y,Fs,[50 550],WL,WL-Ov,0.4,1250,0,1); 

%--------------------------------------------------------------------------%
%Total feature vector:

FV = cat(1,MFCCs,FBE,f0_value',SHR');
smthed = tsmovavg(FV,'t',3);
FV(:,3:end-1) = smthed(:,3:end-1);


%--------------------------------------------------------------------------%
%Delta regression:

FV_lag_1 = cat(2,FV(:,1),FV(:,1:end-1));
FV_FF_1 = cat(2,FV(:,2:end),FV(:,end));


FV_lag_2 = cat(2,FV(:,1),FV(:,1),FV(:,1:end-2));
FV_FF_2 = cat(2,FV(:,3:end),FV(:,end),FV(:,end));

temp_denom = 2*(1^2+2^2);
temp_nom = 1*(FV_FF_1-FV_lag_1)+2*(FV_FF_2-FV_lag_2);

Delta_1 = temp_nom./temp_denom;

FV = cat(1,FV,Delta_1);
%--------------------------------------------------------------------------%
% Functionals:
w = 3;
FF = zeros(size(FV,1)*4,size(FV,2)-2*w);
for i=1+w:size(FV,2)-w
    temp = [];
%          [M,I] = max(FV(:,i-w:i+w)'); 
%          temp = cat(1,temp,[M';I']);
%          [m,I] = min(FV(:,i-w:i+w)'); 
%          temp = cat(1,temp,[M';I';(M-m)']);
         S = std(FV(:,i-w:i+w),0,2);
         temp = cat(1,temp,S);
         S = skewness(FV(:,i-w:i+w),0,2);
         temp = cat(1,temp,S);
         S = kurtosis(FV(:,i-w:i+w),1,2);
         temp = cat(1,temp,S);
         S = mean(FV(:,i-w:i+w),2);
         temp = cat(1,temp,S);
     FF(:,i-w) = temp;
end

FV = cat(2,FF(:,1:w),FF,FF(:,end-w+1:end));
FV = FV(sum(isnan(FV),2)==0,:);
save Features.mat FV














