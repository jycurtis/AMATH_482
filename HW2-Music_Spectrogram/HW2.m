%% GNR Guitar Score
clear; close all; clc;

[y, Fs] = audioread('GNR.m4a');
trgnr = length(y)/Fs;
t = linspace(0,trgnr,length(y));
k = (1/trgnr)*[0:length(y)/2-1 -length(y)/2:-1];
ks = fftshift(k);
S = y';

a = 50;
tau = 0:0.2:trgnr;

for j = 1:length(tau)
    g = exp(-a*(t - tau(j)).^2);
    Sg = g.*S;
    Sgt = fft(Sg);
    Sgt_spec(:,j) = fftshift(abs(Sgt));
end

pcolor(tau,ks,Sgt_spec)
shading interp
set(gca,'ylim',[200 800],'Fontsize',16)
colormap(hot)
colorbar
xlabel('time (t)'), ylabel('frequency (k)')
title("Guns N' Roses Clip Spectrogram")
print('GNR_spectrogram','-dpng')

%% Floyd Bass Score
clear; close all; clc;

[y, Fs] = audioread('Floyd.m4a');
y = y(1:(length(y)-1)/2);
trgnr = length(y)/Fs;
t = linspace(0,trgnr,length(y));
k = (1/trgnr)*[0:length(y)/2-1 -length(y)/2:-1];
ks = fftshift(k);
S = y';

a = 10;
tau = 0:0.5:trgnr;

for j = 1:length(tau)
    g = exp(-a*(t - tau(j)).^2);
    Sg = g.*S;
    Sgt = fft(Sg);
    Sgt_spec(:,j) = fftshift(abs(Sgt));
end

pcolor(tau,ks,log(abs(Sgt_spec)+1))
shading interp
set(gca,'ylim',[0 300],'Fontsize',16)
colormap(hot)
colorbar
xlabel('time (t)'), ylabel('frequency (k)')
print('Floyd_spectrogram','-dpng')

%% Floyd Bass Isolation
clear; close all; clc;

[y, Fs] = audioread('Floyd.m4a');
y = y(1:(length(y)-1)/4);
trgnr = length(y)/Fs;
t = linspace(0,trgnr,length(y));
k = (1/trgnr)*[0:length(y)/2-1 -length(y)/2:-1];
ks = fftshift(k);
S = y';

Fa = 0.001;
Ftau = 100;
f = exp(-Fa*(ks - Ftau).^2);

Ga = 10;
Gtau = 0:0.5:trgnr;
for j = 1:length(Gtau)
    g = exp(-Ga*(t - Gtau(j)).^2);
    Sg = g.*S;
    Sgt = fftshift(fft(Sg));
    Sgtf = f.*Sgt;
    Sgt_spec(:,j) = Sgtf;
end

pcolor(Gtau,ks,log(abs(Sgt_spec)+1))
shading interp
set(gca,'ylim',[0 300],'Fontsize',16)
colormap(hot)
colorbar
xlabel('time (t)'), ylabel('frequency (k)')
print('Floyd_bass_spectrogram','-dpng')

%% Floyd Inverse Gabor
Si = ks*0;
for j = 1:length(Gtau)
    g = exp(-Ga*(t - Gtau(j)).^2);
    Sgti = ifftshift(ifft(Sgt_spec(:,j)))';
    Si = Si + Sgti;
end
plot(t,Si);
p8 = audioplayer(Si,Fs); playblocking(p8);

%% Floyd Guitar Isolation
clear; close all; clc;

[y, Fs] = audioread('Floyd.m4a');
y = y(1:(length(y)-1)/4);
trgnr = length(y)/Fs;
t = linspace(0,trgnr,length(y));
k = (1/trgnr)*[0:length(y)/2-1 -length(y)/2:-1];
ks = fftshift(k);
S = y';

f = ks;
f(f < 130) = 0;
f(f > 0) = 1 ;

% Fa = 0.0001;
% Ftau = 200;
% f = exp(-Fa*(ks - Ftau).^2);

Ga = 100;
Gtau = 0:0.1:trgnr;
for j = 1:length(Gtau)
   g = exp(-Ga*(t - Gtau(j)).^2);
   Sg = g.*S;
   Sgt = fftshift(fft(Sg));
   Sgtf = f.*Sgt;
   Sgt_spec(:,j) = Sgtf;
end

pcolor(Gtau,ks,abs(Sgt_spec))
shading interp
set(gca,'ylim',[0 1000],'Fontsize',16)
colormap(hot)
colorbar
xlabel('time (t)'), ylabel('frequency (k)')
print('Floyd_guitar_spectrogram','-dpng')