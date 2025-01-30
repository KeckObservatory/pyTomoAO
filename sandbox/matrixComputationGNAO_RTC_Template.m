%%*************************************************************************
%
% Compute the matrix components Gx, Cnn, iCxx, Ha and Hx for the GNAO MCAO
% reconstructor. TBC: T/T/F mode filtering option (see "filter" line 435):
% x_hat = (GxT*filterT*Cnn*filter*Gx +iCxx)^-1 GxT*filterT*Cnn
% a = (HaT*W*Ha)^-1 HaT*W*Hx
%
% Stand alone code extracted from OOMAO library Jan 2021
%
% Required functions: p_bilinearSplineInterp.m
%
%**************************************************************************
%% LGS WFS Parameters
tic
D           = 7.9;            % Telescope diameter
nLenslet    = 20;           % Number of WFS lenslets on a side
dSub        = D/nLenslet;   % Subaperture resolution.
nPx         = 8;            % Number of pixels per lenslet
resolution  = nLenslet*nPx; % Total pixels
fieldStopSize = 4;          % Number of spot FWHMs in lenslet
nLGS        = 4;
validLLMap = [     0     0     0     0     0     0     0     1     1     1     1     1     1     0     0     0     0     0     0     0; ...
    0     0     0     0     0     1     1     1     1     1     1     1     1     1     1     0     0     0     0     0; ...
    0     0     0     1     1     1     1     1     1     1     1     1     1     1     1     1     1     0     0     0; ...
    0     0     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     0     0; ...
    0     0     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     0     0; ...
    0     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     0; ...
    0     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     0; ...
    1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1; ...
    1     1     1     1     1     1     1     1     1     0     0     1     1     1     1     1     1     1     1     1; ...
    1     1     1     1     1     1     1     1     0     0     0     0     1     1     1     1     1     1     1     1; ...
    1     1     1     1     1     1     1     1     0     0     0     0     1     1     1     1     1     1     1     1; ...
    1     1     1     1     1     1     1     1     1     0     0     1     1     1     1     1     1     1     1     1; ...
    1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1; ...
    0     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     0; ...
    0     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     0; ...
    0     0     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     0     0; ...
    0     0     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     0     0; ...
    0     0     0     1     1     1     1     1     1     1     1     1     1     1     1     1     1     0     0     0; ...
    0     0     0     0     0     1     1     1     1     1     1     1     1     1     1     0     0     0     0     0; ...
    0     0     0     0     0     0     0     1     1     1     1     1     1     0     0     0     0     0     0     0];
validLLMap = logical(validLLMap);
nValidLenslet = nnz(validLLMap(:));
validActuatorMap = [0   0   0   0   0   0   0   1   1   1   1   1   1   1   0   0   0   0   0   0   0; ...
    0   0   0   0   0   1   1   1   1   1   1   1   1   1   1   1   0   0   0   0   0; ...
    0   0   0   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   0   0   0; ...
    0   0   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   0   0; ...
    0   0   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   0   0; ...
    0   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   0; ...
    0   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   0; ...
    1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1; ...
    1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1; ...
    1   1   1   1   1   1   1   1   1   1   0   1   1   1   1   1   1   1   1   1   1; ...
    1   1   1   1   1   1   1   1   1   0   0   0   1   1   1   1   1   1   1   1   1; ...
    1   1   1   1   1   1   1   1   1   1   0   1   1   1   1   1   1   1   1   1   1; ...
    1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1; ...
    1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1; ...
    0   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   0; ...
    0   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   0; ...
    0   0   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   0   0; ...
    0   0   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   0   0; ...
    0   0   0   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   0   0   0; ...
    0   0   0   0   0   1   1   1   1   1   1   1   1   1   1   1   0   0   0   0   0; ...
    0   0   0   0   0   0   0   1   1   1   1   1   1   1   0   0   0   0   0   0   0];
validActuatorMap = logical(validActuatorMap);

slopesMask = repmat( repmat( validLLMap(:) , 2, 1), 1, nLGS );
%  define mask for wavefront of WFS
nMap     = 2*nLenslet+1;

[iMap0,jMap0] = ndgrid(1:3);
phaseMask = false(nMap^2,nLGS);
for jLenslet = 1:nLenslet
    jOffset = 2*(jLenslet-1);
    for iLenslet = 1:nLenslet
        index1 = jLenslet+nLenslet*(iLenslet-1);
        iOffset = 2*(iLenslet-1);
        for kGs = 1:nLGS
            if slopesMask(index1,kGs)
                index2 = jMap0+jOffset+nMap*(iMap0+iOffset-1);
                phaseMask(index2(:),kGs) = true;
            end
        end
    end
end
%% DM parameters
dmHeights       = 0;
nDmLayer        = length(dmHeights);
dmPitch         = 0.4;
dmCrossCoupling = 0.15;
nActuators      = 21; % actuators across the DM diameter

validActuators{1} = logical([0   0   0   0   0   1   1   1   1   1   1   1   1   1   1   1   0   0   0   0   0; ...
    0   0   0   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   0   0   0; ...
    0   0   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   0   0; ...
    0   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   0; ...
    0   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   0; ...
    1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1; ...
    1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1; ...
    1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1; ...
    1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1; ...
    1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1; ...
    1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1; ...
    1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1; ...
    1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1; ...
    1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1; ...
    1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1; ...
    1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1; ...
    0   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   0; ...
    0   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   0; ...
    0   0   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   0   0; ...
    0   0   0   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   0   0   0; ...
    0   0   0   0   0   1   1   1   1   1   1   1   1   1   1   1   0   0   0   0   0]);



for i = 1:nDmLayer
    tmp = validActuators{i};
    nValidActuators(i) = nnz(tmp(:));
end

%% Atmosphere Parameters
nLayer              = 7;
zenithAngleInDeg    = 0;
airmass             = 1/cos(zenithAngleInDeg*pi/180);
altitude            = [0,0.5,1,2,6,12,20]*1e3*airmass;
L0                  = 30;
r0                  = 0.186;
fractionnalR0       = [0.4557,0.1295,0.0442,0.0506,0.1167,0.0926,0.1107];
wavelength          = 0.5e-6;
%% LGS Asterism

radiusAst       = 35;
LGSwavelength   = 0.589e-6;
arcsec2radian   = pi/180/3600;
LGSdirections   = [arcsec2radian*radiusAst*sqrt(2) pi/180*225.00;...
    arcsec2radian*radiusAst*sqrt(2) pi/180*135.00; arcsec2radian*radiusAst*sqrt(2)...
    pi/180*315.00; arcsec2radian*radiusAst*sqrt(2) pi/180*45.00];
LGSheight       = 90e3*airmass;

for i = 1:nLGS
    directionVectorLGS(1,i) = tan(LGSdirections(i,1))*cos(LGSdirections(i,2));
    directionVectorLGS(2,i) = tan(LGSdirections(i,1))*sin(LGSdirections(i,2));
    directionVectorLGS(3,i) = 1;
end

%% Optimization Directions
nFitSrc         = 49;
fitSrcHeight    = inf;
x               = linspace(-85/2,85/2,7);
[x, y]          = meshgrid(x);
[theta, rho]    = cart2pol(x,y);
zenithOpt       = rho(:)*arcsec2radian;
azimuthOpt      = theta(:);

for i = 1:nFitSrc
    directionVectorFitSrc(1,i) = tan(zenithOpt(i))*cos(azimuthOpt(i));
    directionVectorFitSrc(2,i) = tan(zenithOpt(i))*sin(azimuthOpt(i));
    directionVectorFitSrc(3,i) = 1;
end

%% Define Atm Grid

% Define spatial grid for layered phase at each altitude

vg              = directionVectorLGS;
vs              = directionVectorFitSrc;
vg              = vg(1:2,:);
vs              = vs(1:2,:);

atmGrid         = cell(nLayer,2);
overSampling    = ones(nLayer,1)*2;

for kLayer = 1:nLayer

    pitchLayer = dSub/overSampling(kLayer);
    height = altitude(kLayer);
    m = 1-height/LGSheight;

    dDirecG = vg*height;
    dDirecS = vs*height;
    dmin = min([dDirecG-D/2*m dDirecS-D/2],[],2);
    dmax = max([dDirecG+D/2*m dDirecS+D/2],[],2);

    nPxLayerX = floor((dmax(1)-dmin(1))/pitchLayer)+2;
    nPxLayerY = floor((dmax(2)-dmin(2))/pitchLayer)+2;

    Dx = (nPxLayerX-1)*pitchLayer;
    Dy = (nPxLayerY-1)*pitchLayer;

    sx = dmin(1)-(Dx-(dmax(1)-dmin(1)))/2;
    sy = dmin(2)-(Dy-(dmax(2)-dmin(2)))/2;

    atmGrid{kLayer,1} = linspace(0,1,nPxLayerX)*Dx+sx;
    atmGrid{kLayer,2} = linspace(0,1,nPxLayerY)*Dy+sy;

end

%% set the sparse gradient matrix using a 3x3 stencil


nMap = 2*nLenslet+1;

i0x = [1:3 1:3]; % x stencil row subscript
j0x = [ones(1,3) ones(1,3)*3]; % x stencil col subscript
i0y = [1 3 1 3 1 3]; % y stencil row subscript
j0y = [1 1 2 2 3 3]; % y stencil col subscript
s0x = [-1 -2 -1  1 2  1]/2; % x stencil weight
s0y = -[ 1 -1  2 -2 1 -1]/2; % y stencil weight

i_x = zeros(1,6*nValidLenslet);
j_x = zeros(1,6*nValidLenslet);
s_x = zeros(1,6*nValidLenslet);
i_y = zeros(1,6*nValidLenslet);
j_y = zeros(1,6*nValidLenslet);
s_y = zeros(1,6*nValidLenslet);

[iMap0,jMap0] = ndgrid(1:3);
gridMask = false(nMap);

u   = 1:6;

% Accumulation of x and y stencil row and col subscript and weight
for jLenslet = 1:nLenslet
    jOffset = 2*(jLenslet-1);
    for iLenslet = 1:nLenslet

        if validLLMap(iLenslet,jLenslet)

            iOffset= 2*(iLenslet-1);
            i_x(u) = i0x + iOffset;
            j_x(u) = j0x + jOffset;
            s_x(u) = s0x;
            i_y(u) = i0y + iOffset;
            j_y(u) = j0y + jOffset;
            s_y(u) = s0y;
            u = u + 6;

            gridMask( iMap0 + iOffset , jMap0 + jOffset ) = true;

        end

    end
end

indx = sub2ind([nMap,nMap],i_x,j_x); % mapping the x stencil subscript into location index on the phase map
indy = sub2ind([nMap,nMap],i_y,j_y); % mapping the y stencil subscript into location index on the phase map

% row index of non zero values in the gradient matrix
v = 1:2*nValidLenslet;
v = v(ones(6,1),:);

% sparse gradient matrix
p_Gamma = sparse(v,[indx,indy],[s_x,s_y],2*nValidLenslet,nMap^2);
p_Gamma(:,~gridMask) = [];


p_Gamma = p_Gamma/2/dSub;
Gamma = cell(nLGS,1);
vL = repmat(validLLMap(:),2,1);
for kGs = 1:nLGS
    Gamma{kGs} = p_Gamma(slopesMask(vL,kGs),...
        phaseMask(gridMask(:),kGs));
end
Gamma = blkdiag(Gamma{:});
%GammaT = Gamma';
%% set propagator H from GS to WFS

[x,y]   = meshgrid(linspace(-.5,.5,2*nLenslet+1)*D);
grid    = x+y*1i;
p_H     = cell(nLGS,nLayer);

for kGs = 1:nLGS
    Hl     = cell(1,nLayer);
    for kLayer = 1:nLayer
        pitchLayer = dSub/overSampling(kLayer);
        height = altitude(kLayer);

        % pupil center in layer
        beta  = directionVectorLGS(:,kGs)*height;
        scale = 1-height/LGSheight;
        Hl{1,kLayer} = p_bilinearSplineInterp(...
            atmGrid{kLayer,1}(:),...
            atmGrid{kLayer,2}(:),...
            pitchLayer,...
            real(grid(phaseMask(:,kGs)))*scale+beta(1),...
            imag(grid(phaseMask(:,kGs)))*scale+beta(2));
    end
    p_H(kGs,:) = Hl;
end
H = cell2mat(p_H);
%HT = H';

%% Gx and GxT

Gx = Gamma*H;
GxT = Gx';


%% iCxx: Set bi-harmonic operator (appox to inverse phase covariance matrix)

% Computes sparse-approximated phase covariance

% 5x5 stencil
a = [0 sqrt(2) 1 sqrt(2) 2;...
    sqrt(2) 0 1 2 sqrt(2);...
    1 1 0 1 1;...
    sqrt(2) 2 1 0 sqrt(2);...
    2 sqrt(2) 1 sqrt(2) 0;];

p_L2 = cell(nLayer,1);
for kLayer = 1:nLayer

    m = length(atmGrid{kLayer,2});
    n = length(atmGrid{kLayer,1});
    N = m*n;

    e = ones(N,1);

    ex1 = e;
    ex1(end-m+1:end) = 2;
    ex1 = circshift(ex1,-m);

    ex2 = e;
    ex2(1:m) = 2;
    ex2 = circshift(ex2,m);

    ey1 = e;
    ey1(mod((1:N),m)==1) = 0;
    ey1(mod((1:N),m)==0) = 2;
    ey1 = circshift(ey1,-1);

    ey2 = e;
    ey2(mod((1:N),m)==1) = 2;
    ey2(mod((1:N),m)==0) = 0;
    ey2 = circshift(ey2,1);

    p_L = spdiags([ex1 ey1 -4*e ey2 ex2],[-m -1 0 1 m],N,N);
    p_L2{kLayer} = p_L'*p_L;


    ex1 = e;
    ex1(end-m+1:end) = 2;
    ex1(1:m) = 0;

    ex2 = e;
    ex2(1:m) = 2;
    ex2(end-m+1:end) = 0;

    ey1 = e;
    ey1(mod((1:N),m)==1) = 0;
    ey1(mod((1:N),m)==0) = 2;

    ey2 = e;
    ey2(mod((1:N),m)==1) = 2;
    ey2(mod((1:N),m)==0) = 0;

    E = [ex1 ey1 -4*e ey2 ex2];
    pitchLayer = dSub/overSampling(kLayer);

    % VARIANCE

    L0r0ratio= (L0./r0).^(5./3);
    outVar   = (24.*gamma(6./5)./5).^(5./6).*(gamma(11./6).*gamma(5./6)./(2.*pi.^(8./3))).*L0r0ratio;
    outVar = fractionnalR0(kLayer).*outVar;

    % COVARIANCE
    rho = a*pitchLayer;
    cst      = (24.*gamma(6./5)./5).^(5./6).*...
        (gamma(11./6)./(2.^(5./6).*pi.^(8./3))).*L0r0ratio;
    outCov   = ones(size(rho)).*(24.*gamma(6./5)./5).^(5./6).*...
        (gamma(11./6).*gamma(5./6)./(2.*pi.^(8./3))).*L0r0ratio;
    index         = rho~=0;
    u             = 2.*pi.*rho(index)./L0;
    outCov(index) = cst.*u.^(5./6).*besselk(5./6,u);
    outCov = fractionnalR0(kLayer).*outCov;

    b = 2.*(outVar-outCov);
    b = b*(wavelength/2/pi)^2;

    E = mat2cell(E,ones(size(E,1),1),5);
    c=sum(cellfun(@(x) x*b*x',E));

    p_L2{kLayer} = N/(-1/2*c)*p_L2{kLayer};
end

iCxx = blkdiag(p_L2{:});


%% Compute reconstruction matrix (x_hat) explicitly with optional T/T or T/T/F filtering

iNoiseVar = 1/1e-14;
wavefrontToMeter = fieldStopSize*LGSwavelength/(D/nLenslet)/(resolution/nLenslet);
%  mask = cell(nLGS,1);
%  for k=1:nLGS
%      mask{k} = slopesMask(1:end/2,k);
%  end
%
%  % filtering
%  TTRM     = 1;
%  FocusRM  = 1;
%  filter     = cell(nLGS,2);
%  if TTRM == true
%      modesRM = [2 3];
%      if FocusRM == true
%          modesRM = [2 3 4];
%      end
%      for k=1:nLGS
%          zer = zernike(1:max(modesRM),'resolution',nLenslet,'pupil',mask{k});
%          zx = zer.xDerivative(mask{k},modesRM);
%          zy = zer.yDerivative(mask{k},modesRM);
%
%          filter{k,1} = eye(sum(mask{k}(:)))-zx*pinv(zx);
%          filter{k,2} = eye(sum(mask{k}(:)))-zy*pinv(zy);
%      end
%  else
%      for k=1:nLGS
%          filter{k,1} = eye(sum(mask{k}(:)));
%          filter{k,2} = eye(sum(mask{k}(:)));
%      end
%  end
%  filter = blkdiag(filter{:});
%
%  if isvector(iNoiseVar)
%      Cnn = diag(iNoiseVar);
% elseif isscalar(iNoiseVar)
Cnn = iNoiseVar*eye(sum(slopesMask(:)));
% else
%     Cnn = iNoiseVar;
% end

%Right = HT*GammaT*filter'*Cnn;
%Left=HT*GammaT*filter'*Cnn*filter*Gamma*H + iCxx;

Right = GxT*Cnn;

Left=GxT*Cnn*Gx + iCxx;
%R = pinv(Left)*Right;


%% Influence matrix (Gaussian IF model)
ratioTelDm = 1;
offset = zeros(2,1);

c = 1/sqrt(log(1/dmCrossCoupling));
df = 1e-10;
mx = sqrt(-log(df)*c^2);
x = linspace(-mx,mx,1001);
f = exp(-x.^2/c^2);

dmInfFuncMatrix = [];
iDmInfFuncMatrix= [];
iFCell          = cell(nDmLayer,1);

layersNPixel    = zeros(nDmLayer,1);
D_m             = zeros(nDmLayer,1);
nValidActuatorTotal = 0;


for kDmLayer = 1:nDmLayer
    gaussian(:,1) = x*dmPitch(kDmLayer);
    gaussian(:,2) = f;
    splineP = spline(x*dmPitch(kDmLayer),f);

    xIF = linspace(-1,1,nActuators(kDmLayer))*(nActuators(kDmLayer)-1)/2*dmPitch(kDmLayer) - offset(1);
    yIF = linspace(-1,1,nActuators(kDmLayer))*(nActuators(kDmLayer)-1)/2*dmPitch(kDmLayer) - offset(2);
    [xIF2,yIF2] = ndgrid(xIF,yIF);
    actCoordVector{kDmLayer} = yIF2 + 1i*flip(xIF2);

    actCoord = actCoordVector{kDmLayer};
    D_m(kDmLayer) = max(real(actCoord(:)))-min(real(actCoord(:)));
    do              = dSub;
    layersNPixel(kDmLayer) = round(D_m(kDmLayer)./do)+1;
    nValidActuatorTotal = nValidActuatorTotal + nValidActuators(kDmLayer);

    u0 = ratioTelDm.*linspace(-1,1,layersNPixel(kDmLayer))*(nActuators(kDmLayer)-1)/2*dmPitch(kDmLayer); % scaled by telescope diamter
    nValid = nValidActuators(kDmLayer);
    kIF = 0;

    u = bsxfun( @minus , u0' , xIF );
    wu = zeros(layersNPixel(kDmLayer),nActuators(kDmLayer));

    index_v = u >= -gaussian(end,1) & u <= gaussian(end,1);
    nu = sum(index_v(:));
    wu(index_v) = ppval(splineP,u(index_v));

    v = bsxfun( @minus , u0' , yIF);
    wv = zeros(layersNPixel(kDmLayer),nActuators(kDmLayer));
    index_v = v >= -gaussian(end,1) & v <= gaussian(end,1);
    nv = sum(index_v(:));
    wv(index_v) = ppval(splineP,v(index_v));

    m_modes = spalloc(layersNPixel(kDmLayer)^2,nValid,nu*nv);

    indIF = 1:nActuators(kDmLayer)^2;
    indIF(~validActuators{kDmLayer}) = [];
    [iIF,jIF] = ind2sub([nActuators(kDmLayer),nActuators(kDmLayer)],indIF);
    kIF = 1:nValid;
    wv = sparse(wv(:,iIF(kIF)));
    wu = sparse(wu(:,jIF(kIF)));
    fprintf(' @(influenceFunction)> Computing the 2D DM zonal modes... (%4d,    \n',nValid)
    for kIF = 1:nValid % parfor doesn't work with sparse matrix!
        fprintf('\b\b\b\b%4d',kIF)
        buffer = wv(:,kIF)*wu(:,kIF)';
        m_modes(:,kIF) = buffer(:);
    end
    fprintf('\n')
    modes{kDmLayer} = m_modes;
    F = 2*modes{kDmLayer};
    dmInfFuncMatrix = blkdiag(dmInfFuncMatrix,F);
    iF = pinv(full(F));
    iFCell{kDmLayer,1} = iF;
    iDmInfFuncMatrix = blkdiag(iDmInfFuncMatrix,iF);
end


%% Hx and Ha
outputWavefrontMask = validActuatorMap;
[x,y]               = meshgrid(linspace(-1,1,nLenslet+1)*D/2);
outputPhaseGrid     = complex(x(outputWavefrontMask),y(outputWavefrontMask));
nStar               = nFitSrc;

Hx                  = cell(nStar,nLayer);
Ha                 = cell(nStar,nDmLayer);
%LeftMean    = sparse(nValidActuatorTotal,nValidActuatorTotal);
%RightMean = sparse(nValidActuatorTotal,length(mmse.Hss(1,:)));

for kGs=1:nStar
    %Hx propagator
    for kAtmLayer = 1:nLayer
        pitchAtmLayer = dSub/overSampling(kAtmLayer);
        height        = altitude(kAtmLayer);
        % pupil center in layer
        beta  = directionVectorFitSrc(:,kGs)*height;
        scale = 1-height/fitSrcHeight;
        Hx{kGs,kAtmLayer} = p_bilinearSplineInterp(...
            atmGrid{kAtmLayer,1}(:),...
            atmGrid{kAtmLayer,2}(:),...
            pitchAtmLayer,...
            real(outputPhaseGrid)*scale+beta(1),...
            imag(outputPhaseGrid)*scale+beta(2));
    end
    %Hdm propagator
    for kDmLayer = 1:nDmLayer
        pitchDmLayer = dSub;
        height       = dmHeights(kDmLayer);
        % pupil center in layer
        beta  = directionVectorFitSrc(:,kGs)*height;
        scale = 1-height/fitSrcHeight;
        actCoord = actCoordVector{kDmLayer};
        dmin = min(real(actCoord(:)));
        dmax = max(real(actCoord(:)));
        Dx = (layersNPixel(kDmLayer)-1)*pitchDmLayer;
        sx = dmin-(Dx-(dmax-dmin))/2;
        [x,y] = meshgrid(linspace(0,1,layersNPixel(kDmLayer))*Dx+sx);
        Ha{kGs,kDmLayer} = p_bilinearSplineInterp(...
            x,...
            y,...
            pitchDmLayer,...
            real(outputPhaseGrid)*scale+beta(1),...
            imag(outputPhaseGrid)*scale+beta(2));
    end
    intHa{kGs} = [Ha{kGs,:}]*dmInfFuncMatrix;
    intHx{kGs}  = [Hx{kGs,:}];
        %LeftMean    = LeftMean    + intHa{kGs}'*intHa{kGs};
        %RightMean = RightMean +  intHa{kGs}'*intHx{kGs};
end

%fittingMatrix = pinv(full(LeftMean),1)*RightMean; % a = (HaT*W*Ha)^-1 HaT*W*Hx where W is equal weights on all directions
toc



function P = p_bilinearSplineInterp(xo,yo,do,xi,yi)
    
    ni = length(xi);
    
    nxo = length(xo);
    nyo = length(yo);
    no = nxo*nyo;
    
    % remove the interaporated points out of the original grid
    mask = xi>=xo(1) & yi>=yo(1) & xi<=xo(end) & yi<=yo(end);
    
    % index for the inteporated grid
    index = (1:ni)';
    index = index(mask);
    
    % x & y index for the original grid
    ox = floor((xi-xo(1))/do)+1;
    ox = ox(mask);
    oy = floor((yi-yo(1))/do)+1;
    oy = oy(mask);

    % bi-linear inteporation value
    fxo = abs(xi(mask)-(xo(1)+do*(ox-1)))/do;
    fyo = abs(yi(mask)-(yo(1)+do*(oy-1)))/do;
    s1 = (1-fxo).*(1-fyo);
    s2 = fxo.*(1-fyo);
    s3 = (1-fxo).*fyo;
    s4 = fxo.*fyo;
    
    % vectoraized index for the original grid
    o1 = oy+nyo*(ox-1);
    o2 = oy+nyo*ox;
    o3 = oy+1+nyo*(ox-1);
    o4 = oy+1+nyo*ox;
    
    % masking
    o1(s1==0)=[];
    i1 = index(s1~=0);
    s1(s1==0)=[];
    o2(s2==0)=[];
    i2 = index(s2~=0);
    s2(s2==0)=[];
    o3(s3==0)=[];
    i3 = index(s3~=0);
    s3(s3==0)=[];
    o4(s4==0)=[];
    i4 = index(s4~=0);
    s4(s4==0)=[];
    
    % intepolation matrix
    P1 = sparse(i1,o1,s1,ni,no);
    P2 = sparse(i2,o2,s2,ni,no);
    P3 = sparse(i3,o3,s3,ni,no);
    P4 = sparse(i4,o4,s4,ni,no);
    
    P = P1+P2+P3+P4;
end
