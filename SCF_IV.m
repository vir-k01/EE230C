function [ Vout , J] = IVeta (U0 , kT , m , rs, rd , W , Vg , alphaG , alphaD )
%% Constants
hbar = 1.055 e -34; % [J s]
q = 1.602 e -19; % [C]
hbareV = hbar /q; % [ eV ]
ns0 = m* kT /(2* pi * hbareV ^2) ;
vt = sqrt (2* kT /( pi *m));
%% Energy grid
NE = 501;
E = linspace ( -1 ,1 , NE );
dE = E (2) -E (1) ;
D = (m /(2* pi * hbareV ^2) ); % Lorentzian Density of states per eV
D = D ./( dE * sum (D)); % Normalizing to one
%% Bias
IV = 500;
VV = linspace (0 ,1 , IV );
for iV = 1: IV
Vd = VV ( iV );
eta0 = -(( alphaG * Vg ) +( alphaD * Vd))/ kT ;
etas = 0;
dU = 1;
while dU > 1e -6
f = 1./(1+ exp (- eta0 + etas ));
N( iV ) = dE * sum (D .* f);
etaNew = U0 *N( iV );
dU = abs ( etas - etaNew );
etas = etas +0.0001*( etaNew - etas );
end
J( iV ) = W* ns0 * vt * exp ( etas ) *( 1- rs - (1 - rd )* exp (- Vd / kT ) );
Vout = VV ;
end
end
