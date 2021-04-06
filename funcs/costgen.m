function [H,h]=costgen(T,S,Q,R,dim,x0)
    Qbar=kron(eye(dim.N),Q); 
    H=S'*Qbar*S+kron(eye(dim.N),R);   
    h=S'*Qbar*T*x0;
end