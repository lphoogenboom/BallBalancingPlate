function [H,h]=costgen(P,S,Q,R,dim)
    Qbar=blkdiag(kron(eye(dim.N),Q),zeros(dim.nx)); 
    H=S'*Qbar*S+kron(eye(dim.N),R);   
    h=S'*Qbar*P;
end