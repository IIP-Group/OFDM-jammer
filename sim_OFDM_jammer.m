% ==========================================================================
% -- Single-Antenna Jammers in MIMO-OFDM Can Resemble Multi-Antenna Jammers
% --------------------------------------------------------------------------
% -- (c) 2023 Gian Marti
% -- e-mail: gimarti@ethz.ch
% --------------------------------------------------------------------------
% -- If you use this simulator or parts of it, then you must cite our paper:
%
% -- Gian Marti and Christoph Studer,
% -- "Single-Antenna Jammers in MIMO-OFDM Can Resemble Multi-Antenna Jammers," 
% -- IEEE Communication Letters, 2023
% ==========================================================================

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% configurable parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%
par.B = 8;                    % number of receive antennas
par.U = 2;                    % number of transmit antennas and data streams
par.L = 4;                    % number of channel taps
par.P = 16;                   % cyclic prefix length
par.N_sc = 64;                % number of total subcarriers
par.tonelist = [-26:-22, -20:-8, -6:-1, 1:6, 8:20, 22:26] + 32; % list of used subcarriers, only meaningful for par.N_sc = 64;
par.num_tones = length(par.tonelist);
par.num_ofdm_symbols = 50;    % number of OFDM symbols sent per channel realization
par.rho_dB = 30;              % jammer strength relative to signal strength (in dB)
par.mod = 'QPSK';             % transmit constellation
par.random_jammer_power = 0;  % if true, then the jammer-power is uniformly (in dB) sampled from 
                              % the range [0, par.rho_dB] for every trial. 
                              % Otherwise, it is deterministically equal par.rho_dB.

par.trials = 100;             % number of Monte-Carlo trials (different channel realizations)
par.SNRdB_list = -5:1:20;     % list of SNR values (in dB) to be simulated

par.plot = 1;                 % whether or not to plot the results
par.printmsg = 1;             % whether or not to print log messages

rng(0);                       % random seed


%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% simulation start
%%%%%%%%%%%%%%%%%%%%%%%%%%%

par.simName = [num2str(par.B) 'x' num2str(par.U) '_' par.mod '_L', num2str(par.L) '_P' num2str(par.P) ...
  '_Nsc' num2str(par.N_sc) '_Nofd' num2str(par.num_ofdm_symbols) '_rho' num2str(par.rho_dB) ...
  '_' num2str(par.trials) 'Trials'];

switch (par.mod)
  case 'BPSK'
    par.symbols = [ -1 1 ];
  case 'QPSK'
    par.symbols = [ ...
      -1-1i,-1+1i, ...
      +1-1i,+1+1i ];
  case '16QAM'
    par.symbols = [ ...
      -3-3i,-3-1i,-3+3i,-3+1i, ...
      -1-3i,-1-1i,-1+3i,-1+1i, ...
      +3-3i,+3-1i,+3+3i,+3+1i, ...
      +1-3i,+1-1i,+1+3i,+1+1i ];
  case '64QAM'
    par.symbols = [ ...
      -7-7i,-7-5i,-7-1i,-7-3i,-7+7i,-7+5i,-7+1i,-7+3i, ...
      -5-7i,-5-5i,-5-1i,-5-3i,-5+7i,-5+5i,-5+1i,-5+3i, ...
      -1-7i,-1-5i,-1-1i,-1-3i,-1+7i,-1+5i,-1+1i,-1+3i, ...
      -3-7i,-3-5i,-3-1i,-3-3i,-3+7i,-3+5i,-3+1i,-3+3i, ...
      +7-7i,+7-5i,+7-1i,+7-3i,+7+7i,+7+5i,+7+1i,+7+3i, ...
      +5-7i,+5-5i,+5-1i,+5-3i,+5+7i,+5+5i,+5+1i,+5+3i, ...
      +1-7i,+1-5i,+1-1i,+1-3i,+1+7i,+1+5i,+1+1i,+1+3i, ...
      +3-7i,+3-5i,+3-1i,+3-3i,+3+7i,+3+5i,+3+1i,+3+3i ];
end

% normalize transmission constellation
par.constellation_energy = mean(abs(par.symbols).^2);
par.symbols = par.symbols/sqrt(par.constellation_energy);
par.Es = mean(abs(par.symbols).^2); % equal to 1

% precompute bit labels
par.Q = log2(length(par.symbols)); % number of bits per symbol
par.bits = de2bi(0:length(par.symbols)-1,par.Q,'left-msb');

% track simulation time
time_elapsed = 0;

% intermediate results
y_jl_f = zeros(par.B,par.N_sc,par.num_ofdm_symbols);
y_oj_comp_f = zeros(par.B,par.N_sc,par.num_ofdm_symbols);
y_oj_noncomp_f = zeros(par.B,par.N_sc,par.num_ofdm_symbols);
y_comp_f = zeros(par.B,par.N_sc,par.num_ofdm_symbols);
y_noncomp_f = zeros(par.B,par.N_sc,par.num_ofdm_symbols);

% results
Y_oj_comp = zeros(par.trials, par.num_tones,par.B, par.num_ofdm_symbols);
Y_oj_noncomp = zeros(par.trials, par.num_tones,par.B, par.num_ofdm_symbols);

MSE_jl = zeros(1, length(par.SNRdB_list));
MSE_comp = zeros(1, length(par.SNRdB_list));
MSE_noncomp = zeros(1, length(par.SNRdB_list));

res.BER_jl = zeros(1, length(par.SNRdB_list));
res.BER_comp = zeros(1, length(par.SNRdB_list));
res.BER_noncomp = zeros(1, length(par.SNRdB_list));
res.BER_projs = zeros(par.L, length(par.SNRdB_list));

res.spatial_interference_distr = zeros(par.B,par.num_tones);

% generate bit stream
bits = randi([0 1],par.trials,par.num_ofdm_symbols,par.U,par.Q,par.num_tones);
idx = zeros(par.trials, par.num_ofdm_symbols, par.U,par.num_tones);


% trials loop
tic
for t=1:par.trials

  % generate jammer's channel in the time domain
  j_ = sqrt(0.5)*randn(par.B,par.L) + 1j*sqrt(0.5)*randn(par.B,par.L);
  j_f = (1/sqrt(par.N_sc))*fft(j_,par.N_sc,2);


  % generate the UEs' channel in the time domain
  H_ = sqrt(0.5)*randn(par.B,par.U,par.L) + 1j*sqrt(0.5)*randn(par.B,par.U,par.L);
  H_f = (1/sqrt(par.N_sc))*fft(H_,par.N_sc,3); % E[||H_f(:,:,i)||_F^2] = B*U*L/N_sc for all i=1,...,N_sc
  % so the expected receive signal energy per subcarrier is E_s*B*U*L/N_sc = B*U*L/N_sc

  if par.random_jammer_power
    par.rho = par.U*10^(par.rho_dB*rand()/10);
  else
    par.rho = par.U*10^(par.rho_dB/10);
  end

  for snrindex = 1:length(par.SNRdB_list)
    
    par.N0 = par.L*par.U*10^(-par.SNRdB_list(snrindex)/10); 
    
    for k=1:par.num_ofdm_symbols
  
      %%% generate transmit signals
        
      % generate jammer's transmit signal in the frequency domain for OFDM-compliant case
      w_f = sqrt(0.5)*randn(1,par.N_sc)+ 1j*sqrt(0.5)*randn(1,par.N_sc);
      w_no_cp = sqrt(par.N_sc)*ifft(w_f); %this does in fact not change the Gaussian statistics
      w = [w_no_cp(:,par.N_sc-par.P+1:par.N_sc), w_no_cp];
      w = sqrt(par.rho)*w;
  
      % generate the jammer's transmit signal in the time domain for OFDM-noncompliant case
      w_noncomp = sqrt(0.5)*randn(1,par.N_sc+par.P)+ 1j*sqrt(0.5)*randn(1,par.N_sc+par.P);
      w_noncomp = sqrt(par.rho)*w_noncomp;       

      % generate UEs' transmit signal in the frequency domain for OFDM-compliant case
      for n=1:par.num_tones
        idx(t,k,:,n) = bi2de(bits(t,k,:,:,n),'left-msb')+1;
      end
      S_f_used_tones = par.symbols(squeeze(idx(t,k,:,:))); % symbol vectors
      S_f = zeros(par.U, par.N_sc);
      S_f(:,par.tonelist) = S_f_used_tones; % only transmit signal over the used subcarriers
      S_no_cp = sqrt(par.N_sc)*ifft(S_f,par.N_sc,2);
      S = [S_no_cp(:,par.N_sc-par.P+1:par.N_sc), S_no_cp];

    
      %%% generate receive signals
      y_jl = zeros(par.B, par.N_sc+par.P+par.L-1);          % jammerless receive signal
      y_oj_comp = zeros(par.B, par.N_sc+par.P+par.L-1);     % only compliant jammer
      y_oj_noncomp = zeros(par.B, par.N_sc+par.P+par.L-1);  % only noncompliant jammer
      y_comp = zeros(par.B, par.N_sc+par.P+par.L-1);        % full receive signal, jammer compliant
      y_noncomp = zeros(par.B, par.N_sc+par.P+par.L-1);     % full receive signal, jammer noncompliant
    
      for b=1:par.B
        for u=1:par.U
          y_jl(b,:) = y_jl(b,:) + conv(squeeze(H_(b,u,:)),S(u,:));
        end
        y_oj_comp(b,:) = conv(j_(b,:),w);
        y_oj_noncomp(b,:) = conv(j_(b,:),w_noncomp);  
      end
      y_comp = y_jl + y_oj_comp;
      y_noncomp = y_jl + y_oj_noncomp;
    

      %%% generate noise in the time domain
      N = sqrt(0.5)*randn(par.B,par.N_sc)+ 1j*sqrt(0.5)*randn(par.B,par.N_sc);
      N = sqrt(par.N0)*N;

      %%% truncate signals in time-domain to remove cylic prefix and channel reverberation
      y_jl_trunc = y_jl(:,par.P+1:par.P+par.N_sc) + N;
      y_oj_comp_trunc = y_oj_comp(:,par.P+1:par.P+par.N_sc); % no noise 
      y_oj_noncomp_trunc = y_oj_noncomp(:,par.P+1:par.P+par.N_sc); % no noise
      y_comp_trunc = y_comp(:,par.P+1:par.P+par.N_sc) + N;
      y_noncomp_trunc = y_noncomp(:,par.P+1:par.P+par.N_sc) + N;
    
      %%% convert to frequency domain
      y_jl_f(:,:,k) = (1/sqrt(par.N_sc))*fft(y_jl_trunc, par.N_sc, 2);
      y_oj_comp_f(:,:,k) = (1/sqrt(par.N_sc))*fft(y_oj_comp_trunc, par.N_sc, 2);
      y_oj_noncomp_f(:,:,k) = (1/sqrt(par.N_sc))*fft(y_oj_noncomp_trunc, par.N_sc, 2);
      y_comp_f(:,:,k) = (1/sqrt(par.N_sc))*fft(y_comp_trunc, par.N_sc, 2);
      y_noncomp_f(:,:,k) = (1/sqrt(par.N_sc))*fft(y_noncomp_trunc, par.N_sc, 2);


      %%% remove subcarriers that are not used
      y_jl_f_tones = y_jl_f(:,par.tonelist,:);
      y_oj_comp_f_tones = y_oj_comp_f(:,par.tonelist,:);
      y_oj_noncomp_f_tones = y_oj_noncomp_f(:,par.tonelist,:);
      y_comp_f_tones = y_comp_f(:,par.tonelist,:);
      y_noncomp_f_tones = y_noncomp_f(:,par.tonelist,:);

      H_f_tones = H_f(:,:,par.tonelist);
    
    
      %%% collect jammer statistics for later analysis
      for n=1:par.num_tones
        Y_oj_comp(t,n,:,k) = y_oj_comp_f_tones(:,n,k);
        Y_oj_noncomp(t,n,:,k) = y_oj_noncomp_f_tones(:,n,k);    
      end

  
    end % iterate over OFDM symbols with same channel realization


    for n=1:par.num_tones
  
      %%% extract jammer statistics
      [U_comp,S_comp,~] = svd(squeeze(Y_oj_comp(t,n,:,:)));
      [U_noncomp,S_noncomp,~] = svd(squeeze(Y_oj_noncomp(t,n,:,:)));
      res.spatial_interference_distr(:,n) = res.spatial_interference_distr(:,n) + diag(S_noncomp); % interference distribution over different spatial dimensions

      
  
      % form best rank-(B-1) projectors: those that remove most of the
      % jammer interference per subcarrier
      j_est_comp = U_comp(:,1);
      P_comp = eye(par.B) - j_est_comp*j_est_comp';
      j_est_noncomp = U_noncomp(:,1);
      P_noncomp = eye(par.B) - j_est_noncomp*j_est_noncomp';

      Projs = zeros(par.L,par.B,par.B);
      for b=1:par.L
        Projs(b,:,:) = eye(par.B) - U_noncomp(:,1:b)*U_noncomp(:,1:b)';
      end
  
      %%% detect signals using zero-forcing (and, in some cases, jammer mitigation)
      S_est_jl = (sqrt(par.N_sc)*(H_f_tones(:,:,n)))\squeeze(y_jl_f_tones(:,n,:));
      S_est_comp = (sqrt(par.N_sc)*(P_comp*squeeze(H_f_tones(:,:,n))))\(P_comp*squeeze(y_comp_f_tones(:,n,:)));
      S_est_noncomp = (sqrt(par.N_sc)*(P_noncomp*squeeze(H_f_tones(:,:,n))))\(P_noncomp*squeeze(y_noncomp_f_tones(:,n,:)));
      S_est_projs = zeros(par.B,par.U,par.num_ofdm_symbols);
      for b=1:par.L
        S_est_projs(b,:,:) = (sqrt(par.N_sc)*(squeeze(Projs(b,:,:))*squeeze(H_f_tones(:,:,n))))\(squeeze(Projs(b,:,:))*squeeze(y_noncomp_f_tones(:,n,:)));
      end

      % -- compute bit outputs
      S_est_jl_vec = reshape(S_est_jl, [par.num_ofdm_symbols*par.U,1]);
      [~,idxhat_vec] = min(abs(S_est_jl_vec*ones(1,length(par.symbols))-ones(par.num_ofdm_symbols*par.U,1)*par.symbols).^2,[],2);
      idxhat_jl = reshape(idxhat_vec, [par.U, par.num_ofdm_symbols]);
      bithat_jl = par.bits(idxhat_jl,:);

      S_est_comp_vec = reshape(S_est_comp, [par.num_ofdm_symbols*par.U,1]);
      [~,idxhat_vec] = min(abs(S_est_comp_vec*ones(1,length(par.symbols))-ones(par.num_ofdm_symbols*par.U,1)*par.symbols).^2,[],2);
      idxhat_comp = reshape(idxhat_vec, [par.U, par.num_ofdm_symbols]);
      bithat_comp = par.bits(idxhat_comp,:);

      S_est_noncomp_vec = reshape(S_est_noncomp, [par.num_ofdm_symbols*par.U,1]);
      [~,idxhat_vec] = min(abs(S_est_noncomp_vec*ones(1,length(par.symbols))-ones(par.num_ofdm_symbols*par.U,1)*par.symbols).^2,[],2);
      idxhat_noncomp = reshape(idxhat_vec, [par.U, par.num_ofdm_symbols]);
      bithat_noncomp = par.bits(idxhat_noncomp,:);

      bithat_projs = zeros(par.L,par.U*par.num_ofdm_symbols,par.Q);
      for b=1:par.L
        S_est_vec = reshape(S_est_projs(b,:,:), [par.num_ofdm_symbols*par.U,1]);
        [~,idxhat_vec] = min(abs(S_est_vec*ones(1,length(par.symbols))-ones(par.num_ofdm_symbols*par.U,1)*par.symbols).^2,[],2);
        idxhat = reshape(idxhat_vec, [par.U, par.num_ofdm_symbols]);
        bithat_projs(b,:,:) = par.bits(idxhat,:);
      end

      % -- compute error metrics
      bit_tensor = squeeze(bits(t,:,:,:,n));
      bit_tensor = permute(bit_tensor,[3 2 1]);
      true_bits = bit_tensor(:,:)';      
      res.BER_jl(snrindex) = res.BER_jl(snrindex) + sum(sum(true_bits~=bithat_jl));  
      res.BER_comp(snrindex) = res.BER_comp(snrindex) + sum(sum(true_bits~=bithat_comp));
      res.BER_noncomp(snrindex) = res.BER_noncomp(snrindex) + sum(sum(true_bits~=bithat_noncomp));
      for b=1:par.L
        res.BER_projs(b,snrindex) = res.BER_projs(b,snrindex) + sum(sum(true_bits~=squeeze(bithat_projs(b,:,:))));
      end

    end


  end % iterate over SNRs

  % -- keep track of simulation time
  if(par.printmsg)
    if toc>10
      time = toc;
      time_elapsed = time_elapsed + time;
        fprintf('estimated remaining simulation time: %3.0f min.\n', ...
          time_elapsed*(par.trials/t-1)/60);
      tic
    end
  end

end % iterate over MC trials

% normalize results
res.BER_jl = res.BER_jl/(par.trials*par.U*par.num_ofdm_symbols*par.Q*par.num_tones);
res.BER_comp = res.BER_comp/(par.trials*par.U*par.num_ofdm_symbols*par.Q*par.num_tones);
res.BER_noncomp = res.BER_noncomp/(par.trials*par.U*par.num_ofdm_symbols*par.Q*par.num_tones);
res.BER_projs = res.BER_projs/(par.trials*par.U*par.num_ofdm_symbols*par.Q*par.num_tones);

res.spatial_interference_distr = res.spatial_interference_distr./sum(res.spatial_interference_distr,1);
res.spatial_interference_distr_sc_std = std(res.spatial_interference_distr.');
res.spatial_interference_distr_sc_avg = mean(res.spatial_interference_distr.');



%%% plot results
if par.plot

  figure(1)
  semilogy(par.SNRdB_list, res.BER_jl, 'LineWidth', 2.5)
  hold on
  semilogy(par.SNRdB_list, res.BER_comp, 'LineWidth', 2.5)
  semilogy(par.SNRdB_list, res.BER_noncomp, 'LineWidth', 2.5)
  hold off
  grid on
  axis([min(par.SNRdB_list) max(par.SNRdB_list) 1e-4 1])
  xlabel('average SNR per receive antenna [dB]','FontSize',12)  
  ylabel('uncoded bit error-rate (BER)','FontSize',12)
  legend(["Jammerless", "OFDM-compliant jammer", "Cyclic prefix-violating jammer"],'FontSize',12,'Interpreter','none')
  set(gca,'FontSize',12)
  set(gcf,'position',[10,10,400,300])

  figure(2)
  bar(1:par.B,res.spatial_interference_distr_sc_avg)
  hold on
  er = errorbar(1:par.B,res.spatial_interference_distr_sc_avg, ...
    2*res.spatial_interference_distr_sc_std, ...
    2*res.spatial_interference_distr_sc_std);    
  er.Color = [0 0 0];                            
  er.LineStyle = 'none';  
  hold off
  axis([0.25 par.B+0.75 0 1])
  grid on
  xlabel('ordered spatial dimension index')
  ylabel('fraction of receive interference')
  set(gcf,'position',[10,10,400,300])
  
  figure(3)
  semilogy(par.SNRdB_list, res.BER_projs(1,:), 'LineWidth', 2.5)
  hold on
  for b=2:par.L
    semilogy(par.SNRdB_list, res.BER_projs(b,:), 'LineWidth', 2.5)
  end
  hold off
  grid on
  axis([min(par.SNRdB_list) max(par.SNRdB_list) 1e-4 1])
  xlabel('average SNR per receive antenna [dB]','FontSize',12)  
  ylabel('uncoded bit error-rate (BER)','FontSize',12)
  numbers = {};
  for b = 1:par.L
    numbers{b} = "Null " + num2str(b) + " Dimensions";
  end
  legend(numbers,'FontSize',12,'Interpreter','none')
  set(gca,'FontSize',12)
  set(gcf,'position',[10,10,400,300])

end

