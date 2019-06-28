function [beta,alpha] = mlest(x)
    m1 = mean(abs(x));
    m2 = std(x,1);
    beta = m1/m2;
    for i = 1:1024
        beta = beta - g(beta,x);
    end
    alpha = (beta*mean(abs(x(:)).^beta))^(1/beta);
end

function g0_g1 = g(beta,x)
    abs_x_beta = abs(x).^beta;
    sum_abs_x_beta = sum(abs_x_beta);
    log_abs_x = log(abs(x));
    sum_abs_x_beta_log_abs_x = sum(abs_x_beta.*log_abs_x);
    sum_abs_x_beta_log_abs_x_2 = sum(abs_x_beta.*log_abs_x.^2);
    beta_1 = 1/beta;
    beta_2 = beta_1^2;
    beta_3 = beta_1^3;
    N_1 = 1/length(x(:));

    g0 = + psi(0,beta_1)*beta_1 + 1 ...
         - sum_abs_x_beta_log_abs_x/sum_abs_x_beta ...
         + log(beta*N_1*sum_abs_x_beta)*beta_1;

    g1 = - psi(0,beta_1)*beta_2 - psi(1,beta_1)*beta_3 + beta_2 ...
         - sum_abs_x_beta_log_abs_x_2/sum_abs_x_beta ...
         + sum_abs_x_beta_log_abs_x^2/sum_abs_x_beta^2 ...
         + sum_abs_x_beta_log_abs_x/(beta*sum_abs_x_beta) ...
         - log(beta*N_1*sum_abs_x_beta)*beta_2;

    g0_g1 = g0/g1;
end
