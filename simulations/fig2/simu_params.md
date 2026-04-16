# Forest task #

## Environment parameters ##
- 50 steps 
- Step size = 0.1
- Std = 1.0  
- Obs function : 
    - $\mu = tanh(x)$ : full bayes still manages to get out
    - $\mu = clamp(x, -1, 1)$ if $abs(x) \geq 0.5$, $\mu = 0.0$ otherwise : higher contrast between full bayes and commitment model (fixed belief)

## Full bayesian model parameters ## 
- $\beta = 1.0$
- $\sigma^2 = 1.0$
- $a_0 = 2$ 

## Fixed belief model parameters ##
- $\tau = 3.0$
- $\alpha = 0.01$
- $\theta = 0.1$

