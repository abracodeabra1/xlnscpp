# Alternatives for summation (and related) operations in xlns16

`xlns16.cpp` provides functions that include performing the summation of an array of `xlns16` values:

`xlns16_sum(a,n)` returns the `xlns16` sum of the `n` elements of the `xlns16` array, `a`.

`xlns16_vec_dot(a,b,n)` returns the `xlns16` dot product of the `xlns16` arrays, `a` and `b`.

`xlns16_softmax(a,c,n)` computes the softmax of `a` and outputs it in `c`. 

`xlns16_layernorm(x,out,gamma,beta,n,eps)` computes the layernorm of `x` as `out`.

inline xlns16 xlns16_sum_monte(const xlns16 *a, size_t n) 
inline xlns16 xlns16_vec_dot_monte(const xlns16 *a, const xlns16 *b, size_t n) 
inline void xlns16_softmax_monte(const xlns16 *a, xlns16 *c, size_t n) 
inline void xlns16_layernorm_monte(const xlns16 *x, xlns16 *out,const xlns16 *gamma, const xlns16 *beta,size_t n, float eps) 

xlns16_sum(a,n) 
xlns16_vec_dot(a,b,n) 
xlns16_softmax_monte(a,c,n) 
xlns16_layernorm(x,out,gamma,beta,n,eps) 
