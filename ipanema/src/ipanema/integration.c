
float quad_gauss_legendre(float (*func)(float), const float a, const float b)
{
  //Returns the integral of the function func between a and b, by ten-point Gauss-Legendre inte- gration: the function is evaluated exactly ten times at interior points in the range of integration. {
  int j;
  float xr,xm,h,s;
  static float x[] = {0.0, 0.1488743389, 0.4333953941,
                      0.6794095682, 0.8650633666, 0.9739065285}; 
  static float w[] = {0.0, 0.2955242247, 0.2692667193,
                      0.2190863625, 0.1494513491, 0.0666713443};
  
  xm = 0.5 * (b+a); xr = 0.5 * (b-a);
  s = 0;
  for (j=1; j<=5; j++)
  {
    h = xr * x[j];
    s += w[j] * ( (*func)(xm+h) + (*func)(xm-h) );
  }

  return s *= xr; // scale
}






