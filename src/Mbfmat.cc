// Mbfmat.cc

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// disable some of the less sensible Visual C++ warnings
#pragma warning( disable : 4305 )  // 'initializing' : truncation from 'const double' to 'double'
#pragma warning( disable : 4244 )  // 'initializing' : converion from 'double' to 'double'

#include "Mbfmat.h"
#include "trilib.h"

#ifndef FALSE
  #define FALSE 0
  #define TRUE  1
#endif

#ifndef min
  #define min(a,b) (((a) < (b)) ? (a) : (b))
  #define max(a,b) (((a) > (b)) ? (a) : (b))
#endif

/////// Vector class //////////////////////////////////////////////////

Vector::Vector(void)
 {
  minmaxvalid=0;
  n=0;
  vec=NULL;
  inf=NULL;
  ninf=0;
 }

Vector::Vector(int nel)
 {
  minmaxvalid=0;
  n=nel;
  vec=(double*) calloc(n, sizeof(double));
  inf=NULL;
  ninf=0;
 }
 
Vector::Vector(int nel, double *v)
 {
  int i;
  
  minmaxvalid=0;
  n=nel;
  vec=(double*) calloc(n, sizeof(double));
  
  for (i=0; i<n; i++)
    vec[i]=v[i];
  inf=NULL;
  ninf=0;
 }

Vector::Vector(const Vector &v)
 {
  int i;
  
  minmaxvalid=0;
  n=v.n;
  vec=(double*) calloc(n, sizeof(double));
  
  for (i=0; i<n; i++)
    vec[i]=v.vec[i];
  inf=NULL;
  ninf=0;
 }

Vector::~Vector(void)
 {
  if (vec)
    free(vec);
  if (inf)
    free(inf);
 }

//double& Vector::operator[] (const int i)
// {
//  return vec[i];
// }

Vector Vector::operator+ (const double f)
 {
  int i;
  Vector v(*this);
    
  for (i=0; i<n; i++)
    v.vec[i] += f;
  return v;
 }

Vector Vector::operator* (const double f)
 {
  int i;
  Vector v(*this);
    
  for (i=0; i<n; i++)
    v.vec[i] *= f;
  return v;
 }

Vector Vector::operator* (const double *f)
 {
  int i;
  Vector v(*this);
    
  for (i=0; i<n; i++)
    v.vec[i] *= f[i];
  return v;
 }

Vector Vector::operator/ (const double *f)
 {
  int i;
  Vector v(*this);
    
  for (i=0; i<n; i++)
    v.vec[i] /= f[i];
  return v;
 }

Vector& Vector::operator*= (const double *f)
 {
  int i;
    
  for (i=0; i<n; i++)
    vec[i] *= f[i];
  return *this;
 }

Vector& Vector::operator/= (const double *f)
 {
  int i;
    
  for (i=0; i<n; i++)
    vec[i] /= f[i];
  return *this;
 }

void Vector::copy(const double *f)
 {
  int i;
    
  for (i=0; i<n; i++)
    vec[i] = f[i];
  return;
 }

Vector& Vector::operator/= (const double f)
 {
  int i;
    
  for (i=0; i<n; i++)
    vec[i] /= f;
  return *this;
 }

Vector Vector::operator+ (const Vector &v2)
 {
  int i;
  Vector v(*this);
    
  for (i=0; i<n; i++)
    v.vec[i] += v2.vec[i];
  return v;
 }

Vector Vector::operator- (const Vector &v2)
 {
  int i;
  Vector v(*this);
    
  for (i=0; i<n; i++)
    v.vec[i] -= v2.vec[i];
  return v;
 }

Vector& Vector::operator= (const Vector &v)
 {
  int i;
  
  if (vec)
    free(vec);
  minmaxvalid=0;
  n=v.n;
  vec=(double*) calloc(n, sizeof(double));
  
  for (i=0; i<n; i++)
    vec[i]=v.vec[i];
  return *this;
 }

void Vector::getminmax(void)
 {
  int i;
  bool first=true;

//  minmax[0]=MAXFLOAT;
//  minmax[1]=-MAXFLOAT;
  for (i=1; i<n; i++)
   {
// whatch it! At some point I made an empty isfinite for Windows in trilib.h
    if (!isfinite(vec[i])) continue;
    if (first)
     {
      minmax[0]=vec[i];
      minmax[1]=vec[i];
      first=false;
      continue;
     }
    if (minmax[0]>vec[i]) minmax[0]=vec[i];
    if (minmax[1]<vec[i]) minmax[1]=vec[i];
   }
  minmaxvalid=1;
 }

double Vector::mn(void)
 {
  if (!minmaxvalid)
    getminmax();
  return minmax[0];
 }

double Vector::mx(void)
 {
  if (!minmaxvalid)
    getminmax();
  return minmax[1];
 }


int Vector::save(const char filename[], int bin, char format[])
 {
  int i, j, ret;
  double **m;
  char form[80]=" %7.4f", name[256];

  if (vec==NULL)
    return FALSE;
  if (format)
    strcpy(form, format);
  strcpy(name, filename);  // filename is const, name isn't

  m=matrix(1,n,1,1);
  for (i=0; i<n; i++)
    m[i+1][1]=vec[i];

  if (bin)
   {
    if (inf)
      ret=binputtailmatrix(name, m, n, 1, ninf, inf);
    else
      ret=binputmatrix(name, m, n, 1);
   }
  else
   {
    if (inf)
      ret=puttailmatrix(name, m, n, 1, format, ninf, inf);
    else
      ret=putmatrix(name, m, n, 1, format);
   }
  free_matrix(m, 1, n, 1, 1);
  return ret;
 }

int Vector::saveRow(const char filename[], int bin, char format[])
 {
  int i, j, ret;
  double **m;
  char form[80]=" %7.4f", name[256];

  if (vec==NULL)
    return FALSE;
  if (format)
    strcpy(form, format);
  strcpy(name, filename);  // filename is const, name isn't

  m=matrix(1,1,1,n);
  for (i=0; i<n; i++)
    m[1][i+1]=vec[i];

  if (bin)
   {
    if (inf)
      ret=binputmatrix(name, m, 1, n, ninf, inf);
    else
      ret=binputmatrix(name, m, 1, n);
   }
  else
   {
    if (inf)
      ret=putmatrix(name, m, 1, n, format, ninf, inf);
    else
      ret=putmatrix(name, m, 1, n, format);
   }
  free_matrix(m, 1, 1, 1, n);
  return ret;
 }

void Vector::cleanInfo(void)
 {
  int ip;
  for (ip=0; ip<ninf; ip++)
    if (inf[ip]=='\n')
      inf[ip]='\0';
 }

char* Vector::infoItem(char *templ, int ind)
 {
  if (ninf==0 || inf==NULL)
    return NULL;
  return find_tail_item(ninf, inf, templ, ind);
 }

bool Vector::addInfo(const char *templ, char *value)
 {
  return tail_add_item(&ninf, &inf, templ, value);
 }


/////// Matrix class //////////////////////////////////////////////////

Matrix::Matrix(void)
 {
  freeable=1;
  minmaxvalid=0;
  nr=nc=0;
  mat=NULL;
  inf=NULL;
  ninf=0;
 }

Matrix::Matrix(int nrow, int ncol)
 {
  freeable=1;
  minmaxvalid=0;
  nr=nrow;
  nc=ncol;
  inf=NULL;
  ninf=0;
  mat=(double*) calloc(nr*nc, sizeof(double));
 }
 
Matrix::Matrix(int nrow, int ncol, double *v, int copy)
 {
  int i;
  
  minmaxvalid=0;
  nr=nrow;
  nc=ncol;
  if (copy)
   {
    freeable=1;
    mat=(double*) calloc(nr*nc, sizeof(double));
    for (i=0; i<nr*nc; i++)
      mat[i]=v[i];
   }
  else
   {
    freeable=0;
    mat=v;
   }
  inf=NULL;
  ninf=0;
 }

Matrix::Matrix(const Matrix &m, int copy)
 {
  int i;
  
  minmaxvalid=0;
  nr=m.nr;
  nc=m.nc;
  if (copy)
   {
    freeable=1;
    mat=(double*) calloc(nr*nc, sizeof(double));
    for (i=0; i<nr*nc; i++)
      mat[i]=m.mat[i];
   }
  else
   {
    freeable=0;
    mat=m.mat;
   }
  inf=NULL;
  ninf=0;
 }

Matrix::~Matrix(void)
 {
  if (mat && freeable)
    free(mat);
  if (inf)
    free(inf);
 }

Matrix Matrix::operator+ (const double f)
 {
  int i;
  Matrix m=(*this);
    
  for (i=0; i<nr*nc; i++)
    m.mat[i] += f;
  return m;
 }

Matrix Matrix::operator* (const double f)
 {
  int i;
  Matrix m(*this);
    
  for (i=0; i<nr*nc; i++)
    m.mat[i] *= f;
  return m;
 }

Matrix Matrix::operator+ (const Matrix &m2)
 {
  int i;
  Matrix m(*this);
    
  for (i=0; i<nr*nc; i++)
    m.mat[i] += m2.mat[i];
  return m;
 }

Matrix Matrix::operator- (const Matrix &m2)
 {
  int i;
  Matrix m(*this);
    
  for (i=0; i<nr*nc; i++)
    m.mat[i] -= m2.mat[i];
  return m;
 }

Matrix Matrix::operator* (const Matrix &m2)
 {
  int i, j, k;
  double val;
  Matrix m(nr, m2.nc);
  
  if (nc!=m2.nr)
   {
    m=Matrix(0,0);
    return m;
   }
    
  for (i=0; i<nr; i++)
    for (j=0; j<m2.nc; j++)
     {
      val=0;
      for (k=0; k<nc; k++)
        val += mat[i*nc+k]*m2.mat[k*m2.nc+j];
      m.mat[i*m2.nc+j]=val;
     }
  return m;
 }

int Matrix::mul(const Matrix &m1, const Matrix &m2)
 {
  int i, j, k, nk;
  double val;
  
  if (m1.nc != m2.nr)
    return 0;

  if (mat && freeable)
    free(mat);
  else if (mat)
   {
    return 0;
   }
  minmaxvalid=0;

  nr=m1.nr;
  nc=m2.nc;
  nk=m1.nc;
  mat=(double*) calloc(nr*nc, sizeof(double));
  for (i=0; i<nr; i++)
    for (j=0; j<nc; j++)
     {
      val=0;
      for (k=0; k<nk; k++)
        val += m1.mat[i*nk+k]*m2.mat[k*nc+j];
      mat[i*nc+j]=val;
     }
  return 1;
 }

Matrix Matrix::tmul(const Matrix &m2)
 {
  int i, j, k;
  double val;
  Matrix m(nc, m2.nc);
  
  if (nr!=m2.nr)
   {
    m=Matrix(0,0);
    return m;
   }
    
  for (i=0; i<nc; i++)
    for (j=0; j<m2.nc; j++)
     {
      val=0;
      for (k=0; k<nr; k++)
        val += mat[k*nc+i]*m2.mat[k*m2.nc+j];
      m.mat[i*m2.nc+j]=val;
     }
  return m;
 }

Matrix Matrix::mult(const Matrix &m2)
 {
  int i, j, k;
  double val;
  Matrix m(nr, m2.nr);
  
  if (nc!=m2.nc)
   {
    m=Matrix(0,0);
    return m;
   }
    
  for (i=0; i<nr; i++)
    for (j=0; j<m2.nr; j++)
     {
      val=0;
      for (k=0; k<nc; k++)
        val += mat[i*nc+k]*m2.mat[j*m2.nc+k];
      m.mat[i*m2.nr+j]=val;
     }
  return m;
 }

Matrix Matrix::tmult(const Matrix &m2)
 {
  int i, j, k;
  double val;
  Matrix m(nc, m2.nr);
  
  if (nr!=m2.nc)
   {
    m=Matrix(0,0);
    return m;
   }
    
  for (i=0; i<nc; i++)
    for (j=0; j<m2.nr; j++)
     {
      val=0;
      for (k=0; k<nr; k++)
        val += mat[k*nc+i]*m2.mat[j*m2.nc+k];
      m.mat[i*m2.nr+j]=val;
     }
  return m;
 }

  
Matrix& Matrix::operator= (const Matrix &m)
 {
  int i;
  
  if (inf) free(inf);
  inf=NULL;
  ninf=0;
  if (mat && freeable)
   {
    if (nr!=m.nr || nc!=m.nc)
     {
      free(mat);
      mat=NULL;
     }
   }
  if (mat==NULL)
   {
    nr=m.nr;
    nc=m.nc;
    mat=(double*) calloc(nr*nc, sizeof(double));
   }
  minmaxvalid=0;
  for (i=0; i<nr*nc; i++)
    mat[i]=m.mat[i];
  return *this;
 }

void Matrix::confiscate (Matrix &m)
 {
  int i;
  
  if (inf) free(inf);
  inf=NULL;
  ninf=0;
  if (mat && freeable)
   {
    free(mat);
    mat=NULL;
   }
  nr=m.nr;
  nc=m.nc;
  mat=m.mat;
  minmaxvalid=m.minmaxvalid;
  minmax[0]=m.minmax[0];
  minmax[1]=m.minmax[1];
  m.nr=0;
  m.nc=0;
  m.mat=NULL;
 }

void Matrix::getminmax(void)
 {
  int i;
  bool first=true;

//  minmax[0]=-MAXFLOAT;
//  minmax[1]=MAXFLOAT;
//  minmax[0]=mat[0];
//  minmax[1]=mat[0];
  for (i=1; i<nr*nc; i++)
   {
// whatch it! At some point I made an empty isfinite for Windows in trilib.h
    if (!isfinite(mat[i])) continue;
    if (first)
     {
      minmax[0]=mat[i];
      minmax[1]=mat[i];
      first=false;
      continue;
     }
    if (minmax[0]>mat[i]) minmax[0]=mat[i];
    if (minmax[1]<mat[i]) minmax[1]=mat[i];
   }
  minmaxvalid=1;
 }

double Matrix::mn(void)
 {
  if (!minmaxvalid)
    getminmax();
  return minmax[0];
 }

double Matrix::mx(void)
 {
  if (!minmaxvalid)
    getminmax();
  return minmax[1];
 }

Vector Matrix::col(int i)
 {
  int j;
  Vector v(nr);
    
  for (j=0; j<nr; j++)
    v.vec[j] = mat[j*nc+i];
  return v;
 }

Vector Matrix::row(int i)
 {
  int j;
  Vector v(nc);
    
  for (j=0; j<nc; j++)
    v.vec[j] = mat[i*nc+j];
  return v;
 }

int Matrix::save(const char filename[], int bin, char format[])
 {
  int i, j, ret;
  double **m;
  char form[80]=" %7.4f", name[256];

  if (mat==NULL)
    return FALSE;
  if (format)
    strcpy(form, format);
  strcpy(name, filename);  // filename is const, name isn't

  m=matrix(1,nr,1,nc);
  for (i=0; i<nr; i++)
    for (j=0; j<nc; j++)
      m[i+1][j+1]=mat[i*nc+j];

  if (bin)
    ret=binputmatrix(name, m, nr, nc);
  else
    ret=putmatrix(name, m, nr, nc, format);
  free_matrix(m, 1, nr, 1, nc);
  return ret;
 }

int Matrix::load(const char filename[], bool trans,
		 int fr, int lr, int sr, int fc, int lc, int sc)
 {
  int i, j, nr1, nc1;
  double **m;
  char name[256];

  if (inf) free(inf);
  inf=NULL;
  ninf=0;
  nr=nc=0;
  if (mat)
   {
    free(mat);
    mat=NULL;
   }
  minmaxvalid=0;
  strcpy(name, filename);  // filename is const, name isn't
//  if (m=getmatrix(name, &nr1, &nc1))
//   {
//    if (fr<0) fr=0;
//    if (lr<=0) lr=nr1-1;
//    if (sr<=0) sr=1;
//    if (fc<0) fc=0;
//    if (lc<=0) lc=nc1-1;
//    if (sc<=0) sc=1;
//    if (lr<fr) lr=fr;
//    if (lc<fc) lc=fc;
//    nr=(min(nr1-1,lr)-min(nr1-1,fr))/sr+1;
//    nc=(min(nc1-1,lc)-min(nc1-1,fc))/sc+1;
//    mat=(double*) calloc(nr*nc, sizeof(double));
//    for (i=0; i<nr; i++)
//      for (j=0; j<nc; j++)
//        if (trans)m
//          mat[j*nr+i]=m[fr+i*sr+1][fc+j*sc+1];
//	else
//          mat[i*nc+j]=m[fr+i*sr+1][fc+j*sc+1];
//    free_matrix(m, 1, nr, 1, nc);
//    if (trans)
//     { i=nc; nc=nr; nr=i; }
//   }
  m=getmatrix(name, &nr1, &nc1, &ninf, &inf, NULL, NULL,
	      fr+1, lr+1, sr, fc+1, lc+1, sc, trans);
  if (m)
   {
    cleanInfo();
    nr=nr1;
    nc=nc1;
//    if (trans)
//     {
//      mat=(double*) calloc(nr*nc, sizeof(double));
//      for (i=0; i<nr; i++)
//	for (j=0; j<nc; j++)
//          mat[j*nr+i]=m[i+1][j+1];
//      free_matrix(m, 1, nr, 1, nc);
//      i=nc; nc=nr; nr=i;
//     }
//    else
      mat=&(m[1][1]);	// gaat niet goed voor MSDOS, maar dat doen we niet meer.
      free(m+1);
   }
  else
    return FALSE;
  return TRUE;
 }

void Matrix::cleanInfo(void)
 {
  int ip;
  for (ip=0; ip<ninf; ip++)
    if (inf[ip]=='\n')
      inf[ip]='\0';
 }

char* Matrix::infoItem(char *templ, int ind)
 {
  if (ninf==0 || inf==NULL)
    return NULL;
  return find_tail_item(ninf, inf, templ, ind);
 }

bool Matrix::addInfo(const char *templ, char *value)
 {
  return tail_add_item(&ninf, &inf, templ, value);
 }


bool Matrix::checkInfNaN(void)
 {
  int i;
  bool ok=true;

  for (i=0; i<nr*nc; i++)
    if (!isfinite(mat[i]))
     {
      mat[i]=0;
      ok=false;
     }
  
  return ok;
 }

double **Matrix::offset(int c_offs, int r_offs)
 {
  if (nr==0 || nc==0)
    return NULL;
  double **m=(double**)calloc(nr*nc, sizeof(double*));
  if (!m) return NULL;
  for (int i=0; i<nr; i++)
    m[i]=mat+i*nc-c_offs;
  m -= r_offs;
  return m;
 }

Matrix Matrix::invLU(double &condition)
 {
  condition=0;
  if (nr!=nc)
   {
    fprintf(stderr, "\nMbfmat: Only square matrices can be inverted by LU\n\n");
    return Matrix(0,0);
   }

  Matrix a(*this), inv(nr,nr);
  double **b=a.offset(), *v;
  int ir, ic;

  ::ludec(b, nr, &condition);
  v=(double*)calloc(nr, sizeof(double));
  v--;
  for (ir=1; ir<=nr; ir++)
   {
    for (ic=1; ic<=nc; ic++)
      v[ic]=0;
    v[ir]=1;
    solvelu(b, nr, v);
    for (ic=1; ic<=nc; ic++)
      inv[ic-1][ir-1]=v[ic];
   }
  free(b+1);
  free(v+1);

  return inv;
 }

Matrix Matrix::ludec(double &condition)
 {
  condition=0;
  if (nr!=nc)
   {
    fprintf(stderr, "\nMbfmat: Only square matrices can be LU-decomposed\n\n");
    return Matrix(0,0);
   }

  Matrix a(*this);
  double **b=a.offset();

  ::ludec(b, nr, &condition);
  return a;
 }

Matrix Matrix::solveLU(Matrix v)
 {
  Matrix sol(v);
  double **b=v.offset();
  double **s=sol.offset();

//  ::solvelu(b, nrow, s);    

  return sol;
 }

Matrix Matrix::pseudInv(double maxRepErr, int &nInd, double &condition)
 {
  condition=0;

  Matrix a(*this), inverse(nc,nr);
  double **u=a.offset(), **inv=inverse.offset(), **v;
  double val, *sigma, totsqr, thissqr, lsig, ssig;
  int i, j, k, nsing, nout=0;

  nsing=nr>nc?nr:nc;
  sigma=vector(1, nsing);
  v=matrix(1, nc, 1, nc);
  svdcmp(u, nr, nc, sigma, v);

  totsqr=0;
  for (i=1; i<=(nr>nc?nc:nr); i++)
    totsqr += sigma[i]*sigma[i];
  thissqr=totsqr;

  lsig=ssig=sigma[1];
  for (i=1; i<=nsing; i++)
   {
    thissqr -= sigma[i]*sigma[i];
    if (thissqr<0)
      thissqr=0;
    lsig=sigma[i];
    if (sigma[i]>=0)
      sigma[i]=1/sigma[i];
    if (sqrt(thissqr/totsqr)<maxRepErr)
      break;
   }

  for (i++; i<=nsing; i++)
   {
    nout++;
    sigma[i]=0;
   }

  nInd=nsing-nout;
  condition=ssig/lsig;
  for (i=1; i<=nr; i++)
    for (j=1; j<=nc; j++)
      u[i][j] *= sigma[j];
  for (i=1; i<=nc; i++)
    for (j=1; j<=nr; j++)
     {
      val=0;
      for (k=1; k<=nc; k++)
        val += v[i][k]*u[j][k];
      inv[i][j]=val;
     }

  free(u+1);
  free(inv+1);
  free_matrix(v, 1, nc, 1, nc);
  free_vector(sigma, 1, nsing);

  return inverse;
 }
