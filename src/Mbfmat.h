// Mbfmat.h

#ifndef MBFMAT_H
#define MBFMAT_H

class Vector
 {
 public:
  Vector(void);
  Vector(const Vector &vec);
  Vector(int n);
  Vector(int n, double *v);
  ~Vector(void);
  
 private:
  int n;
  double *vec;
  char  *inf;
  int   ninf;

 private:
  int minmaxvalid;
  double minmax[2];
  void getminmax(void);

 public:
  int nel(void) {return n;}
  
 public:
  double mn();
  double mx();
  void copy(const double* f);
  
 public:
  double& operator[] (const int i) {return vec[i];};
  Vector operator+ (const double f);
  Vector operator* (const double f);
  Vector operator* (const double *f);
  Vector operator/ (const double *f);
  Vector& operator*= (const double *f);
  Vector& operator/= (const double *f);
  Vector& operator/= (const double f);
  Vector operator+ (const Vector &v2);
  Vector operator- (const Vector &v2);
  Vector& operator= (const Vector &vec);
  
  int save(const char filename[], int bin=1, char format[]=NULL);
  int saveRow(const char filename[], int bin=1, char format[]=NULL);

  void    cleanInfo  (void);
  char*   infoItem   (char *templ, int ind=0);
  bool	  addInfo    (const char *templ, char *value);
  
  friend class Matrix;
 };

class Matrix
 {
 public:
  Matrix(void);
  Matrix(const Matrix &m, int copy=1);
  Matrix(int nr, int nc);
  Matrix(int nr, int nc, double *d, int copy=1);
  ~Matrix(void);
  
 private:
  int nr, nc;
  double *mat;
  char  *inf;
  int   ninf;

 private:
  int freeable;
  int minmaxvalid;
  double minmax[2];
  void getminmax(void);

 public:
  int nrow(void) {return nr;}
  int ncol(void) {return nc;}
  
 public:
  double mn();
  double mx();
  
 public:
  double* operator[] (const int i) {return mat+nc*i;};
  Matrix operator+ (const double f);
  Matrix operator* (const double f);
  Matrix operator+ (const Matrix &m2);
  Matrix operator- (const Matrix &m2);
  Matrix operator* (const Matrix &m);
  Matrix& operator= (const Matrix &m);

  void confiscate(Matrix &m);
  int mul(const Matrix &m1, const Matrix &m2);
  Matrix tmul(const Matrix &m2);
  Matrix mult(const Matrix &m2);
  Matrix tmult(const Matrix &m2);

  Vector col(int i);
  Vector row(int i);

  double **offset(int c_offs=1, int r_offs=1);
  int save(const char filename[], int bin=1, char format[]=NULL);
  int load(const char filename[], bool trans=false,
	   int fr=-1, int lr=-1, int sr=-1, int fc=-1, int lc=-1, int sc=-1);

  void    cleanInfo  (void);
  char*   infoItem   (char *templ, int ind=0);
  bool	  addInfo    (const char *templ, char *value);
    
  bool checkInfNaN(void);
  Matrix invLU(double &condition);
  Matrix ludec(double &condition);
  Matrix solveLU(Matrix v);
  Matrix pseudInv(double maxRepErr, int &nInd, double &condition);
 };

#endif
