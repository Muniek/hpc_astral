#include "mpi.h"
#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

//zeroing the given square array
void to_zero(vector< vector<double> > &res_a, int size) {
 for(int i = 0; i < size; i++)
   for(int j = 0; j < size; j++)
     res_a[i][j] = 0.0;
}

//fills given array with data get from serial iteration algorithm
void serial(vector< vector<double> > &res_a, int size, int computations) {
  double h = 1.0/(size - 1);
  to_zero(res_a, size);

  for(int i = 0; i < size - 1; i++)
    res_a[i][0] = pow(sin(M_PI * i * h), 2);

  for(int ci = 0; ci < computations; ci++)
    for(int i = 1; i < size - 1; i++)
      for(int j = 1; j < size - 1; j++)
        res_a[i][j] = 0.25 * (res_a[i - 1][j] + res_a[i + 1][j] + res_a[i][j - 1] + res_a[i][j + 1]); 
}

void parallel(int size) {
  int nodes = MPI::COMM_WORLD.Get_size();
  int rows = ceil((double)size/nodes);
  int rank = MPI::COMM_WORLD.Get_rank();
  vector< vector<double> > res_a;

  //first node
  if (rank == 0) {
    rows += 1;
    res_a.resize(rows, vector<double>(size));
  //last node
  } else if(rank == nodes-1) {
    rows = size - (ceil((double)nodes/rows));
    res_a.resize((rows + 1), vector<double>(size));
  } else {//central nodes
    rows += 2;
    res_a.resize(rows, vector<double>(size));
  }

  //zeroing the array
  for(int i = 0; i < rows; i++) {
    for(int j =0; j < size; j++)  {
      res_a[i][j] = 0.0;
    }
  }
}

//fills given array with data get from analytical algorithm
void analytical(vector< vector<double> > &res_a, int size, int computations) {
  double h = 1.0/(size - 1);
  to_zero(res_a, size);

  for(int i = 0; i < size - 1; i++) {
    for(int j = 0; j < size - 1; j++) {
      double x = j * h;
      double y = i * h;

      for(int ci = 1; ci < computations; ci++) {
        if (ci == 2) continue;

        double numerator = 4 * (-1 + cos(ci * M_PI)) * (1/sinh(ci * M_PI)) * sin(ci * M_PI * y) * sinh(ci * M_PI * (x - 1));
        double denominator = (M_PI * ( (-4) * ci + pow(ci, 3)));
        res_a[i][j] -= numerator / denominator;   
      } 
    }
  } 
}

double global_error(vector< vector<double> > &analytical_a, vector< vector<double> > &iterative_a, int size)  {
  double global_e = 0.0;
  for(int i = 0; i < size; i++) {
    for(int j = 0; j < size; j++) {
      global_e += abs(analytical_a[i][j] - iterative_a[i][j]);
    }
  }
  return global_e;
}

double average_error(vector< vector<double> > &analytical_a, vector< vector<double> > &iterative_a, int size)  {
  return global_error(analytical_a, iterative_a, size) / size;
}

main()  {
  int n = 50;
  int computations = 50;
  vector< vector<double> > res_serial(n, vector<double>(n));
  serial(res_serial, n, computations);
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < n; j++)
      cout << res_serial[i][j] << " ";
    cout << endl;
  }
  cout << endl;
  vector< vector<double> > res_analytical(n, vector<double>(n));
  analytical(res_analytical, n, 100);
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < n; j++)
      cout << res_analytical[i][j] << " ";
    cout << endl;
  } 
  cout << endl;
  cout << "Global error: " << global_error(res_analytical, res_serial, n) << endl;
  cout << "Average error: " << average_error(res_analytical, res_serial, n) << endl;
  return 0;
}
