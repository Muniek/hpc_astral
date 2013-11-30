#include "mpi.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <iomanip>

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

void parallel(int size, int computations) {
  int nodes = MPI::COMM_WORLD.Get_size();
  int rows_oryg = ceil((double)size/nodes);
  int rows_halo;
  int rank = MPI::COMM_WORLD.Get_rank();
  vector< vector<double> > res_a;

  //first node
  if (rank == 0) {
    rows_halo = 1;
    res_a.resize(rows_oryg + rows_halo, vector<double>(size));
  //last node
  } else if(rank == nodes-1) {
    rows_oryg = size - rows_oryg * (nodes-1);
    rows_halo = 1;
    res_a.resize((rows_oryg + rows_halo), vector<double>(size));
  } else {//central nodes
    rows_halo = 2;
    res_a.resize(rows_oryg + rows_halo, vector<double>(size));
  }
  
  int rows_eff = rows_oryg + rows_halo;

  //zeroing the array
  for(int i = 0; i < rows_eff; i++) {
    for(int j =0; j < size; j++)  {
      res_a[i][j] = 0.0;
    }
  }

  double h = 1.0/(size - 1);

  //filling the x = 0 column with primary data
  for(int i = 0; i < rows_eff; i++)
    if (rank == 0)
      res_a[i][0] = pow(sin(M_PI * i * h), 2); 
    else
      res_a[i][0] = pow(sin(M_PI * h * (i + rank * rows_oryg - 1)), 2);

  //gauss-seidel iterations
  for(int ci = 0; ci < computations; ci++)
    for(int i = 1; i < rows_oryg; i++)
      for(int j = 1; j < size - 1; j++)
        res_a[i][j] = 0.25 * (res_a[i - 1][j] + res_a[i + 1][j] + res_a[i][j - 1] + res_a[i][j + 1]); 


  cout << setprecision(3);
  const int m = 1;
  if (rank == 0) {
    //printing the array
     
    for(int i = 0; i < rows_eff; i++) {
      for(int j = 0; j < size; j++)
        cout << res_a[i][j] << " ";
      cout << endl;
    }
    //sending message
    MPI::COMM_WORLD.Send(&m, 1, MPI_INT, (rank+1), 0);  
} else {
    int recv = 0;
    MPI::Status status;
    MPI::COMM_WORLD.Recv(&recv, 1, MPI_INT, rank-1, MPI::ANY_TAG, status); 

    while (!recv) { }

    if (recv) {
      //printing the array
      for(int i = 0; i < rows_eff; i++) {
        for(int j = 0; j < size; j++)
          cout << res_a[i][j] << " ";
        cout << endl;
      }
      if (rank != (nodes -1)) { 
        MPI::COMM_WORLD.Send(&m, 1, MPI_INT, (rank+1), 0);  
      }
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

main(int argc, char *argv[])  {
  MPI::Init(argc, argv);
  int n = 9;
  int computations = 100;
  parallel(n, computations);
  MPI::Finalize();
  return 0;
}
