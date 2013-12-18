#include "mpi.h"
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <iomanip>

using namespace std;

void print_array(double** &res_a, int rows, int columns) {
  for(int i = 0; i < rows; i++) {
    for(int j = 0; j < columns; j++)
      cout << res_a[i][j] << " ";
    cout << endl;
  }
}

void to_zero(vector< vector<double> > &res_a, int rows, int columns) {
  for(int i = 0; i < rows; i++) {
      for(int j =0; j < columns; j++)  {
        res_a[i][j] = 0.0;
      }
    }
}

void gauss_seidel(double** &res_a, int rows, int columns) {
  for(int i = 1; i < rows - 1; i++)
    for(int j = 1; j < columns - 1; j++)
      res_a[i][j] = 0.25 * (res_a[i - 1][j] + res_a[i + 1][j] + res_a[i][j - 1] + res_a[i][j + 1]); 
}

void parallel(int size, int computations) {
  //total number of used nodes
  int nodes = MPI::COMM_WORLD.Get_size();
  //the number of current node
  int rank = MPI::COMM_WORLD.Get_rank();
  //the value of step
  double h = 1.0/(size - 1);
  //setting the boolean helpers
  //for fist node
  bool isFirst = false;
  if (rank == 0) isFirst = true;
  //for last node
  bool isLast = false;
  if (rank == nodes - 1) isLast = true;
  //for middle one
  bool isMiddle = !(isFirst || isLast);
  //the proper amount of rows for computing for current node
  //the -2 is because in whole computed array the first and last rows equals 0s (that's the condition)
  int comp_rows = ceil((double)(size - 2) / nodes); 
  //this amount if different for last row - it's just what's left
  if (isLast)
    comp_rows = (size - 2) - comp_rows * (nodes - 1); 
  //setting up the array
  //to comp_rows there are 2 extra (halo) ones,
  //where for first and last node, they're just an edges instead of being halo ones
  int all_rows = comp_rows + 2;
 
  double **res_a = (double**)malloc(all_rows * sizeof(double*));
  for(int i = 0; i < all_rows; i++)
    res_a[i] = (double*)malloc(size * sizeof(double));
  //zeroing the newly created array
  for(int i = 0; i < all_rows; i++)
    for(int j = 0; j < size; j++)
      res_a[i][j] = 0.0;
  //filling the x = 0 column with proper data - sin^2(PI * y) 
  for(int i = 0; i < all_rows; i++)
    res_a[i][0] = pow(sin(M_PI * h * (i + (rank * (ceil((double)(size - 2) / nodes))))), 2);  
  //the given condition - in last row it should be only 0s, despite the calculated, very small value (the error)
  if (isLast)
    res_a[all_rows - 1][0] = 0.0;
  //computing gauss-seidel operations 
  for(int ci = 0; ci < computations; ci++) {
    gauss_seidel(res_a, all_rows, size);
    //transfering the data between the nodes 
    for (int i = 0; i < nodes - 1; i++) {
      if (rank == i) {
        MPI::COMM_WORLD.Send(res_a[all_rows - 2], size, MPI_DOUBLE, (rank + 1), ci);
        MPI::COMM_WORLD.Recv(res_a[all_rows - 1], size, MPI_DOUBLE, (rank + 1), ci);  
      }
      
      if (rank == i + 1) {
        MPI::COMM_WORLD.Recv(res_a[0], size, MPI_DOUBLE, (rank - 1), ci);  
        MPI::COMM_WORLD.Send(res_a[1], size, MPI_DOUBLE, (rank - 1), ci); 
      }
    }
  }
  //waiting for every node to finish
  MPI::COMM_WORLD.Barrier();

  //print the array (test)
  cout << setprecision(3);
  const int m = 1;
  if (rank == 0) {
    print_array(res_a, all_rows, size);
    cout << endl;
    MPI::COMM_WORLD.Send(&m, 1, MPI_INT, (rank+1), 0);  
  } else {
    int recv = 0;
    MPI::Status status;
    MPI::COMM_WORLD.Recv(&recv, 1, MPI_INT, rank-1, MPI::ANY_TAG, status); 

    while (!recv) { }

    if (recv) {
      print_array(res_a, all_rows, size);
      if (rank != (nodes -1)) { 
        MPI::COMM_WORLD.Send(&m, 1, MPI_INT, (rank+1), 0);  
      }
      cout << endl;
    }
  }
}

//fills given array with data get from analytical algorithm
void analytical(vector< vector<double> > &res_a, int size, int computations) {
  double h = 1.0/(size - 1);
  to_zero(res_a, size, size);

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
  //the size of the matrix
  int n = atoi(argv[1]);
  //the number of iterations
  int computations = atoi(argv[2]);
  parallel(n, computations);

  MPI::Finalize();
  return 0;
}
