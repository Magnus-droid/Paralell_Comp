	#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <sys/time.h>
#include <cooperative_groups.h>

#include "../inc/argument_utils.h"
namespace cg = cooperative_groups;

// Convert 'struct timeval' into seconds in double prec. floating point
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)

typedef int64_t int_t;
typedef double real_t;

int_t
    M,
    N,
    max_iteration,
    snapshot_frequency;

real_t
    dt,
    *h_temp,
    *h_thermal_diffusivity,
    //1: Declare device side pointers to store host-side data.
    *d_temp,
    *d_thermal_diffusivity;

#define T(x,y)                      temp[(y) * (N + 2) + (x)]
#define THERMAL_DIFFUSIVITY(x,y)    thermal_diffusivity[(y) * (N + 2) + (x)]

#define cudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void __global__ time_step_kernel ( real_t *temp, real_t *thermal_diffusivity, int_t M, int_t N, real_t dt);
void __device__ boundary_condition (real_t *temp, int_t M, int_t N, int x, int y );
void domain_init ( void );
void domain_save ( int_t iteration );
void domain_finalize ( void );


int
main ( int argc, char **argv )
{
    OPTIONS *options = parse_args( argc, argv );
    if ( !options )
    {
        fprintf( stderr, "Argument parsing failed\n" );
        exit(1);
    }

    M = options->M;
    N = options->N;
    max_iteration = options->max_iteration;
    snapshot_frequency = options->snapshot_frequency;

    domain_init();

    struct timeval t_start, t_end;
    gettimeofday ( &t_start, NULL );

    for ( int_t iteration = 0; iteration <= max_iteration; iteration++ )
    {
        //6: Launch the time-step kernel using cooperative groups syntax

	void *kernelArgs[] = {
		(void*) &d_temp,
		(void*) &d_thermal_diffusivity,
		(void*) &M,
		(void*) &N,
		(void*) &dt,
	};

	dim3 blockSize ( 32, 32, 1 );
        int gridX = (unsigned int) ceil( (double)(N+2)/(double)blockSize.x );
	int gridY = (unsigned int) ceil( (double)(M+2)/(double)blockSize.y );
	dim3 gridSize ( gridX, gridY, 1 );
	cudaLaunchCooperativeKernel ( (void*) time_step_kernel, gridSize, blockSize, kernelArgs );
	cudaErrorCheck( cudaPeekAtLastError() );

        if ( iteration % snapshot_frequency == 0 )
        {
            printf (
                "Iteration %ld of %ld (%.2lf%% complete)\n",
                iteration,
                max_iteration,
                100.0 * (real_t) iteration / (real_t) max_iteration
            );

            //7: Copy data from device to host.
	    cudaMemcpy ( h_temp, d_temp, (M+2) * (N+2) * sizeof(real_t), cudaMemcpyDeviceToHost );
            domain_save ( iteration );
        }
    }

    gettimeofday ( &t_end, NULL );
    printf ( "Total elapsed time: %lf seconds\n",
            WALLTIME(t_end) - WALLTIME(t_start)
            );


    domain_finalize();

    exit ( EXIT_SUCCESS );
}


// 4: Make time_step() a cooperative CUDA kernel
//         where one thread is responsible for one grid point.
void __global__
time_step_kernel (real_t *temp, real_t *thermal_diffusivity, int_t M, int_t N, real_t dt )
{
    //define global thread index
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    cg::grid_group grid = cg::this_grid();

    boundary_condition (temp, M, N, x, y );

    if (x > 0 && x < N && y > 0 && y < M && (x + y) % 2 == 0) {
        real_t c, t, b, l, r, K, A, D, new_value;

        c = T(x, y);

        t = T(x - 1, y);
        b = T(x + 1, y);
        l = T(x, y - 1);
        r = T(x, y + 1);

        K = THERMAL_DIFFUSIVITY(x, y);
        A = -K * dt;
        D = 1.0f + 4.0f * K * dt;

        new_value = (c - A * (t + b + l + r)) / D;

        temp[y * (N + 2) + x] = new_value;
    }
    // black-tile threads go here after failing the if check and wait until red tiles are done computing
    grid.sync();
    //after the threads have been synced, the black tiles are computed
    if (x > 0 && x < N && y > 0 && y < M && (x + y) % 2 == 1) {
        real_t c, t, b, l, r, K, A, D, new_value;

        c = T(x, y);

        t = T(x - 1, y);
        b = T(x + 1, y);
        l = T(x, y - 1);
        r = T(x, y + 1);

        K = THERMAL_DIFFUSIVITY(x, y);
        A = -K * dt;
        D = 1.0f + 4.0f * K * dt;

        new_value = (c - A * (t + b + l + r)) / D;

        temp[y * (N + 2) + x] = new_value;
    }

    grid.sync();
}


//5: Make boundary_condition() a device function and
//         call it from the time_step-kernel.
//         Chose appropriate threads to set the boundary values.
void __device__
boundary_condition ( real_t *temp, int_t M, int_t N, int x, int y )
{
    //Apply bountry conditions if the provided global index (x, y) shows that the thread is a boundary thread.
    if ( x == 1 ) T(x - 1, y) = T(x + 1, y);
    if ( y == 1 ) T(x, y - 1) = T(x, y + 1);
    if ( x == N ) T(x + 1, y) = T(x - 1, y);
    if ( y == M ) T(x, y + 1) = T(x, y - 1);
}


void
domain_init ( void )
{
    h_temp = (real_t*) malloc ( (M+2)*(N+2) * sizeof(real_t) );
    h_thermal_diffusivity = (real_t*) malloc ( (M+2)*(N+2) * sizeof(real_t) );

    //2: Allocate device memory.
    cudaMalloc ( (void**) &d_temp, (M+2) * (N+2) * sizeof(real_t));
    cudaMalloc ( (void**) &d_thermal_diffusivity, (M+2) * (N+2) * sizeof(real_t));

    dt = 0.1;

    for ( int_t y = 1; y <= M; y++ )
    {
        for ( int_t x = 1; x <= N; x++ )
        {
            real_t temperature = 30 + 30 * sin((x + y) / 20.0);
            real_t diffusivity = 0.05 + (30 + 30 * sin((N - x + y) / 20.0)) / 605.0;

            h_temp[ y * (N+2) + x] = temperature;
            h_thermal_diffusivity[ y * (N+2) + x ] = diffusivity;
        }
    }

    //3: Copy data from host to device.
    cudaMemcpy ( d_temp,
		 h_temp,
		 (M+2) * (N+2) * sizeof(real_t),
		 cudaMemcpyHostToDevice
    );

    cudaMemcpy ( d_thermal_diffusivity,
		 h_thermal_diffusivity,
		 (M+2) * (N+2) * sizeof(real_t),
		 cudaMemcpyHostToDevice
    );
}


void
domain_save ( int_t iteration )
{
    int_t index = iteration / snapshot_frequency;
    char filename[256];
    memset ( filename, 0, 256*sizeof(char) );
    sprintf ( filename, "data/%.5ld.bin", index );

    FILE *out = fopen ( filename, "wb" );
    if ( ! out )
    {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        exit(1);
    }
    fwrite( h_temp, sizeof(real_t), (N + 2) * (M + 2), out );
    fclose ( out );
}


void
domain_finalize ( void )
{
    free ( h_temp );
    free ( h_thermal_diffusivity );

    //8: Free device memory.
    cudaFree ( d_temp );
    cudaFree ( d_thermal_diffusivity );
}
