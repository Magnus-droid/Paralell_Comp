#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <sys/time.h>
#include <pthread.h>

#include "../inc/argument_utils.h"

// Convert 'struct timeval' into seconds in double prec. floating point
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)
#define NUM_THREADS 2

typedef int64_t int_t;
typedef double real_t;

int_t
    M,
    N,
    max_iteration,
    snapshot_frequency;

real_t
    *temp[2] = { NULL, NULL },
    *thermal_diffusivity,
    dt;

struct
ThreadArgs
{
    int_t id;
    int_t start_row;
    int_t end_row;
};


#define T(x,y)                      temp[0][(y) * (N + 2) + (x)]
#define T_next(x,y)                 temp[1][((y) * (N + 2) + (x))]
#define THERMAL_DIFFUSIVITY(x,y)    thermal_diffusivity[(y) * (N + 2) + (x)]

void time_step ( void );
void boundary_condition( void );
void border_exchange( void );
void domain_init ( void );
void domain_save ( int_t iteration );
void domain_finalize ( void );


void
swap ( real_t** m1, real_t** m2 )
{
    real_t* tmp;
    tmp = *m1;
    *m1 = *m2;
    *m2 = tmp;
}


pthread_barrier_t
    barrier;

void* parallel_version ( void* args )
{
    struct ThreadArgs* thread_args = (struct ThreadArgs*)args;

    printf("THREAD %ld started work with start_row : end_row = %ld : %ld\n", thread_args->id, thread_args->start_row, thread_args->end_row);

    pthread_barrier_wait ( &barrier );
    for ( int_t iteration = 0; iteration <= max_iteration; iteration++)
    {


//boundary_condition()
        if ( thread_args->id == 0 )  //highest region
	{
            for ( int_t x = 1; x <= N; x++ )
            {
                T(x, 0) = T(x, 2);
	    }
	}

	if ( thread_args->id == NUM_THREADS - 1) //lowest region
	{
	    for ( int_t x = 1; x <= N; x++ )
	    {
                 T(x, M+1) = T(x, M-1);
	    }
        }

        for ( int_t y = thread_args->start_row; y <= thread_args->end_row; y++ )
        {
            T(0, y) = T(2, y);
            T(N+1, y) = T(N-1, y);
        }


//time_step()
        real_t c, t, b, l, r, K, new_value;

        for ( int_t y = thread_args->start_row; y <= thread_args->end_row; y++ )
        {
            for ( int_t x = 1; x <= N; x++ )
            {
                c = T(x, y);

                t = T(x - 1, y);
                b = T(x + 1, y);
                l = T(x, y - 1);
                r = T(x, y + 1);
                K = THERMAL_DIFFUSIVITY(x, y);

                new_value = c + K * dt * ((l - 2 * c + r) + (b - 2 * c + t));

                T_next(x, y) = new_value;
            }
        }

        if ( iteration % snapshot_frequency == 0 )
        {
            printf (
                "Iteration %ld of %ld (%.2lf%% complete)\n",
                iteration,
                max_iteration,
                100.0 * (real_t) iteration / (real_t) max_iteration
            );

            domain_save ( iteration );
        }

        swap( &temp[0], &temp[1] );

    }
    printf("THREAD %ld completed work\n", thread_args->id);
    return NULL;
}



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

    pthread_barrier_init( &barrier, NULL, NUM_THREADS );

    pthread_t threads[NUM_THREADS];
    struct ThreadArgs thread_args[NUM_THREADS];

    int_t rows_per_thread = M / NUM_THREADS;

    for ( int i = 0; i < NUM_THREADS; i++ )
    {
        thread_args[i].id = i;
        thread_args[i].start_row = i * rows_per_thread; //dont need border exchange beacause of this overlap that's 'supposed' to be solved with a barrier
	thread_args[i].end_row = (i+1) * rows_per_thread;

        if ( pthread_create( &threads[i], NULL, parallel_version, &thread_args[i]) != 0 )
        {
       	    perror("pthread_create");
            exit(EXIT_FAILURE);
        }
    }

    for ( int i = 0; i < NUM_THREADS; i++ )
    {
        if ( pthread_join(threads[i], NULL) !=0 )
        {
	    perror("pthread_join");
            exit(EXIT_FAILURE);
        }

    }

    pthread_barrier_destroy( &barrier );

    gettimeofday ( &t_end, NULL );
    printf ( "Total elapsed time: %lf seconds\n",
            WALLTIME(t_end) - WALLTIME(t_start)
            );


    domain_finalize();

    exit ( EXIT_SUCCESS );
}


void
time_step ( void )
{
    real_t c, t, b, l, r, K, new_value;

    for ( int_t y = 1; y <= M; y++ )
    {
        for ( int_t x = 1; x <= N; x++ )
        {
            c = T(x, y);

            t = T(x - 1, y);
            b = T(x + 1, y);
            l = T(x, y - 1);
            r = T(x, y + 1);
            K = THERMAL_DIFFUSIVITY(x, y);

            new_value = c + K * dt * ((l - 2 * c + r) + (b - 2 * c + t));

            T_next(x, y) = new_value;
        }
    }
}


void
boundary_condition ( void )
{
    for ( int_t x = 1; x <= N; x++ )
    {
        T(x, 0) = T(x, 2);
        T(x, M+1) = T(x, M-1);
    }

    for ( int_t y = 1; y <= M; y++ )
    {
        T(0, y) = T(2, y);
        T(N+1, y) = T(N-1, y);
    }
}


void
domain_init ( void )
{
    temp[0] = malloc ( (M+2)*(N+2) * sizeof(real_t) );
    temp[1] = malloc ( (M+2)*(N+2) * sizeof(real_t) );
    thermal_diffusivity = malloc ( (M+2)*(N+2) * sizeof(real_t) );

    dt = 0.1;

    for ( int_t y = 1; y <= M; y++ )
    {
        for ( int_t x = 1; x <= N; x++ )
        {
            real_t temperature = 30 + 30 * sin((x + y) / 20.0);
            real_t diffusivity = 0.05 + (30 + 30 * sin((N - x + y) / 20.0)) / 605.0;

            T(x,y) = temperature;
            T_next(x,y) = temperature;
            THERMAL_DIFFUSIVITY(x,y) = diffusivity;
        }
    }
}


void
domain_save ( int_t iteration )
{
    int_t index = iteration / snapshot_frequency;
    char filename[256];
    memset ( filename, 0, 256*sizeof(char) );
    sprintf ( filename, "data/%.5ld.bin", index );

    FILE *out = fopen ( filename, "wb" );
    if ( ! out ) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        exit(1);
    }

    fwrite( temp[0], sizeof(real_t), (N+2)*(M+2), out );
    fclose ( out );
}


void
domain_finalize ( void )
{
    free ( temp[0] );
    free ( temp[1] );
    free ( thermal_diffusivity );
}
