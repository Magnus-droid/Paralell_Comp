#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>

#include "../inc/argument_utils.h"

// Convert 'struct timeval' into seconds in double prec. floating point
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)

#define MPI_RANK_ROOT        ( rank == 0 )
#define HAS_TOP_ROW          ( coords[0] == 0 )
#define HAS_BOTTOM_ROW       ( coords[0] == dims[0] - 1 )
#define HAS_LEFTMOST_COL     ( coords[1] == 0 )
#define HAS_RIGHTMOST_COL    ( coords[1] == dims[1] - 1 )

typedef int64_t int_t;
typedef double real_t;


int_t
    M,
    N,
    max_iteration,
    snapshot_frequency;

MPI_Comm mpi_cart_comm;

int rank,
    size,
    dims[2],
    periods[2],
    coords[2],
    local_M,
    local_N;

real_t
    *temp[2] = { NULL, NULL },
    *thermal_diffusivity,
    dt;

#define T(x,y)                      temp[0][(y) * (local_N + 2) + (x)]
#define T_next(x,y)                 temp[1][((y) * (local_N + 2) + (x))]
#define THERMAL_DIFFUSIVITY(x,y)    thermal_diffusivity[(y) * (local_N + 2) + (x)]

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


int
main ( int argc, char **argv )
{
    // - Initialize MPI.
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    // - Create a cartesian communicator
    MPI_Dims_create( size, 2, dims );
    MPI_Cart_create( MPI_COMM_WORLD, 2, dims, periods, 0, &mpi_cart_comm );
    MPI_Cart_coords( mpi_cart_comm, rank, 2, coords );

    // - Parse arguments in the rank 0 processes
    //   and broadcast to the other processes
    if ( MPI_RANK_ROOT )
    {
        OPTIONS *options = parse_args( argc, argv );
        if ( !options )
        {
            fprintf( stderr, "Argument parsing failed\n" );
            exit(1);
        }

        N = options->N;
        M = options->M;
        max_iteration = options->max_iteration;
        snapshot_frequency = options->snapshot_frequency;
    }

    MPI_Bcast( &N, 1, MPI_INT64_T, 0, mpi_cart_comm );
    MPI_Bcast( &M, 1, MPI_INT64_T, 0, mpi_cart_comm );
    MPI_Bcast( &max_iteration, 1, MPI_INT64_T, 0, mpi_cart_comm );
    MPI_Bcast( &snapshot_frequency, 1, MPI_INT64_T, 0, mpi_cart_comm );

    printf("[Proccess %d located at coords (%d, %d)] dims %d x %d\n", rank, coords[0], coords[1], dims[0], dims[1]);


    domain_init();

    struct timeval t_start, t_end;
    gettimeofday ( &t_start, NULL );

    for ( int_t iteration = 0; iteration <= max_iteration; iteration++ )
    {
        border_exchange();

        boundary_condition();

        time_step();

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

    gettimeofday ( &t_end, NULL );
    printf ( "Total elapsed time: %lf seconds\n",
            WALLTIME(t_end) - WALLTIME(t_start)
            );

    domain_finalize();
    //maybe add MPI_Comm_free( mpi_cart_comm ); not sure?

    //finalize MPI
    MPI_Finalize();
    exit ( EXIT_SUCCESS );
}


void
time_step ( void )
{
    // iterate over the new subgrid size (local_M x local_N) that has been initialized in domain_init()

    real_t c, t, b, l, r, K, new_value;

    for ( int_t y = 1; y <= local_M; y++ )
    {
        for ( int_t x = 1; x <= local_N; x++ )
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

    // assuming that top left is the (0, 0) position in the grid

    // top row (for each x: y need to be adjusted)
    if ( HAS_TOP_ROW )
    {
	for ( int_t x = 1; x <= local_N; x++ )
	{
	    T(x, 0) = T(x, 2);
	}
    }

    // bottom row (for each x: y needs to be adjusted)
    if ( HAS_BOTTOM_ROW )
    {
	for ( int_t x = 1; x <= local_N; x++ )
	{
	    T(x, local_M+1) = T(x, local_M-1);
	}
    }

    //left column (for each y: x needs to be adjusted)
    if ( HAS_LEFTMOST_COL )
    {
	for (int_t y = 1; y <= local_M; y++ )
	{
	    T(0, y) = T(2, y);
	}
    }

    //right column (for each y: x needs to be adjusted)
    if ( HAS_RIGHTMOST_COL )
    {
	for ( int_t y = 1; y <= local_M; y++ )
	{
	    T(local_N+1, y) = T(local_N-1, y);
	}
    }

 }


void
border_exchange ( void )
{
    // This implementations assumes that 8 is the maximum amount of cores used.
    // If there were for example 16 cores, the logic would need to be a bit different
    // because then a process can have no global borders

    // BUG: The border exchange causes a deadlock and I suspect there is some behaviour of MPI_Sendrecv that I don't understand

    // determine ranks of procs
    int over, under, left, right;
    MPI_Cart_shift( mpi_cart_comm, 0, 1, &over, &under );
    MPI_Cart_shift( mpi_cart_comm, 1, 1, &left, &right );

    // Need one nw type for each kind of border there can be
    MPI_Datatype top_border_type, left_border_type, bottom_border_type, right_border_type;

    // Define the dimensions of the border regions
    int bigsizes[2] = { local_M + 2, local_N + 2 };

    int row_sizes[2]     = { 1, local_N     };
    int top_starts[2]    = { 0, 1           };
    int bottom_starts[2] = { local_M + 1, 1 };

    int col_sizes[2]     = { local_M, 1     };
    int left_starts[2]   = { 1, 0           };
    int right_starts[2]  = { 1, local_N + 1 };

    // Create and commit the new datatypes
    MPI_Type_create_subarray(2, bigsizes, row_sizes, top_starts, MPI_ORDER_C, MPI_DOUBLE, &top_border_type);
    MPI_Type_create_subarray(2, bigsizes, row_sizes, bottom_starts, MPI_ORDER_C, MPI_DOUBLE, &bottom_border_type);
    MPI_Type_create_subarray(2, bigsizes, col_sizes, left_starts, MPI_ORDER_C, MPI_DOUBLE, &left_border_type);
    MPI_Type_create_subarray(2, bigsizes, col_sizes, right_starts, MPI_ORDER_C, MPI_DOUBLE, &right_border_type);

    MPI_Type_commit(&top_border_type);
    MPI_Type_commit(&bottom_border_type);
    MPI_Type_commit(&left_border_type);
    MPI_Type_commit(&right_border_type);

    if (HAS_TOP_ROW)
    {
	// if the process has a global top border:
	// send local bottom row down, receive local top row from below
	MPI_Sendrecv( &temp[0][local_M * (local_N + 2)],
		      1,
		      bottom_border_type,
		      under,
		      0,
		      &temp[0][0],
		      1,
		      top_border_type,
		      under,
		      0,
		      mpi_cart_comm,
		      MPI_STATUS_IGNORE
	);
    }

    if (HAS_BOTTOM_ROW)
    {
	// if the process has a global bottom border:
	// send local top row up, receive local bottom row from above
	MPI_Sendrecv( &temp[0][0],
		      1,
		      top_border_type,
		      over,
		      1,
		      &temp[0][local_M * (local_N + 2)],
		      1,
		      bottom_border_type,
	              over,
		      1,
		      mpi_cart_comm,
		      MPI_STATUS_IGNORE
	);
    }

    if (HAS_LEFTMOST_COL)
    {
	// if the process has a global left border:
	// send local rightmost col to the right, receive local leftmost col from the rigth
	MPI_Sendrecv( &temp[0][local_N],
	              1,
	              right_border_type,
		      right,
		      2,
		      &temp[0][0],
		      1,
		      left_border_type,
	              right,
		      2,
		      mpi_cart_comm,
		      MPI_STATUS_IGNORE
	);
    }

    if (HAS_RIGHTMOST_COL)
    {
	// if the process has a global right border:
	// send local leftmost col to the left, receive local rigtmost col from the left
	MPI_Sendrecv( &temp[0][0],
		      1,
		      top_border_type,
		      left,
		      3,
		      &temp[0][local_N],
		      1,
		      bottom_border_type,
	              left,
		      3,
		      mpi_cart_comm,
		      MPI_STATUS_IGNORE
	);
    }

    // Free the MPI datatypes
    MPI_Type_free( &top_border_type    );
    MPI_Type_free( &bottom_border_type );
    MPI_Type_free( &left_border_type   );
    MPI_Type_free( &right_border_type  );
}


void
domain_init ( void )
{
    // calculate the size of each subgrid that is dependent on the dimentions (which in turn are dependent on how many procs)
    // calculate the x and y offsets which are based on the coordiantes of a process
    // allocate memeory for subgrid and calculate initial values with the x and y offset.
    //Assumption: N is x direction, M is y direction, dimentions are given as MxN

    local_M = M / dims[0];
    local_N = N / dims[1];
    int_t local_x_offset = coords[1] * local_N;
    int_t local_y_offset = coords[0] * local_M;

    temp[0] = malloc( ( local_M+2 ) * ( local_N+2 ) * sizeof(real_t) );
    temp[1] = malloc( ( local_M+2 ) * ( local_N+2 ) * sizeof(real_t) );
    thermal_diffusivity = malloc( ( local_M+2 ) * ( local_N+2 ) * sizeof(real_t) );
//    printf("[Proc: %d] x_offset = %d, y_offset = %d\n", rank, local_x_offset, local_y_offset);


    dt = 0.1;

    for ( int_t y = 1; y <= local_M; y++ )
    {
        for ( int_t x = 1; x <= local_N; x++ )
        {
            real_t temperature = 30 + 30 * sin((local_x_offset+x + local_y_offset+y) / 20.0);
            real_t diffusivity = 0.05 + (30 + 30 * sin((N - (local_x_offset+x) + local_y_offset+y) / 20.0)) / 605.0;

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

    // open file using the cartesian communicator in write-only mode,
    // or create it if it doesn't exist yet
    MPI_File out;
    MPI_File_open(
	mpi_cart_comm,
	filename,
	MPI_MODE_CREATE | MPI_MODE_WRONLY,
	MPI_INFO_NULL,
	&out
    );

    if ( ! out ) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        exit(1);
    }

    //create nwe datatype, similar to border_exchange()
    MPI_Datatype subarray;
    int starts[2]   = { 1, 1 };                       // Don't need the halo cells
    int subsizes[2] = { local_M, local_N };           // Size of the subarray
    int bigsizes[2] = { local_M + 2, local_N + 2 };   // Size of the global array

    MPI_Type_create_subarray(
        2,
        bigsizes,
        subsizes,
        starts,
        MPI_ORDER_C,
        MPI_DOUBLE,
        &subarray
    );

    MPI_Type_commit(&subarray);

    // Write to file using the new datatype

    MPI_File_set_view(out, 0, MPI_DOUBLE, subarray, "native", MPI_INFO_NULL);
    MPI_File_write_all(out, temp[0], (local_M + 2) * (local_N + 2), MPI_DOUBLE, MPI_STATUS_IGNORE);

    // free the datatype and close file after writing to it
    MPI_Type_free( &subarray );
    MPI_File_close( &out );

}


void
domain_finalize ( void )
{
    free ( temp[0] );
    free ( temp[1] );
    free ( thermal_diffusivity );
}
