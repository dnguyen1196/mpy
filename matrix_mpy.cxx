// Matrix multiplication using C++ threads

//      c++ -std=c++11 -pthread -O2 matrix_mpy.cxx ee193_utils.cxx 

#include <iostream>
#include <sstream>
#include <thread>
#include <vector>
#include <mutex>
#include "bits.hxx"
#include "ee193_utils.hxx"

using namespace std;

mutex g_crit;                   // To lock our critical section in th_func()
mutex mut;                      // For printing.

///////////////////////////////
// Matrix is the main class that we use.
// It has methods to declare a matrix, allocate space, initialize it, do slow
// single-threaded matrix multiply, and printing support.
// It also has one fast matrix-multiply method that you'll write yourself.
///////////////////////////////
class Matrix {
    // The private members.
    vector<float> data; // Data is stored in a 1D vector.
    int _N;             // _Nx_N matrix (where _N = 2^LOG2_N).
    int nbits_per_dim;  // This is the LOG2_N we're given.
    int index (int r, int c) const { return ((r << this->nbits_per_dim) | c); }

public:
    Matrix (int nbits_per_dim); // Create a matrix, allocate its storage.
    int N() const { return (this->_N); }

    // Access an element (note that operator[] can only take 1 arg, not 2).
    float &operator() (int r,int c) {return(this->data[this->index(r,c)]);}
    float operator() (int r,int c) const {return(this->data[this->index(r,c)]);}

    bool operator== (const Matrix &other) const;        // Full equality check
    void compare (const Matrix &M2) const;              // Die on first mismatch
    // Initialize a matrix; to I, to random #s in [0,1], or cyclic ints.
    void init_identity();
    void init_random(float min, float max);
    void init_cyclic_order ();

    void mpy_dumb (const Matrix &A, const Matrix &B);   // 1 thread, unblocked
    // 1 thread, but blocked.
    void mpy1 (const Matrix &A, const Matrix &B, int BS);
    // multithreaded & blocked.
    void mpy2 (const Matrix &A, const Matrix &B, int BS, int n_procs);

    string row_str(int row) const;      // Print one matrix row to a string.
    string str() const;                 // Ditto for the entire matrix.
private:
    void compute_block_multiplication(const Matrix& A, const Matrix &B, int rb, int rc, int rk, int BS);

    void fill_block_multithreaded(const Matrix& A, const Matrix& B, int threadId, int BS, int numthreads);
};

Matrix::Matrix (int nbits_per_Dim) {
    this->nbits_per_dim = nbits_per_Dim;
    this->_N = (1<<nbits_per_dim);
    unsigned int n_elements = (1 << (nbits_per_dim+nbits_per_dim));
    this->data = vector<float> (n_elements);
}

bool Matrix::operator== (const Matrix &other) const {
    return (this->data == other.data);
}

// Like ==. But: on mismatch, prints the first mismatching element and dies.
void Matrix::compare (const Matrix &M2) const {
    for (int r=0; r<_N; ++r)
        for (int c=0; c<_N; ++c)
            if ((*this)(r,c) != M2(r,c))
            DIE ("M1["<<r<<","<<c<<"]="<<(*this)(r,c)
                      << ", M2["<<r<<","<<c<<"]="<<M2(r,c));
}

void Matrix::init_identity() {
    for (int r=0; r<_N; ++r)
        for (int c=0; c<_N; ++c)
            this->data[index(r,c)] = ((r==c)?1.0:0.0);
}

void Matrix::init_cyclic_order() {
    for (int r=0; r<_N; ++r)
        for (int c=0; c<_N; ++c)
            this->data[index(r,c)] = bit_get (r+c, this->nbits_per_dim-1, 0);
}

// Printing support.
string Matrix::row_str(int row) const {
    ostringstream os;
    os << "{";
    for (int c=0; c<_N; ++c)
        os << (c==0?"":", ") << (*this)(row,c);
    os << "}";
    return (os.str());
}
string Matrix::str() const {
    string s = "{";
    for (int r=0; r<_N; ++r)
        s += this->row_str(r);
    s += "}";
    return (s);
}

// Simple algorithm for multiplying two matrices.
void Matrix::mpy_dumb (const Matrix &A, const Matrix &B) {
    for (int r=0; r<_N; ++r) {
        for (int c = 0; c < _N; ++c) {
            float sum = 0.0;
            for (int k = 0; k < _N; ++k)
                sum += (A(r, k) * B(k, c));
            this->data[index(r, c)] = sum;
        }
    }
}


void Matrix::compute_block_multiplication(const Matrix &A, const Matrix &B, int rb, int kb, int cb, int BS) {
    for (int ri = 0; ri < BS; ri++) {
        for(int ki = 0; ki < BS; ki++) {
            int r = rb*BS + ri;
            int k = kb*BS + ki;

            for(int ci = 0; ci < BS; ci++) {
                int c = cb*BS + ci;
                if (k==0) {
                    this->data[this->index(r,c)] = A(r,k)*B(k,c);
                } else{
                    this->data[this->index(r,c)] += A(r,k)*B(k,c);
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////
// One thread, blocked. Loop order rB, kB, cB, r, k, c.
// This function is for you to write.
void Matrix::mpy1 (const Matrix &A, const Matrix &B, int BS) {
    int NBLK=_N/BS;
    // An NBLKxNBLK grid of blocks
    // Single threaded block-matrix multiplication
    // Loop through the block and insert into *this->data?
    for (int rb = 0; rb < NBLK; rb++) {
        for(int kb = 0; kb < NBLK; kb++) {
            for (int cb = 0; cb < NBLK; ++cb) {
                // Go through each block
                // Do matrix multiplication within this block
                this->compute_block_multiplication(A, B, rb, kb, cb, BS);
            }
        }
    }
}


void Matrix::fill_block_multithreaded(const Matrix &A, const Matrix &B, int threadId, int BS, int numthreads) {
    int NBLK = this->_N/BS;
    for(int i = threadId; i < NBLK*NBLK; i = i + numthreads) {
        // Find the block coordinates of the block we're filling in C
        int rB = i/NBLK;
        int cB = i % NBLK;
        for (int kB = 0; kB < NBLK; kB++) {
            this->compute_block_multiplication(A, B, rB, kB, cB, BS);
        }
    }
}

////////////////////////////////////////////////////////////////
// This function does multithreaded, blocked matrix multiplication
//      A, B: the input matrices
//      BS: block size; i.e., you should use blocks of BSxBS.
//      n_procs: how many processors to use.
// You must store the output in (*this), which already has its .data array
// allocated (but not necessarily cleared).
// Note that you can find out the size of the A, B and (*this) matrices by
// either looking at the _N member variable, or calling Matrix.N().
void Matrix::mpy2 (const Matrix &A, const Matrix &B, int BS, int n_procs) {
    vector<thread> threads;
    int NBLK=_N/BS; // An NBLKxNBLK grid of blocks
    // Launch as many threads as we are allowed to but each thread handling
    // some separate portion of the matrix
    // There will be rb * cb blocks to take care of so each thread will handle
    // NBLK**2/n_procs
    for(int i = 0; i < n_procs; i++) {
        threads.push_back(thread(&Matrix::fill_block_multithreaded, this, A, B, i, BS, n_procs));
    }
    for(auto &p : threads) {
        p.join();
    }
}

// Wrapper function around Matrix::mpy2(). It just runs ::mpy2() several times
// and checks how long it took.
static void run_mpy2 (int BS, int n_cores, const Matrix &a, const Matrix &b,
                      const Matrix &c, Matrix &d) {
    for (int i=0; i<4; ++i) {
        auto start = start_time();
        d.mpy2 (a, b, BS, n_cores);
        long int time = delta_usec (start);
        c.compare (d);
        cout<<"mpy2 with "<<n_cores<<" cores="<<(time/1000000.0)<<"sec"<<endl;
    }
}

main () {
    // Time mpy_dumb() for 1Kx1K.
    LOG ("Timing mpy_dumb() on 1Kx1K matrices");
    int LOG2_N=10;
    Matrix a(LOG2_N), b(LOG2_N), c(LOG2_N), d(LOG2_N);
    a.init_cyclic_order();
    b.init_identity();

    auto start = start_time();
    c.mpy_dumb (b, a);
    long int time = delta_usec (start);
    LOG ("1Kx1K mpy_dumb() took "<<(time/1000000.0)<<"sec");

    // Time mpy_dumb(), mpy1() and mpy2() for 2Kx2K.
    LOG2_N=11;
    a = Matrix(LOG2_N); b=Matrix(LOG2_N); c=Matrix(LOG2_N); d=Matrix(LOG2_N);
    a.init_cyclic_order();
    b.init_identity();
    //
    start = start_time();
    c.mpy_dumb (b, a);
    time = delta_usec (start);
    LOG ("2Kx2K mpy_dumb() took "<<(time/1000000.0)<<"sec");
//
    int BS = 128;
    for (int i=0; i<4; ++i) {
        auto start = start_time();
        d.mpy1 (a, b, BS);
        long int time = delta_usec (start);
        LOG ("2Kx2K mpy1 took "<<(time/1000000.0)<<"sec");
        c.compare (d);
    }

    // mpy2: using 1, 2, 4, 8 and 16 cores.
    run_mpy2 (BS,  1, a, b, c, d);      // Parameters are BS, # cores, matrices
    run_mpy2 (BS,  2, a, b, c, d);
//    run_mpy2 (BS,  4, a, b, c, d);
//    run_mpy2 (BS,  8, a, b, c, d);
//    run_mpy2 (BS, 16, a, b, c, d);
}
     
                                                                                                                  
