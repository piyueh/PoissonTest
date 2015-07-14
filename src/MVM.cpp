# include <iostream>
# include <cmath>

# include <Teuchos_XMLParameterListCoreHelpers.hpp>
# include <Teuchos_GlobalMPISession.hpp>
# include <Teuchos_CommandLineProcessor.hpp>
# include <Teuchos_VerboseObject.hpp>
# include <Teuchos_TimeMonitor.hpp>
# include <Teuchos_RCP.hpp>

# include <Tpetra_DefaultPlatform.hpp>
# include <Tpetra_Map.hpp>
# include <Tpetra_Vector.hpp>
# include <Tpetra_MultiVector.hpp>
# include <Tpetra_CrsMatrix.hpp>
# include <MatrixMarket_Tpetra.hpp>

# ifndef NODETYPE
# define NODETYPE Kokkos::Compat::KokkosCudaWrapperNode
# endif

using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::rcpFromRef;
using Teuchos::ArrayRCP;
using Teuchos::tuple;
using Teuchos::CommandLineProcessor;
using Teuchos::Time;
using Teuchos::TimeMonitor;

typedef int 		LO_t;
typedef int 		GO_t;
typedef double 		SC_t;
typedef NODETYPE	ND_t;
typedef Teuchos::Comm<int>									COMM_t;
typedef Tpetra::Map<LO_t, GO_t, ND_t>						MAP_t;
typedef Tpetra::Vector<SC_t, LO_t, GO_t, ND_t>				VEC_t;
typedef Tpetra::CrsMatrix<SC_t, LO_t, GO_t, ND_t>			SPM_t;
typedef Tpetra::MultiVector<SC_t, LO_t, GO_t, ND_t> 		MV_t;
typedef RCP<const COMM_t>									COMM_ptr_t;
typedef RCP<const MAP_t>									MAP_ptr_t;


void generateA(MAP_ptr_t &map, RCP<SPM_t> &A, GO_t N);


int main(int argc, char **argv)
{
	// Initialize Tpetra (MPI and Kokkos nodes)
	Tpetra::initialize(&argc, &argv);

	// Get global communicator
	COMM_ptr_t 	comm = Tpetra::getDefaultComm();

	// obtain the rank of current process and the size of communicator
	int 		myRank = comm->getRank();
	int 		procNum = comm->getSize();

	// set up fancy output, which can be used by Trilinos objects
	Teuchos::oblackholestream 		blackHole;
	std::ostream 			&out = (myRank == 0) ? std::cout : blackHole;

	// setup the domain and descritization information
	int 		Nx = 100;

	CommandLineProcessor 		CLP;
	CLP.setDocString("Trilinos MVM test.");
	CLP.setOption("Nx", &Nx, "Number of grid points on x direction. (Ny = Nx)");
	CLP.parse(argc, argv);

	out << "Final value of Nx: " << Nx << std::endl;

	int 		N = Nx * Nx;

	// create a Map and obtain the number of nodes contained in current process
	MAP_ptr_t 	map = rcp(new MAP_t(N, 0, comm, Tpetra::GloballyDistributed));

	// number of local entries
	LO_t 		localN = map->getNodeNumElements();

	// definition of all variables
	RCP<MV_t>			x, y;
	RCP<SPM_t>			A;

	// set x and y coordinates
	x = rcp(new MV_t(map, 1));
	y = rcp(new MV_t(map, 1));
	comm->barrier();

	// set sparse matrix A, a fully Neumann BC Poisson problem
	A = rcp(new SPM_t(map, 5, Tpetra::StaticProfile));
	A->setObjectLabel("coefficient matrix");
	generateA(map, A, Nx);
	comm->barrier();
	A->fillComplete();


	// initialize a timer
	RCP<Time> 	MVMTime = TimeMonitor::getNewCounter("Wall-time of MVM");

	for(int i=0; i<10; ++i)
	{
		x->randomize();
		y->randomize();
		comm->barrier();

		TimeMonitor 	localTimer(*MVMTime);
		// Matrix-Vector Multiplication
		A->apply(*x, *y);
		comm->barrier();
	}


	TimeMonitor::summarize(comm.ptr(), out, true, true, true, 
			Teuchos::Intersection, "", false);

	Tpetra::finalize();
	return 0;
}


void generateA(MAP_ptr_t &map, RCP<SPM_t> &A, GO_t N)
{
	LO_t 		localN = map->getNodeNumElements();

	for(LO_t i=0; i<localN; ++i)
	{
		GO_t 	gN = map->getGlobalElement(i);
		GO_t 	gI = gN % N;
		GO_t 	gJ = gN / N;

		if (gI == 0 && gJ == 0)
			A->insertGlobalValues(gN, tuple<GO_t>(gN, gN+1, gN+N),
									  tuple<SC_t>(3, -1, -1)); 

		else if (gI == N - 1 && gJ == 0)
			A->insertGlobalValues(gN, tuple<GO_t>(gN-1, gN, gN+N), 
									  tuple<SC_t>(-1., 2., -1.)); 

		else if (gI == 0 && gJ == N - 1)
			A->insertGlobalValues(gN, tuple<GO_t>(gN-N, gN, gN+1), 
									  tuple<SC_t>(-1., 2., -1.)); 

		else if (gI == N - 1 && gJ == N - 1)
			A->insertGlobalValues(gN, tuple<GO_t>(gN-N, gN-1, gN), 
									  tuple<SC_t>(-1., -1., 2.)); 

		else if (gI == 0)
			A->insertGlobalValues(gN, tuple<GO_t>(gN-N, gN, gN+1, gN+N), 
									  tuple<SC_t>(-1., 3., -1., -1.)); 

		else if (gI == N - 1)
			A->insertGlobalValues(gN, tuple<GO_t>(gN-N, gN-1, gN, gN+N), 
									  tuple<SC_t>(-1., -1., 3., -1.)); 

		else if (gJ == 0)
			A->insertGlobalValues(gN, tuple<GO_t>(gN-1, gN, gN+1, gN+N), 
									  tuple<SC_t>(-1., 3., -1., -1.)); 

		else if (gJ == N - 1)
			A->insertGlobalValues(gN, tuple<GO_t>(gN-N, gN-1, gN, gN+1), 
									  tuple<SC_t>(-1., -1., 3., -1.)); 

		else
			A->insertGlobalValues(gN, tuple<GO_t>(gN-N, gN-1, gN, gN+1, gN+N), 
									  tuple<SC_t>(-1., -1., 4., -1., -1.)); 
	}
}	
