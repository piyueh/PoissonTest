# include <iostream>
# include <cmath>

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

# include <Amesos2.hpp>

# include <Ifpack2_Factory.hpp>
# include <Ifpack2_Preconditioner.hpp>


using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::rcpFromRef;
using Teuchos::ArrayRCP;
using Teuchos::tuple;
using Teuchos::FancyOStream;
using Teuchos::ParameterList;
using Teuchos::CommandLineProcessor;
using Teuchos::Time;
using Teuchos::TimeMonitor;


typedef int 		LO_t;
typedef int 		GO_t;
typedef double 		SC_t;
typedef Kokkos::Compat::KokkosCudaWrapperNode 			ND_t;
//typedef Kokkos::Compat::KokkosSerialWrapperNode 			ND_t;
typedef Teuchos::Comm<int> 								COMM_t;
typedef Tpetra::Map<LO_t, GO_t, ND_t> 					MAP_t;
typedef Tpetra::Vector<SC_t, LO_t, GO_t, ND_t> 			VEC_t;
typedef Tpetra::CrsMatrix<SC_t, LO_t, GO_t, ND_t> 		SPM_t;
typedef Tpetra::MultiVector<SC_t, LO_t, GO_t, ND_t> 	MV_t;
typedef Tpetra::Operator<SC_t, LO_t, GO_t, ND_t> 		OP_t;
typedef RCP<const COMM_t> 								COMM_ptr_t;
typedef RCP<const MAP_t> 								MAP_ptr_t;
typedef RCP<FancyOStream> 								OUT_ptr_t;
typedef Amesos2::Solver<SPM_t, MV_t> 					SOLVER_t;


void generateXY(MAP_ptr_t &, RCP<VEC_t> &, RCP<VEC_t> &, GO_t, SC_t);
void generateRHS(MAP_ptr_t &, RCP<VEC_t> &, RCP<VEC_t> &, RCP<VEC_t> &, SC_t, SC_t);
void generateExactSoln(MAP_ptr_t &, RCP<VEC_t> &, RCP<VEC_t> &, RCP<VEC_t> &, SC_t);
void generateA(MAP_ptr_t &, RCP<SPM_t> &, GO_t);


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
	OUT_ptr_t 	out = Teuchos::VerboseObjectBase::getDefaultOStream();

	// adjust the properties of the fancy output
	out->setTabIndentStr("  ");
	out->setShowLinePrefix(false);
	out->setMaxLenLinePrefix(10);
	out->setShowTabCount(false);
	out->setShowProcRank(true);
	out->setProcRankAndSize(myRank, procNum);
	out->setOutputToRootOnly(0);

	// setup the domain and descritization information
	int 		Nx = 100;
	double 		n = 1.0;

	CommandLineProcessor 		CLP;
	
	CLP.setDocString("2D Poisson Solver using Trilinos.");

	CLP.setOption("Nx", &Nx, "Number of grid points on x direction. (Ny = Nx)");
	CLP.setOption("n", &n, "Wave number");

	CLP.parse(argc, argv);

	*out << "The final value of Nx: " << Nx << std::endl;
	*out << "The final value of n: " << n << std::endl;

	int 		N = Nx * Nx;
	double 		Lx = 1.0;
	double 		dx = Lx / double(Nx);
	
	// create a Map and obtain the number of nodes contained in current process
	MAP_ptr_t 	map = rcp(new MAP_t(N, 0, comm, Tpetra::GloballyDistributed));

	int 		localN = map->getNodeNumElements();


	// output the infomation of this map
	map->describe(*out);
	*out << std::endl;

	// definition of all variables
	RCP<VEC_t> 		x, y;
	RCP<VEC_t> 		p;
	RCP<VEC_t> 		p_exat;
	RCP<VEC_t> 		f;
	RCP<VEC_t> 		err;
	RCP<SPM_t> 		A;
	RCP<ParameterList> 	solverParams;

	// set x and y coordinates
	x = rcp(new VEC_t(map));
	y = rcp(new VEC_t(map));
	x->setObjectLabel("x coordinates");
	y->setObjectLabel("y coordinates");
	generateXY(map, x, y, Nx, dx);
	comm->barrier();
	x->print(std::cout);
	y->print(std::cout);


	// set unknowns p and its exact solution
	p = rcp(new VEC_t(map));
	p_exat = rcp(new VEC_t(map));
	p->setObjectLabel("pressure (unknowns)");
	p_exat->setObjectLabel("exact solution");
	generateExactSoln(map, p_exat, x, y, n);
	comm->barrier();
	p->print(std::cout);
	p_exat->print(std::cout);



	// set RHS = 2 * (4*pi*n)^2 * cos(2*n*pi*x) * cos(2*n*pi*y)
	f = rcp(new VEC_t(map));
	f->setObjectLabel("RHS");
	generateRHS(map, f, x, y, n, dx);
	comm->barrier();
	f->print(std::cout);


	err = rcp(new VEC_t(map));
	err->setObjectLabel("absolute error");
	err->print(std::cout);


	A = rcp(new SPM_t(map, 5, Tpetra::StaticProfile));
	A->setObjectLabel("coefficient matrix");
	generateA(map, A, Nx);
	comm->barrier();
	A->fillComplete();
	A->print(std::cout);


	RCP<const SPM_t> 	constA = rcpFromRef(*A);
	RCP<const VEC_t> 	constf = rcpFromRef(*f);

	RCP<SOLVER_t> 		solver = 
		Amesos2::create<SPM_t, MV_t>("Basker", constA, p, constf);
	solver->Teuchos::Describable::describe(std::cout);

	auto pp = solver->getValidParameters();
	pp->print();

	// create and start the timer
	RCP<Time> 	solveTime = TimeMonitor::getNewCounter("Wall-time of solve()");
	solveTime->enable();
	solveTime->start(true);
	// solve
	solver->solve();
	// stop timer
	solveTime->stop();


	err->update(1.0, *p, -1.0, *p_exat, 0);
	SC_t 	norm2 = err->norm2();

	TimeMonitor::summarize();

	*out << "\tL2 Norm of Errors: " << norm2 << std::endl;

	Tpetra::finalize();
	return 0;
}


void generateXY(MAP_ptr_t &map, RCP<VEC_t> &x, RCP<VEC_t> &y, GO_t N, SC_t dL)
{
	LO_t 		localN = map->getNodeNumElements();

	for(LO_t i=0; i<localN; ++i)
	{
		GO_t 	gN = map->getGlobalElement(i);
		GO_t 	gI = gN % N;
		GO_t 	gJ = gN / N;
		x->replaceLocalValue(i, (SC_t(gI) + 0.5) * dL);
		y->replaceLocalValue(i, (SC_t(gJ) + 0.5) * dL);
	}
}


void generateRHS(MAP_ptr_t &map, RCP<VEC_t> &f, 
				 RCP<VEC_t> &x, RCP<VEC_t> &y, SC_t n, SC_t dL)
{
	LO_t 		localN = map->getNodeNumElements();
	SC_t 		coef1 = 2 * M_PI * n;
	SC_t 		coef2 = 2 * coef1 * coef1 * dL * dL;

	ArrayRCP<const SC_t> 	xVw = x->get1dView();
	ArrayRCP<const SC_t> 	yVw = y->get1dView();
	ArrayRCP<SC_t> 	fVw = f->get1dViewNonConst();


	for(LO_t i=0; i<localN; ++i)
		fVw[i] = coef2 * std::cos(coef1 * xVw[i]) * std::cos(coef1 * yVw[i]);


	LO_t 		dIdx = map->getLocalElement(0);
	if (dIdx != -1)
		fVw[dIdx] = std::cos(coef1 * xVw[dIdx]) * std::cos(coef1 * yVw[dIdx]);
}


void generateExactSoln(MAP_ptr_t &map, RCP<VEC_t> &f, 
					   RCP<VEC_t> &x, RCP<VEC_t> &y, SC_t n)
{
	LO_t 		localN = map->getNodeNumElements();
	SC_t 		coef1 = 2 * M_PI * n;

	ArrayRCP<const SC_t> 	xVw = x->get1dView();
	ArrayRCP<const SC_t> 	yVw = y->get1dView();
	ArrayRCP<SC_t> 	fVw = f->get1dViewNonConst();

	for(LO_t i=0; i<localN; ++i)
		fVw[i] = std::cos(coef1 * xVw[i]) * std::cos(coef1 * yVw[i]);

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
			A->insertGlobalValues(gN, Teuchos::tuple<GO_t>(gN), Teuchos::tuple<SC_t>(1.)); 

		else if (gI == N - 1 && gJ == 0)
			A->insertGlobalValues(gN, Teuchos::tuple<GO_t>(gN-1, gN, gN+N), 
									 Teuchos::tuple<SC_t>(-1., 2., -1.)); 

		else if (gI == 0 && gJ == N - 1)
			A->insertGlobalValues(gN, Teuchos::tuple<GO_t>(gN-N, gN, gN+1), 
									 Teuchos::tuple<SC_t>(-1., 2., -1.)); 

		else if (gI == N - 1 && gJ == N - 1)
			A->insertGlobalValues(gN, Teuchos::tuple<GO_t>(gN-N, gN-1, gN), 
									 Teuchos::tuple<SC_t>(-1., -1., 2.)); 

		else if (gI == 0)
			A->insertGlobalValues(gN, Teuchos::tuple<GO_t>(gN-N, gN, gN+1, gN+N), 
									 Teuchos::tuple<SC_t>(-1., 3., -1., -1.)); 

		else if (gI == N - 1)
			A->insertGlobalValues(gN, Teuchos::tuple<GO_t>(gN-N, gN-1, gN, gN+N), 
									 Teuchos::tuple<SC_t>(-1., -1., 3., -1.)); 

		else if (gJ == 0)
			A->insertGlobalValues(gN, Teuchos::tuple<GO_t>(gN-1, gN, gN+1, gN+N), 
									 Teuchos::tuple<SC_t>(-1., 3., -1., -1.)); 

		else if (gJ == N - 1)
			A->insertGlobalValues(gN, Teuchos::tuple<GO_t>(gN-N, gN-1, gN, gN+1), 
									 Teuchos::tuple<SC_t>(-1., -1., 3., -1.)); 

		else
			A->insertGlobalValues(gN, Teuchos::tuple<GO_t>(gN-N, gN-1, gN, gN+1, gN+N), 
									 Teuchos::tuple<SC_t>(-1., -1., 4., -1., -1.)); 
	}
}	
