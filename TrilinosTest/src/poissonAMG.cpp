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

# include <BelosSolverFactory.hpp>
# include <BelosTpetraAdapter.hpp>

# include <Ifpack2_Factory.hpp>
# include <Ifpack2_Preconditioner.hpp>

# include <MueLu.hpp>
# include <MueLu_TpetraOperator.hpp>
# include <MueLu_CreateTpetraPreconditioner.hpp>

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
using Tpetra::MatrixMarket::Writer;

typedef int 		LO_t;
typedef int 		GO_t;
typedef double 		SC_t;
typedef NODETYPE	ND_t;
typedef Teuchos::Comm<int>									COMM_t;
typedef Tpetra::Map<LO_t, GO_t, ND_t>						MAP_t;
typedef Tpetra::Vector<SC_t, LO_t, GO_t, ND_t>				VEC_t;
typedef Tpetra::CrsMatrix<SC_t, LO_t, GO_t, ND_t>			SPM_t;
typedef Tpetra::MultiVector<SC_t, LO_t, GO_t, ND_t> 		MV_t;
typedef Tpetra::Operator<SC_t, LO_t, GO_t, ND_t>			OP_t;
typedef RCP<const COMM_t>									COMM_ptr_t;
typedef RCP<const MAP_t>									MAP_ptr_t;
typedef RCP<FancyOStream>									OUT_ptr_t;
typedef Belos::SolverFactory<SC_t, MV_t, OP_t>				SolverFactory;
typedef Belos::SolverManager<SC_t, MV_t, OP_t>				SolverManager;
typedef Belos::LinearProblem<SC_t, MV_t, OP_t>				LinearProblem;
typedef Belos::PseudoBlockCGSolMgr<SC_t, MV_t, OP_t> 		CGSolver_t;
typedef Ifpack2::Preconditioner<SC_t, LO_t, GO_t, ND_t>		PREC_t;


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

	std::string precName = "MueLu";
	std::string solverName = "CG";

	CommandLineProcessor 		CLP;
	
	CLP.setDocString("2D Poisson Solver using Trilinos.");

	CLP.setOption("Nx", &Nx, "Number of grid points on x direction. (Ny = Nx)");
	CLP.setOption("n", &n, "Wave number");
	CLP.setOption("precName", &precName, "Name of the preconditioner");
	CLP.setOption("solverName", &solverName, "Name of the solver");

	CLP.parse(argc, argv);

	*out << "Final value of Nx: " << Nx << std::endl;
	*out << "Final value of n: " << n << std::endl;
	*out << "Name of the preconditioner: " << precName << std::endl;
	*out << "Name of the solver: " << solverName << std::endl;

	int 		N = Nx * Nx;
	double 		Lx = 1.0;
	double 		dx = Lx / double(Nx);

	std::string 	precParamsXML = precName + "_Params.xml";
	std::string 	solverParamsXML = solverName + "_Params.xml";
	
	// create a Map and obtain the number of nodes contained in current process
	MAP_ptr_t 	map = rcp(new MAP_t(N, 0, comm, Tpetra::GloballyDistributed));

	int 		localN = map->getNodeNumElements();


	// output the infomation of this map
	map->describe(*out);

	// definition of all variables
	RCP<VEC_t>			x, y;
	RCP<VEC_t>			p;
	RCP<VEC_t>			p_exat;
	RCP<VEC_t>			f;
	RCP<VEC_t>			err;
	RCP<SPM_t>			A;

	RCP<ParameterList>	solverParams = 
							Teuchos::getParametersFromXmlFile(solverParamsXML);


	// set x and y coordinates
	x = rcp(new VEC_t(map));
	y = rcp(new VEC_t(map));
	x->setObjectLabel("x coordinates");
	y->setObjectLabel("y coordinates");
	generateXY(map, x, y, Nx, dx);
	comm->barrier();


	// set unknowns p and its exact solution
	p = rcp(new VEC_t(map));
	p_exat = rcp(new VEC_t(map));
	p->setObjectLabel("pressure (unknowns)");
	p_exat->setObjectLabel("exact solution");
	generateExactSoln(map, p_exat, x, y, n);
	comm->barrier();


	// set RHS = 2 * (4*pi*n)^2 * cos(2*n*pi*x) * cos(2*n*pi*y)
	f = rcp(new VEC_t(map));
	f->setObjectLabel("RHS");
	generateRHS(map, f, x, y, n, dx);
	comm->barrier();


	// initialize a vector for error
	err = rcp(new VEC_t(map));
	err->setObjectLabel("absolute error");


	// set sparse matrix A, a fully Neumann BC Poisson problem
	A = rcp(new SPM_t(map, 5, Tpetra::StaticProfile));
	A->setObjectLabel("coefficient matrix");
	generateA(map, A, Nx);
	comm->barrier();
	A->fillComplete();
	Writer<SPM_t>::writeSparseFile("matrixA.mtx", A, "A", 
			"Coefficient matrix of 2D Poisson problem with Neumann BCs");


	RCP<MueLu::TpetraOperator<SC_t, LO_t, GO_t, ND_t>> M = 
					MueLu::CreateTpetraPreconditioner(A, precParamsXML);


	// create problem set
	RCP<LinearProblem> 	problem = rcp(new LinearProblem (A, p, f));
	//problem->setLeftPrec(M); // set preconditioner
	//problem->setHermitian(); // let the sys. konw this is a symm sys.
	problem->setProblem(); // confirm the problem set


	// set up solver instance
	SolverFactory 			factory;
	RCP<SolverManager> 		solver = factory.create(solverName, solverParams);
	solver->setProblem(problem);


	// create and start the timer
	RCP<Time> 	solveTime = TimeMonitor::getNewCounter("Wall-time of solve()");
	solveTime->enable();
	solveTime->start(true);
	// solve the problem
	Belos::ReturnType 	result = solver->solve();
	// stop timer
	solveTime->stop();

	if (result == Belos::Converged)
		*out << "Success" << std::endl;
	else
		*out << "Failed" << std::endl;

	err->update(1.0, *p, -1.0, *p_exat, 0);
	SC_t 	norm2 = err->norm2();

	*out << "\tL2 Norm of Errors: " << norm2 << std::endl;
	*out << "\tNumber of Iterations: " 
		 << solver->getNumIters() << std::endl;

	TimeMonitor::summarize();

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
		fVw[dIdx] += std::cos(coef1 * xVw[dIdx]) * std::cos(coef1 * yVw[dIdx]);
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
