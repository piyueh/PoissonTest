/**
 * @file poisson.cpp
 * @brief main function
 * @author Pi-Yueh Chuang
 * @version alpha
 * @date 2015-07-01
 */

# include "TriPoisson.hpp"
# include "typedef.hpp"
# include "misc.hpp"

int main(int argc, char **argv)
{

	// Initialize Tpetra (MPI and Kokkos nodes)
	Tpetra::initialize(&argc, &argv);

	// Get global communicator
	RCP<const COMM_t> 		comm = Tpetra::getDefaultComm();

	// obtain the rank of current process and the size of communicator
	int 					myRank = comm->getRank();
	int 					procNum = comm->getSize();

	// set up fancy output, which can be used by Trilinos objects under MPI
	RCP<FancyOStream> 		out = Teuchos::VerboseObjectBase::getDefaultOStream();
	outInitSetting(out, myRank, procNum);

	// adjust the properties of the fancy output



	// setup the domain and descritization information
	int 		Nx = 100;
	int 		Ny = 100;
	double 		Lx = 1.0;
	double 		Ly = 1.0;
	double 		dx = Lx / double(Nx);
	double 		dy = Ly / double(Ny);

	double 		n = 1.0;
	double 		coef_1 = - 8.0 * M_PI * M_PI * n * n;

	int 		N = Nx * Ny;
	
	// create a Map and obtain the number of nodes contained in current process
	Teuchos::RCP<const map_t> 		map = 
		Teuchos::rcp(new map_t(N, 0, comm, Tpetra::GloballyDistributed));

	int 				localN = map->getNodeNumElements();


	// output the infomation of this map
	map->describe(*out);
	*out << std::endl;

	// create a vector for coordinates of descretized space
	vector_t 			x_raw(map), y_raw(map);
	x_raw.setObjectLabel("x coordinates");
	y_raw.setObjectLabel("y coordinates");
	Teuchos::ArrayRCP<scalar_t> 	x = x_raw.get1dViewNonConst();
	Teuchos::ArrayRCP<scalar_t> 	y = y_raw.get1dViewNonConst();

	for(int i=0; i<localN; ++i)
	{
		int gN = map->getGlobalElement(i);
		int gI = gN % Nx;
		int gJ = gN / Nx;
		x[i] = (scalar_t(gI) + 0.5) * dx;
		y[i] = (scalar_t(gJ) + 0.5) * dy;
	}
	comm->barrier();

	for(int rank=0; rank<procNum; ++rank)
	{
		if (myRank == rank)
		{
			for(int i=0; i<localN; ++i)
			{
				int gN = map->getGlobalElement(i);
				int gI = gN % Nx;
				int gJ = gN / Nx;
				std::cout << myRank << " "
						  << i << " " << gN
						  << " (" << gI << ", " << gJ << ") "
						  << x[i] << " " << y[i] << std::endl;
			}
		}
		comm->barrier();
	}
	*out << std::endl;


	// create a vector for unknowns (assuem it is pressure)
	vector_t 			p_raw(map);
	p_raw.setObjectLabel("pressure");
	p_raw.putScalar(0.0);
	Teuchos::ArrayRCP<scalar_t> 	p = p_raw.get1dViewNonConst();

	// create a vector for RHS
	vector_t 			f_raw(map);
	f_raw.setObjectLabel("RHS");
	Teuchos::ArrayRCP<scalar_t> 	f = f_raw.get1dViewNonConst();

	for(int i=0; i<localN; ++i)
	{
		f[i] = - coef_1 * dx * dx * 
			std::cos(2 * M_PI * n * x[i]) * std::cos(2 * M_PI * n * y[i]);
	}
	comm->barrier();

	for(int rank=0; rank<procNum; ++rank)
	{
		if (myRank == rank)
		{
			for(int i=0; i<localN; ++i)
			{
				int gN = map->getGlobalElement(i);
				int gI = gN % Nx;
				int gJ = gN / Nx;
				std::cout << myRank << " "
						  << i << " " << gN
						  << " (" << gI << ", " << gJ << ") "
						  << f[i] << std::endl;
			}
		}
		comm->barrier();
	}
	*out << std::endl;


	// create a sparse matrix A for coefficients
	Teuchos::RCP<SPMtx> 		A = 
		Teuchos::rcp(new SPMtx(map, 5, Tpetra::StaticProfile));

	for(int i=0; i<localN; ++i)
	{
		int gN = map->getGlobalElement(i);
		int gI = gN % Nx;
		int gJ = gN / Nx;

		if (gI == 0 && gJ == 0)
		{
			A->insertGlobalValues(gN, 
					Teuchos::tuple<GO_t>(gN), 
					Teuchos::tuple<scalar_t>(1)); 

			f[gN] = std::cos(2 * M_PI * n * x[i]) * 
				std::cos(2 * M_PI * n * y[i]);
		}
		else if (gI == Nx - 1 && gJ == 0)
		{
			A->insertGlobalValues(gN, 
					Teuchos::tuple<GO_t>(gN-1, gN, gN+Nx), 
					Teuchos::tuple<scalar_t>(-1., 2., -1.)); 
		}
		else if (gI == 0 && gJ == Ny - 1)
		{
			A->insertGlobalValues(gN, 
					Teuchos::tuple<GO_t>(gN-Nx, gN, gN+1), 
					Teuchos::tuple<scalar_t>(-1., 2., -1.)); 
		}
		else if (gI == Nx - 1 && gJ == Ny - 1)
		{
			A->insertGlobalValues(gN, 
					Teuchos::tuple<GO_t>(gN-Nx, gN-1, gN), 
					Teuchos::tuple<scalar_t>(-1., -1., 2.)); 
		}
		else if (gI == 0)
		{
			A->insertGlobalValues(gN, 
					Teuchos::tuple<GO_t>(gN-Nx, gN, gN+1, gN+Nx), 
					Teuchos::tuple<scalar_t>(-1., 3., -1., -1.)); 
		}
		else if (gI == Nx - 1)
		{
			A->insertGlobalValues(gN, 
					Teuchos::tuple<GO_t>(gN-Nx, gN-1, gN, gN+Nx), 
					Teuchos::tuple<scalar_t>(-1., -1., 3., -1.)); 
		}
		else if (gJ == 0)
		{
			A->insertGlobalValues(gN, 
					Teuchos::tuple<GO_t>(gN-1, gN, gN+1, gN+Nx), 
					Teuchos::tuple<scalar_t>(-1., 3., -1., -1.)); 
		}
		else if (gJ == Ny - 1)
		{
			A->insertGlobalValues(gN, 
					Teuchos::tuple<GO_t>(gN-Nx, gN-1, gN, gN+1), 
					Teuchos::tuple<scalar_t>(-1., -1., 3., -1.)); 
		}
		else
		{
			A->insertGlobalValues(gN, 
					Teuchos::tuple<GO_t>(gN-Nx, gN-1, gN, gN+1, gN+Nx), 
					Teuchos::tuple<scalar_t>(-1., -1., 4., -1., -1.)); 
		}
	}
	comm->barrier();

	A->fillComplete();


	// exact solution
	vector_t 			extSoln_raw(map);
	extSoln_raw.setObjectLabel("Exact solution");
	Teuchos::ArrayRCP<scalar_t> 	extSoln = extSoln_raw.get1dViewNonConst();

	for(int i=0; i<localN; ++i)
	{
		extSoln[i] = std::cos(2 * M_PI * n * x[i]) * std::cos(2 * M_PI * n * y[i]);
	}
	comm->barrier();

	// create linear system and solver
	Teuchos::RCP<Teuchos::ParameterList> 	solverParams = Teuchos::parameterList();

	/*
	solverParams->set("Convergence Tolerance", 1e-12);
	solverParams->set("Maximum Iterations", 40000);
	solverParams->set("Assert Positive Definiteness", true);
	solverParams->set("Verbosity", 1);
	solverParams->set("Output Style", 1);
	solverParams->set("Output Frequency", 1);
	*/


	Belos::SolverFactory<scalar_t, MV_t, OP_t> 	factory;

	Teuchos::RCP<Belos::SolverManager<scalar_t, MV_t, OP_t>> 	solver = 
		factory.create("GMRES", solverParams);

	auto temp = solver->getCurrentParameters();
	temp ->print();

	Teuchos::RCP<Belos::LinearProblem<scalar_t, MV_t, OP_t>> 	problem = 
		Teuchos::rcp(new Belos::LinearProblem<scalar_t, MV_t, OP_t>
				(A, Teuchos::rcpFromRef(p_raw), Teuchos::rcpFromRef(f_raw)));

	problem->setProblem();

	solver->setProblem(problem);

	Belos::ReturnType 	result = solver->solve();

	if (result == Belos::Converged)
	{
		for(int rank=0; rank<procNum; ++rank)
		{
			if (myRank == rank)
			{
				for(int i=0; i<localN; ++i)
				{
					int gN = map->getGlobalElement(i);
					int gI = gN % Nx;
					int gJ = gN / Nx;
					std::cout << myRank << "\t"
							  << i << "\t" << gN
							  << " (" << gI << ", " << gJ << ")\t"
							  << p[i] << "\t" << extSoln[i] << std::endl;
				}
			}
			comm->barrier();
		}
		*out << solver->getNumIters() << std::endl;
		*out << std::endl;
	}
	else
	{
		*out << "Failed" << std::endl;
	}

	Tpetra::finalize();
	return 0;
}
