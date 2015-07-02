# pragma once

# include "TriPoisson.hpp"
# include "typedef.hpp"

class LinSolver
{
	public:

		LinSolver() = default;

		virtual void initialize(RCP<const MAP_t> & map_);

		RCP<const MAP_t> 			map;

		RCP<VEC_t> 					x;
		RCP<VEC_t> 					rhs;
		RCP<SPM_t> 					A;

		RCP<ParameterList> 			solverParams;

	private:

		SolverFactory 				factory;
		RCP<SolverManager> 			solver;
		RCP<LinearProblem> 			linProblem;

}


void LinSolver::initialize(RCP<const MAP_t> & map_, std::string solverName)
{
	map = map_;

	rhs = rcp(new VEC_t(map));

	A = rcp(new SPM_t(map, 0, Tpetra::StaticProfile));

	solverParams = Teuchos::parameterList();

	solver = factory.create(solverName, solverParams);


}
