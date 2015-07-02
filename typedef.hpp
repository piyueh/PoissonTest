/**
 * @file typedef.hpp
 * @brief containning all type definitions
 * @author Pi-Yueh Chuang
 * @version alpha
 * @date 2015-07-01
 */

# pragma once

# include "TriPoisson.hpp"

// type definition of local ordinal index
typedef int 		LO_t;

// type definition of global ordinal index
typedef int 		GO_t;

// type definition of scalars
typedef double 		SC_t;

// type definition of exectutation node
typedef Kokkos::Compat::KokkosCudaWrapperNode 			ND_t;

// type definition of MPI communicator
typedef Teuchos::Comm<int> 								COMM_t;

// type definition of map for distributed vectors
typedef Tpetra::Map<LO_t, GO_t, ND_t> 					MAP_t;

// type definition of distributed vectors
typedef Tpetra::Vector<SC_t, LO_t, GO_t, ND_t> 			VEC_t;

// type definition of distributed sparse matrices
typedef Tpetra::CrsMatrix<SC_t, LO_t, GO_t, ND_t> 		SPM_t;

// type definition of distributed multi-vectors
typedef Tpetra::MultiVector<SC_t, LO_t, GO_t, ND_t> 	MV_t;

// type definition of distributed operators
typedef Tpetra::Operator<SC_t, LO_t, GO_t, ND_t> 		OP_t;

// solver factory of Belos
typedef Belos::SolverFactory<SC_t, MV_t, OP_t> 			SolverFactory;

// solver manager of Belos
typedef Belos::SolverManager<SC_t, MV_t, OP_t> 			SolverManager;

// type definition of linear system from Belos
typedef Belos::LinearProblem<SC_t, MV_t, OP_t> 			LinearProblem;



// in order to not use namespace
using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::rcpFromRef;
using Teuchos::ArrayRCP;
using Teuchos::FancyOStream;
using Teuchos::ParameterList;
