/**
 * @file TriPoisson.hpp
 * @brief All external headers
 * @author Pi-Yueh Chuang
 * @version alpha
 * @date 2015-07-01
 */

# pragma once

// c++ standard libraries
# include <iostream>
# include <cmath>

// Teuchos
# include <Teuchos_GlobalMPISession.hpp>
# include <Teuchos_VerboseObject.hpp>
# include <Teuchos_RCP.hpp>

// Tpetra
# include <Tpetra_DefaultPlatform.hpp>
# include <Tpetra_Map.hpp>
# include <Tpetra_Vector.hpp>
# include <Tpetra_MultiVector.hpp>
# include <Tpetra_CrsMatrix.hpp>

// Belos
# include <BelosSolverFactory.hpp>
# include <BelosTpetraAdapter.hpp>
