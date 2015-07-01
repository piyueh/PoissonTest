/**
 * @file misc.hpp
 * @brief Miscellaneous
 * @author Pi-Yueh Chuang
 * @version alpha
 * @date 2015-07-01
 */
# pragma once

# include "TriPoisson.hpp"
# include "typedef.hpp"

/**
 * @brief To modify the default setting of Teuchos::FancyOStream
 *
 * @param out The RCP pointer of an FancyOStream instance
 * @param rank The rank of current MPI process
 * @param size The size of MPI communicator
 */
void outInitSetting(RCP<FancyOStream> & out, const int rank, const int size)
{
	out->setTabIndentStr("  ");
	out->setShowLinePrefix(false);
	out->setMaxLenLinePrefix(10);
	out->setShowTabCount(false);
	out->setShowProcRank(true);
	out->setProcRankAndSize(rank, size);
	out->setOutputToRootOnly(0);
}

