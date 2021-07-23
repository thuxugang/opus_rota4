// Copyright Maarten L. Hekkelman, Radboud University 2008-2011.
// Copyright Coos Baakman, Jon Black, Wouter G. Touw & Gert Vriend, Radboud university medical center 2015.
//   Distributed under the Boost Software License, Version 1.0.
//       (See accompanying file LICENSE_1_0.txt or copy at
//             http://www.boost.org/LICENSE_1_0.txt)

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "mas.h"

#include "dssp.h"
#include "structure.h"

#include <boost/bind.hpp>
#include <boost/date_time/date_clock_device.hpp>
#include <boost/date_time/gregorian/gregorian.hpp>
#include <boost/foreach.hpp>
#include <boost/format.hpp>

#if defined(_MSC_VER)
#include <conio.h>
#include <ctype.h>
#endif
#include <iostream>


#define foreach BOOST_FOREACH


std::string ResidueToDSSPLine(const MResidue& residue)
{
/*
  This is the header line for the residue lines in a DSSP file:

  # AA STRUCTURE ACC
*/
  boost::format kDSSPResidueLine(
	  "%4.4d %1.1s %1.1s %4.4d");

  char code = kResidueInfo[residue.GetType()].code;
  if (residue.GetType() == kCysteine and residue.GetSSBridgeNr() != 0)
    code = 'a' + ((residue.GetSSBridgeNr() - 1) % 26);

  char ss;
  switch (residue.GetSecondaryStructure())
  {
    case alphahelix:  ss = 'H'; break;
    case betabridge:  ss = 'B'; break;
    case strand:    ss = 'E'; break;
    case helix_3:    ss = 'G'; break;
    case helix_5:    ss = 'I'; break;
    case turn:      ss = 'T'; break;
    case bend:      ss = 'S'; break;
    case loop:      ss = 'C'; break;
  }

  return (kDSSPResidueLine % residue.GetNumber() % code % ss
	  % floor(residue.Accessibility() + 0.5)).str();

}

void WriteDSSP(MProtein& protein, std::ostream& os)
{
  double accessibleSurface = 0;  // calculate accessibility as
  foreach (const MChain* chain, protein.GetChains())
  {
    foreach (const MResidue* residue, chain->GetResidues())
      accessibleSurface += residue->Accessibility();
  }

  // per residue information
  os << "# AA STRUCTURE ACC" << std::endl;

  std::vector<const MResidue*> residues;

  foreach (const MChain* chain, protein.GetChains())
  {
    foreach (const MResidue* residue, chain->GetResidues())
    {
      residues.push_back(residue);
    }
  }

  // keep residues sorted by residue number as assigned during reading the PDB file
  sort(residues.begin(), residues.end(), boost::bind(&MResidue::GetNumber, _1) < boost::bind(&MResidue::GetNumber, _2));

  //const MResidue* last = nullptr;
  foreach (const MResidue* residue, residues)
  {
    // insert a break line whenever we detect missing residues
    // can be the transition to a different chain, or missing residues in the current chain
    //if (last != nullptr and last->GetNumber() + 1 != residue->GetNumber())
    //{
    //  char breaktype = ' ';
    //  if (last->GetChainID() != residue->GetChainID())
    //    breaktype = '*';
    //  os << (kDSSPResidueLine % (last->GetNumber() + 1) % breaktype) << std::endl;
    //}
    os << ResidueToDSSPLine(*residue) << std::endl;
    //last = residue;
  }
}
