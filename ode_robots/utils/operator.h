/***************************************************************************
 *   Copyright (C) 2005-2011 LpzRobots development team                    *
 *    Georg Martius  <georg dot martius at web dot de>                     *
 *    Frank Guettler <guettler at informatik dot uni-leipzig dot de        *
+ *    Frank Hesse    <frank at nld dot ds dot mpg dot de>                  *
 *    Ralf Der       <ralfder at mis dot mpg dot de>                       *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 *                                                                         *
 ***************************************************************************/
#ifndef __OPERATOR_H
#define __OPERATOR_H

#include "globaldata.h"

namespace lpzrobots {
  /**
     An Operator observes an agent (robot) and manipulates it if necessary.
     For instance if the robot is falled over the operator can flip it back.
     This is an abstract base class and subclasses should overload at least
     observe()
   */

  class OdeAgent;

  class Operator {
  public:
    /** type of manipulation of the robot (for display) and or operation
        RemoveOperator means that the operator should be removed
     */
    enum ManipType {None, Limit, Move, RemoveOperator};
    struct ManipAction {
      ManipType type;
      Pos pos;
    };
    
    Operator(){      
    }

    virtual ~Operator(){      
    }

    /** called every simulation step
        @return what was done with the robot
     */
    virtual ManipAction observe(OdeAgent* agent, GlobalData& global) = 0;

  };

}

#endif