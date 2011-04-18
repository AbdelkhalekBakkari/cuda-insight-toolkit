/*
** EclipseCompat.h
**
**  Created on: Jan 19, 2010
**      Author: lukep
*/

#ifndef ECLIPSECOMPAT_H_
#define ECLIPSECOMPAT_H_

#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __host__
#define __shared__
#endif

#endif /* ECLIPSECOMPAT_H_ */
