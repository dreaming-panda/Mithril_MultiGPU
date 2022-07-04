#ifndef TYPES_H
#define TYPES_H
#include<stdint.h>
typedef uint32_t VertexId;
typedef uint64_t EdgeId;
typedef float DataType; 
typedef int64_t lli;
typedef uint16_t ProcessorId;
typedef uint32_t Dimension;
typedef uint32_t Category;
enum AggregationType
{
    SUM,
    NORM_SUM,
    MAX,
    MIN,
    MEAN
};


#endif
