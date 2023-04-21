/*
Copyright 2021, University of Southern California

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef GRAPHBASIC_H
#define GRAPHBAIC_H
#include"types.h"
struct GraphBase
{
    VertexId num_of_vertices;
    EdgeId num_of_edges;
    Dimension num_feature_dimensions;
    Category num_labels;
} __attribute__((packed));
struct Edge
{
    VertexId src;
    VertexId dst;
    DataType norm_factor;
} __attribute__((packed));
struct Simple_Edge
{
    VertexId v;
    DataType data;
} __attribute__((packed));
struct InEdge
{
    VertexId src;
    DataType norm_factor;
} __attribute__((packed));
struct OutEdge
{
    VertexId dst;
    DataType norm_factor;
} __attribute__((packed));
struct EdgeList
{
    const Simple_Edge *ptx;
    EdgeId num_of_edges;
    VertexId point;
} __attribute__((packed));
struct OutEdgeList
{
    const OutEdge *ptx;
    EdgeId num_out_edges;
    VertexId point;
} __attribute__((packed));
struct InEdgeList
{
    const InEdge *ptx;
    EdgeId num_in_edges;
    VertexId point;
} __attribute__((packed));
struct FeatureVector
{
    DataType *data;
    Dimension vec_len;
} __attribute__((packed));

struct LabelVector
{ // in one-hot representation
    const DataType *data;
    Category vec_len;
} __attribute__((packed));
struct EdgeStruct
{
    VertexId src;
    VertexId dst;
    DataType norm_factor;
} __attribute__((packed));
#endif
