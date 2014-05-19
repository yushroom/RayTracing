/*
 *	FileName:	datatype.h
 *
 *	Programmer: Jiayin Cao
 */

#pragma once

#include "define.h"

//the data type for vector
typedef struct _float4
{
	union
	{
		struct{	float	r , g , b , a ; };
		struct{ float	x , y , z , w ; };
	};

public:
	//default constructor
	_float4();
	_float4( float x , float y , float z , float w );
	_float4( float* d );

	//some operators
	_float4 operator = ( const _float4& d );
	_float4 operator + ( const _float4& d );
	_float4 operator - ( const _float4& d );
	_float4 operator * ( const float factor );
	_float4 operator * ( const _float4& d );
	_float4 operator += ( const _float4& d );
	float& operator [] ( int index );

}_float4;

//some helper functions

//cross product
_float4	cross( const _float4& v1 , const _float4& v2 );

//vector mutiply float
_float4	operator * ( const float factor , _float4& v );

//dot product
float	dot( const _float4& v1 , const _float4& v2 );

//the length of the vector
float	length( const _float4& v );

//normalize the vector
void	normalize( _float4& v );	

//reflect direction
_float4	reflect( _float4& dir , _float4& normal );

//refraction direction
_float4	refract( _float4& dir , _float4& normal , float rate );

//clamp the vector
void	saturate( _float4& v );

//clamp the data
float	clamp( float d );
